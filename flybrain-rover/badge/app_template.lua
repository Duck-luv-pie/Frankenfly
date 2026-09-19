--[==[badge-app
slug=flybadge
name=FlyBadge
icon=FLY
api=2
heap_kb=96
wake_lock=1
version=1.0
author=FlyBrain Rover
]==]

-- FlyBadge -- a real fruit fly's escape circuit, running on your badge.
--
-- The neurons and synapses below are not a model of a fly and not a trained network. They are measured
-- anatomy: a subsample of the male Drosophila CNS connectome (neuPrint male-cns:v1.0, Berg et al. 2026).
-- LC4 and LPLC2 are the looming detectors in the fly's optic lobe. They converge on the giant fibre, the
-- single neuron whose spike launches an escape jump. That convergence is what fires when you swat at a fly
-- and miss, and it is wired here exactly as it is wired in the animal, with the real synapse counts and
-- the real signs.
--
--   swat at the badge   -> the accelerometer sees it -> current into LC4 and LPLC2
--   giant fibre spikes  -> every LED flashes and the fly jumps
--   B                   -> zero the outgoing synapses of LC4 and LPLC2 (a lesion). Swat all you like.
--   A                   -> put them back
--   UP / DOWN           -> sensitivity, saved between runs
--
-- There is no flinch rule anywhere in this file. Deleting the lesion check would not leave a hidden
-- reflex behind, because the escape only ever came out of the wiring.
--
-- The membrane equation runs in integers: the ESP32-C3 has no floating point unit, so one unit is 1/256
-- of a millivolt and the decay factors are 8-bit fractions. badge/sim_fixed.py runs this same arithmetic
-- on a laptop against this same circuit file, which is how the constants were chosen.

--@CIRCUIT@

-- fixed point: 1 unit = 1/256 mV, one step = 10 ms. Must match badge/sim_fixed.py exactly.
local V_REST, V_TH = -13312, -11520
local SYN_DECAY = 35            -- exp(-dt/tau_syn) = exp(-10/5) as a 256ths fraction
local GAIN_Q = 75               -- one int8 weight unit in membrane units
local REFRAC = 1                -- one 10 ms step
local STEPS_PER_TICK = 2        -- ticks are nominally 20 ms, the brain runs at 100 Hz
local CUR_MAX = 4000            -- membrane units; threshold is about 1750
local FLASH_MS, JUMP_MS, FLIRT_MS = 160, 260, 3000

local byte, floor, abs = string.byte, math.floor, math.abs
local V, G, REF, SYN = {}, {}, {}, {}
local spk, spk_n = {}, 0
local lesion, gain = false, 3
local loom, last = 0, nil
local gf_count, gf_rate, win_start = 0, 0, 0
local flash_until, jump_until, flirt_until, last_beacon = 0, 0, 0, 0
local tick_ms, led_state = 0, ""
local ui = {}
local radio_ok = false

local function reset_brain()
  for i = 1, C.n do V[i] = V_REST; G[i] = 0; REF[i] = 0; SYN[i] = 0 end
  spk, spk_n = {}, 0
end

-- One 10 ms step of leaky integrate-and-fire over the real connectome. Event driven: only the neurons
-- that fired last step push current, so the cost is spikes times out-degree, not the whole matrix.
local function step(cur, flirt)
  local syn, colptr, post, w = SYN, C.colptr, C.post, C.w
  for i = 1, C.n do syn[i] = 0 end
  for k = 1, spk_n do
    local i = spk[k]
    if not (lesion and i <= C.eye_last) then
      for e = colptr[i] + 1, colptr[i + 1] do
        local wv = byte(w, e)
        if wv > 127 then wv = wv - 256 end
        local p = byte(post, e)
        syn[p] = syn[p] + wv * GAIN_Q
      end
    end
  end
  local n_new, fired = 0, {}
  local eye_last, gf_first = C.eye_last, C.gf_first
  local lc_a, lc_b = C.lc10a_first, C.lc10a_last
  local gf_now = 0
  for i = 1, C.n do
    local g = floor(G[i] * SYN_DECAY / 256) + syn[i]
    G[i] = g
    if REF[i] > 0 then
      REF[i] = REF[i] - 1
      V[i] = V_REST
    else
      local inj = 0
      if i <= eye_last then inj = cur
      elseif flirt and i >= lc_a and i <= lc_b then inj = 2400 end
      local v = V[i] + floor((V_REST - V[i] + g + inj) / 2)
      if v >= V_TH then
        V[i] = V_REST
        REF[i] = REFRAC
        n_new = n_new + 1
        fired[n_new] = i
        if i >= gf_first then gf_now = gf_now + 1 end
      else
        V[i] = v
      end
    end
  end
  spk, spk_n = fired, n_new
  return gf_now
end

local function set_leds(r, g, b)
  badge.led.set_all(r, g, b)
  badge.led.show()
end

function on_enter(root)
  gain = badge.store.get_int("gain", 3)
  reset_brain()

  local bg = badge.ui.box(root, 320, 240)
  bg:style({ bg_color = 0x000000, border_width = 0 })
  bg:set_pos(0, 0)

  local title = badge.ui.label(root, "FlyBadge")
  title:style({ text_color = 0xFF715B, text_font = 20 })
  title:align("top_left", 10, 8)

  ui.sub = badge.ui.label(root, C.n .. " real neurons   " .. C.n_edges .. " real synapses")
  ui.sub:style({ text_color = 0x888888, text_font = 14 })
  ui.sub:align("top_left", 10, 32)

  ui.fly = badge.ui.label(root, "***")
  ui.fly:style({ text_color = 0xFFFFFF, text_font = 24 })
  ui.fly:set_pos(140, 110)

  ui.gf = badge.ui.label(root, "giant fibre  0 Hz")
  ui.gf:style({ text_color = 0xFFFFFF, text_font = 20 })
  ui.gf:align("top_left", 10, 60)

  ui.state = badge.ui.label(root, "INTACT")
  ui.state:style({ text_color = 0x44DD88, text_font = 18 })
  ui.state:align("top_left", 10, 86)

  ui.drive = badge.ui.bar(root, 0, CUR_MAX, 0)
  ui.drive:set_size(300, 10)
  ui.drive:set_pos(10, 160)
  ui.drive:style({ bg_color = 0x222222 })

  ui.perf = badge.ui.label(root, "")
  ui.perf:style({ text_color = 0x666666, text_font = 14 })
  ui.perf:align("bottom_left", 10, -26)

  ui.hint = badge.ui.label(root, "swat me    B lesion    A restore    UP/DOWN gain")
  ui.hint:style({ text_color = 0x888888, text_font = 14 })
  ui.hint:align("bottom_left", 10, -8)

  badge.led.clear(); badge.led.show()
  led_state = "o"
  win_start = badge.sys.ms()

  radio_ok = badge.radio.enable()
  if radio_ok then
    badge.radio.on_recv(function(mac, rssi, payload)
      if payload == "FLY1" then flirt_until = badge.sys.ms() + FLIRT_MS end
    end)
  end
end

function on_tick()
  local t0 = badge.sys.ms()

  -- Looming, from the only sensor the badge has. There is no light sensor on this hardware, so the
  -- stimulus is the swat itself: a hand coming down moves the badge, and the accelerometer reports it.
  -- Per-axis change catches a rotation as well as a shove, which a magnitude alone would miss.
  local x, y, z = badge.sensor.accel()
  local jerk = 0
  if x then
    if last then jerk = abs(x - last[1]) + abs(y - last[2]) + abs(z - last[3])
    else last = {} end
    last[1], last[2], last[3] = x, y, z
  end
  if badge.sensor.shake() then jerk = jerk + 400 end
  loom = floor(loom * 3 / 4) + jerk        -- leaky, so one swat drives the eye for a few ticks
  local cur = loom * gain
  if cur > CUR_MAX then cur = CUR_MAX end

  local now = badge.sys.ms()
  local flirt = now < flirt_until
  local gf_tick = 0
  for _ = 1, STEPS_PER_TICK do
    gf_tick = gf_tick + step(cur, flirt)
  end
  gf_count = gf_count + gf_tick
  if gf_tick > 0 then                      -- this tick's spikes, not the whole reporting window
    flash_until, jump_until = now + FLASH_MS, now + JUMP_MS
  end

  -- LEDs: white on an escape spike, pink while another badge is nearby, one red dot when lesioned.
  -- Written only when the state changes, so an idle tick costs no native calls.
  local want = (now < flash_until) and "f" or (flirt and "p" or (lesion and "l" or "o"))
  if want ~= led_state then
    led_state = want
    if want == "f" then set_leds(255, 255, 255)
    elseif want == "p" then set_leds(120, 20, 60)
    elseif want == "l" then badge.led.clear(); badge.led.set(1, 60, 0, 0); badge.led.show()
    else badge.led.clear(); badge.led.show() end
  end
  ui.drive:set_value(cur)

  ui.fly:set_pos(140, now < jump_until and 70 or 110)

  if radio_ok and now - last_beacon > 1000 then
    last_beacon = now
    badge.radio.send("FLY1")
  end

  tick_ms = badge.sys.ms() - t0
  if now - win_start >= 1000 then
    gf_rate = floor(gf_count * 1000 / (now - win_start) / 2)
    gf_count, win_start = 0, now
    ui.gf:set_text("giant fibre  " .. gf_rate .. " Hz")
    local st = badge.sys.stats()
    ui.perf:set_text(tick_ms .. " ms/tick   " .. floor(tick_ms / STEPS_PER_TICK) .. " ms/step   heap "
      .. floor(st.lua_used / 1024) .. "k   gain " .. gain)
  end
end

function on_button(button, kind)
  if kind ~= badge.input.KIND.PRESSED then return end
  local B = badge.input.BUTTON
  if button == B.B then
    lesion = true
    ui.state:set_text("LC4 + LPLC2 LESIONED")
    ui.state:style({ text_color = 0xFF4444 })
  elseif button == B.A then
    lesion = false
    ui.state:set_text("INTACT")
    ui.state:style({ text_color = 0x44DD88 })
  elseif button == B.UP then
    gain = math.min(40, gain + 1)
  elseif button == B.DOWN then
    gain = math.max(1, gain - 1)
  elseif button == B.START then
    reset_brain()
    loom = 0
  end
end

function on_exit()
  badge.store.set_int("gain", gain)
  badge.led.clear()
  badge.led.show()
  if radio_ok then badge.radio.on_recv(nil); badge.radio.disable() end
end

-- Test hook. badge/test_lua.lua runs this exact step() against badge/sim_fixed.py to confirm the
-- integer arithmetic agrees spike for spike. Nothing on the badge ever reads it.
_G.__flybadge_test = {
  C = C,
  step = function(cur, flirt) return step(cur, flirt) end,
  reset = reset_brain,
  spikes = function() return spk, spk_n end,
  set_lesion = function(v) lesion = v end,
}

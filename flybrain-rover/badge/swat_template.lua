--[==[badge-app
slug=swat
name=SWAT
icon=><
api=2
heap_kb=96
wake_lock=1
version=1.0
author=FlyBrain Rover
]==]

-- SWAT -- race a real fruit fly's escape reflex, in milliseconds.
--
-- Press A to arm. After a random wait a dark disc looms out of the centre of the screen. Its size is the
-- current injected into LC4 and LPLC2, the 109 looming detectors of this subsample of the male Drosophila
-- connectome (neuPrint male-cns:v1.0), wired to the giant fibre with the real synapse counts and signs.
-- Press A again before the giant fibre spikes and you win the round. On NORMAL it fires at 250 ms, about
-- a human's visual reaction time, so you lose roughly half the rounds. Press B to zero LC4 and LPLC2's
-- outgoing synapses: the fly never escapes again, and the score does the explaining.
--
-- The circuit, the fixed-point constants, reset_brain() and step() are copied byte for byte from
-- badge/app_template.lua, which tests/test_badge_app.py checks against badge/sim_fixed.py spike for
-- spike. Nothing below decides whether the fly escapes: the only way into ESCAPED is step() returning a
-- giant fibre spike, so the spec's 3 s safety timeout ends in its own TIMEOUT state ("fly did not react").
-- Every timestamp is badge.sys.ms(); the fly's latency is counted in 10 ms brain steps, because the
-- brain's clock is exact and the 20 ms tick is only nominal. The brain takes its first looming step on
-- the tick after LOOMING is entered, so brain time never runs ahead of the clock your press is stamped with.

--@CIRCUIT@

local function b64(s)
  local A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  local map = {}
  for i = 1, 64 do map[A:sub(i, i)] = i - 1 end
  local out, buf, nb, n, bits, no = {}, {}, 0, 0, 0, 0
  for i = 1, #s do
    local v = map[s:sub(i, i)]
    if v then
      bits = bits * 64 + v; n = n + 6
      if n >= 8 then
        n = n - 8
        local byte = bits // (2 ^ n); bits = bits % (2 ^ n)
        nb = nb + 1; buf[nb] = string.char(byte)
        if nb == 64 then no = no + 1; out[no] = table.concat(buf); nb = 0 end
      end
    end
  end
  if nb > 0 then no = no + 1; out[no] = table.concat(buf, "", 1, nb) end
  return table.concat(out)
end
C.post, C.w = b64(C.post_b64), b64(C.w_b64)
C.post_b64, C.w_b64 = nil, nil

-- fixed point: 1 unit = 1/256 mV, one step = 10 ms. Must match badge/sim_fixed.py exactly.
local V_REST, V_TH = -13312, -11520
local SYN_DECAY = 35            -- exp(-dt/tau_syn) = exp(-10/5) as a 256ths fraction
local GAIN_Q = 75               -- one int8 weight unit in membrane units
local REFRAC = 1                -- one 10 ms step
local STEPS_PER_TICK = 2        -- ticks are nominally 20 ms, the brain runs at 100 Hz
local CUR_MAX = 4000            -- membrane units; threshold is about 1750

local byte, floor = string.byte, math.floor
local V, G, REF, SYN = {}, {}, {}, {}
local spk, spk_n = {}, 0
local lesion = false

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

-- Everything above this line is the fly. Everything below is the referee.

local min, format = math.min, string.format
local EYE_LAST = C.eye_last

local function now() return badge.sys.ms() end  -- the one clock; the test hook rebinds it

local IDLE, ARMED, LOOMING = "IDLE", "ARMED", "LOOMING"
local HIT, ESCAPED, FALSE_START, TIMEOUT = "HIT", "ESCAPED", "FALSE_START", "TIMEOUT"

local DIFFS = { "SLOW", "NORMAL", "FAST", "INSANE" }
local RAMP_OF = { SLOW = 45, NORMAL = 80, FAST = 120, INSANE = 200 }   -- membrane units per 10 ms step
local ARM_MIN_MS, ARM_SPAN_MS = 800, 1400
local LOOM_MAX_MS, RESULT_MS = 3000, 1400
local FLASH_MS, JUMP_MS = 180, 280
local PERF_MS = 1000                            -- the temporary ms/tick label refreshes this often
local CORAL, DIM, WHITE, RED = 0xFF715B, 0x8A8580, 0xFFFFFF, 0xFF4444

-- LED patterns, as small integers so a tick compares numbers and allocates nothing
local LED_OFF, LED_LESION, LED_WHITE, LED_GREEN, LED_CORAL = 0, 1, 2, 3, 10   -- LED_CORAL + n lit

local state = IDLE
local diff = 2                                  -- index into DIFFS; NORMAL
local hits, rounds = 0, 0
local loom, steps_since_loom = 0, 0             -- injected drive, and brain steps since looming began
local loom_t0, arm_until, result_until = 0, 0, 0
local gf_ms, press_ms = nil, nil                -- the fly's latency (brain steps x 10) and yours
local flash_until, jump_until, flash_key = 0, 0, LED_OFF
local eye_tick = 0                              -- eye spikes this tick, drives the LEDs while looming
local led_key, fly_y, disc_d = -1, 0, 0
local tick_ms, tick_max, perf_until = 0, 0, 0   -- spec criterion 5: ms/tick must stay under 10
local ui = {}

local function show_leds(key)
  if key == LED_WHITE then badge.led.set_all(255, 255, 255)
  elseif key == LED_GREEN then badge.led.set_all(40, 200, 90)
  elseif key >= LED_CORAL then
    badge.led.clear()
    for i = 1, key - LED_CORAL do badge.led.set(i, 255, 113, 91) end   -- clockwise from upper left
  else
    badge.led.clear()
    if key == LED_LESION then badge.led.set(1, 60, 0, 0) end
  end
  badge.led.show()
end

-- The disc is the stimulus: its size is cur / CUR_MAX, the same number the eye neurons receive.
local function draw_disc(progress)
  local d = 10 + floor(200 * progress)
  if d == disc_d then return end
  disc_d = d
  ui.disc:set_size(d, d)
  ui.disc:set_pos(160 - d // 2, 118 - d // 2)
  ui.disc:style({ radius = d // 2 })
end

local function set_times(text, color)
  ui.times:set_text(text)
  ui.times:style({ text_color = color })
end

local function set_score()
  ui.score:set_text(format("%02d / %02d", hits, rounds))
end

local function set_sub()
  if lesion then
    ui.sub:set_text("LC4 + LPLC2 CUT · it cannot see you")
    ui.sub:style({ text_color = RED })
  else
    ui.sub:set_text(C.n .. " neurons · " .. DIFFS[diff])
    ui.sub:style({ text_color = DIM })
  end
end

local function end_round(t, result, text, color)
  state = result
  result_until = t + RESULT_MS
  rounds = rounds + 1
  set_score()
  set_times(text, color)
end

local function arm(t)
  state = ARMED
  arm_until = t + ARM_MIN_MS + badge.sys.random(ARM_SPAN_MS)
  gf_ms, press_ms = nil, nil                    -- a new round; nothing from the last one carries over
  set_times("ready", DIM)
end

local function start_looming(t)
  state = LOOMING
  loom_t0 = t
  loom, steps_since_loom = 0, 0
  set_times("", DIM)
end

-- The fly wins. This is the only way into ESCAPED and its one caller runs it only when step() returned a
-- giant fibre spike this tick. A press in the same tick loses: the fly is faster than the button scan.
local function fly_escapes(t)
  flash_until, jump_until, flash_key = t + FLASH_MS, t + JUMP_MS, LED_WHITE
  local text = format("you --   fly %d ms", gf_ms)
  if press_ms then text = format("fly %d ms · same tick", gf_ms) end
  end_round(t, ESCAPED, text, CORAL)
end

local function you_hit(t)
  hits = hits + 1
  flash_until, flash_key = t + FLASH_MS, LED_GREEN
  end_round(t, HIT, format("you %d ms   fly >%d ms", press_ms, steps_since_loom * 10), CORAL)
end

local function false_start(t)
  end_round(t, FALSE_START, "false start   fly --", WHITE)
end

local function timed_out(t)
  end_round(t, TIMEOUT, "you --   fly did not react", DIM)
end

local function go_idle()
  state = IDLE
  reset_brain()
  loom = 0
  draw_disc(0)                                  -- back to the 10x10 dot before the next round can arm
end

local function tick_looming(t)
  local ramp, gf_tick, cur = RAMP_OF[DIFFS[diff]], 0, 0
  eye_tick = 0
  for _ = 1, STEPS_PER_TICK do
    loom = loom + ramp
    cur = min(CUR_MAX, loom)
    gf_tick = gf_tick + step(cur, false)
    steps_since_loom = steps_since_loom + 1
    if gf_tick > 0 and not gf_ms then gf_ms = steps_since_loom * 10 end
    for k = 1, spk_n do if spk[k] <= EYE_LAST then eye_tick = eye_tick + 1 end end
  end
  if gf_tick > 0 then fly_escapes(t) end
  if state == LOOMING then
    if press_ms then you_hit(t)
    elseif t - loom_t0 >= LOOM_MAX_MS then timed_out(t) end
  end
  draw_disc(cur / CUR_MAX)
end

-- LEDs and the sprite, written only when they change, so an idle tick costs no native calls.
local function paint(t)
  local key
  if t < flash_until then key = flash_key
  elseif state == LOOMING then
    local n = min(6, eye_tick // 3)
    if n > 0 then key = LED_CORAL + n elseif lesion then key = LED_LESION else key = LED_OFF end
  elseif lesion then key = LED_LESION
  else key = LED_OFF end
  if key ~= led_key then led_key = key; show_leds(key) end

  local y = t < jump_until and 74 or 110
  if y ~= fly_y then fly_y = y; ui.fly:set_pos(150, y) end
end

function on_enter(root)
  diff = badge.store.get_int("diff", 2)
  if diff < 1 or diff > #DIFFS then diff = 2 end
  lesion = badge.store.get_int("lesion", 0) == 1
  reset_brain()

  -- Creation order is draw order: LVGL paints later children on top, so the disc covers the background
  -- and the fly sits on the disc without any widget method the SDK does not document.
  ui.bg = badge.ui.box(root, 320, 240)
  ui.bg:style({ bg_color = 0x000000, border_width = 0 })
  ui.bg:set_pos(0, 0)

  ui.disc = badge.ui.box(root, 10, 10)
  ui.disc:style({ bg_color = 0x2A0E0A, border_width = 0, radius = 5 })
  ui.disc:set_pos(155, 113)
  disc_d = 10

  ui.fly = badge.ui.label(root, "**")
  ui.fly:style({ text_color = WHITE, text_font = 24 })
  ui.fly:set_pos(150, 110)
  fly_y = 110

  ui.title = badge.ui.label(root, "SWAT")
  ui.title:style({ text_color = CORAL, text_font = 20 })
  ui.title:align("top_left", 10, 6)

  ui.sub = badge.ui.label(root, "")
  ui.sub:style({ text_color = DIM, text_font = 14 })
  ui.sub:align("top_left", 10, 30)
  set_sub()

  ui.score = badge.ui.label(root, "")
  ui.score:style({ text_color = WHITE, text_font = 18 })
  ui.score:align("top_right", -10, 6)
  set_score()

  ui.times = badge.ui.label(root, "press A")
  ui.times:style({ text_color = DIM, text_font = 20 })
  ui.times:align("top_left", 10, 186)

  ui.hint = badge.ui.label(root, "A swat   B lesion   UP/DOWN speed   START reset")
  ui.hint:style({ text_color = DIM, text_font = 14 })
  ui.hint:align("top_left", 10, 216)

  -- Temporary, spec criterion 5: read ms/tick on the badge itself (the laptop harness cannot), then
  -- delete this label and the block at the end of on_tick that feeds it.
  ui.perf = badge.ui.label(root, "")
  ui.perf:style({ text_color = DIM, text_font = 14 })
  ui.perf:align("top_right", -10, 30)

  state = IDLE
  led_key = lesion and LED_LESION or LED_OFF
  show_leds(led_key)
end

function on_tick()
  local t = now()
  if state == ARMED and t >= arm_until then
    start_looming(t)                            -- first brain step is next tick: brain time never leads t
  elseif state == LOOMING then
    tick_looming(t)
  elseif state == IDLE or state == ARMED then
    for _ = 1, STEPS_PER_TICK do step(0, false) end
  elseif t >= result_until then
    go_idle()
  end
  paint(t)

  -- Temporary, spec criterion 5. The same measurement app_template.lua makes: this tick's cost on the
  -- badge's own clock, refreshed once a second with the worst tick since the last refresh.
  tick_ms = now() - t
  if tick_ms > tick_max then tick_max = tick_ms end
  if t >= perf_until then
    perf_until = t + PERF_MS
    ui.perf:set_text(format("%d ms/tick  max %d  heap %dk", tick_ms, tick_max,
      floor(badge.sys.stats().lua_used / 1024)))
    tick_max = 0
  end
end

function on_button(button, kind)
  if kind ~= badge.input.KIND.PRESSED then return end
  local B, t = badge.input.BUTTON, now()
  if button == B.A then
    if state == IDLE then arm(t)
    elseif state == ARMED then false_start(t)
    elseif state == LOOMING and not press_ms then press_ms = t - loom_t0 end   -- resolved on the tick
  elseif button == B.B then
    lesion = not lesion
    set_sub()
  elseif button == B.UP and state ~= LOOMING then
    diff = diff % #DIFFS + 1                    -- cycles: INSANE wraps to SLOW
    set_sub()
  elseif button == B.DOWN and state ~= LOOMING then
    diff = (diff - 2) % #DIFFS + 1              -- and SLOW wraps to INSANE
    set_sub()
  elseif button == B.START and state ~= LOOMING then
    hits, rounds = 0, 0
    set_score()
  end
end

function on_exit()
  badge.store.set_int("diff", diff)
  badge.store.set_int("lesion", lesion and 1 or 0)
  badge.led.clear()
  badge.led.show()
end

-- Test hook. badge/test_swat.lua drives the whole game on a laptop through this: tick() runs exactly what
-- on_tick runs, with now() returning now_ms instead of badge.sys.ms(). accel_xyz is accepted for parity
-- with FlyBadge and ignored, because SWAT's stimulus is the disc, not the accelerometer. Nothing on the
-- badge ever reads this table.
local test_ms = 0
local function test_now() return test_ms end
_G.__swat_test = {
  step = function(cur, flirt) return step(cur, flirt) end,
  reset = reset_brain,
  set_lesion = function(v) lesion = v end,
  state = function() return state, loom, steps_since_loom, hits, rounds, gf_ms, press_ms end,
  tick = function(now_ms, accel_xyz)
    test_ms = now_ms
    now = test_now
    on_tick()
  end,
  button = function(name)
    on_button(badge.input.BUTTON[name], badge.input.KIND.PRESSED)
  end,
}

--[==[badge-app
slug=swatgame
name=Swat
icon=SWT
api=2
heap_kb=96
wake_lock=1
version=1.0
author=FlyBrain Rover
]==]

-- Swat -- mosquitoes and chickens fly through a swatter bolted to the middle of the screen.
-- SHAKE THE BADGE when it is a mosquito. Hold still for the chickens. Three lives.
--
-- The mosquito is hard to hit because a real one is. Each mosquito carries its own copy of 150 neurons
-- and 900 synapses of the male Drosophila CNS connectome (neuPrint male-cns:v1.0, Berg et al. 2026),
-- the LC4 and LPLC2 looming detectors and the giant fibre they drive. Shaking the badge makes the
-- swatter lunge, the mosquito sees it expand, and if its giant fibre fires it is gone before your hand
-- lands. Nothing else in this file can make a mosquito escape: fly_escapes() is called from exactly one
-- place, under `if gf > 0`, and the harness tests that.
--
-- Measured, not tuned until it felt right (badge/sim_fixed.py, and devpost/swat_game.html in a browser):
--   below about 1800 units of current the giant fibre never fires at all;
--   above it, it fires in 10 to 40 ms, which is faster than the 60 ms the swatter takes to land.
-- So you cannot out-race a mosquito that has seen you. You beat it by shaking when it is already under
-- the swatter and has nowhere to go: inside about 28 px its 400 px/s dodge cannot clear the head.
--
-- A still swatter is not a looming object, so an untouched crossing never trips anything. The shake is
-- the stimulus. That is why shaking early is punished, and it is the circuit doing the punishing.
--
-- No Bluetooth, it panics the badge (badge/SDK_NOTES.md). No FPU, so fixed point.

--@CIRCUIT@

-- base64 -> bytes, once, in 64-byte chunks: a 900-entry table costs 24.6 KB of peak heap to hold 900.
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

local byte, floor, abs = string.byte, math.floor, math.abs
local V, G, REF, SYN = {}, {}, {}, {}
local spk, spk_n = {}, 0
local lesion = false

local function reset_brain()
  for i = 1, C.n do V[i] = V_REST; G[i] = 0; REF[i] = 0; SYN[i] = 0 end
  spk, spk_n = {}, 0
end

-- One 10 ms step of leaky integrate-and-fire over the real connectome. Event driven: only the neurons
-- that fired last step push current, so the cost is spikes times out-degree, not the whole matrix.
-- Byte-identical to badge/app_template.lua, which is checked against the Python reference.
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

-- ---------------------------------------------------------------- the game
-- The swatter never moves. You only choose when. SX/SY is its centre, SR its reach.
local SX, SY, SR = 160, 112, 26
local THR = 1793                        -- measured: 1792 fires nothing, 1793 fires the whole column
local STRIKE_MS, LUNGE_MS = 60, 110     -- your hand takes 60 ms to land; the head is big for 110
local DART = 340                        -- px/s, the escape once the giant fibre has fired
local LANE = 5                          -- how far off centre a bug may be born
local WOBBLE = 4                        -- and how far it drifts while flying
local CARD_MS, LIVES = 700, 3
local WORST = 26                        -- native-call ceiling for one on_tick, enforced by the harness

-- Sprite parts, all axis-aligned boxes, offsets from the bug's centre, drawn facing right and mirrored
-- by negating ox. {ox, oy, w, h, radius, colour}
local MOS = {
  {-2, -9, 16,  7, 3, 0x5A6470},   -- wing
  {-6,  0, 17,  6, 3, 0x3C362E},   -- abdomen
  { 4,  0,  9,  8, 4, 0x2E2A25},   -- thorax
  {11,  0,  8,  8, 4, 0x221F1C},   -- head
  {18,  0,  8,  2, 1, 0x8A4A3A},   -- the needle: this is a mosquito, not a housefly
  {11, -1,  5,  5, 2, 0x5A1610},   -- eye, recoloured when its alarm is high
}
local HEN = {
  {-16, -2, 14, 15, 6, 0xE6D8C6},  -- tail
  {  0,  0, 26, 20, 9, 0xF4E9DA},  -- body
  { -4, 14,  3,  8, 1, 0xE8A33C},  -- legs
  {  4, 14,  3,  8, 1, 0xE8A33C},
  { 13,-13, 14, 14, 6, 0xF4E9DA},  -- head
  { 13,-22,  6,  6, 2, 0xD94F3A},  -- comb
  { 22,-12,  7,  4, 1, 0xE8A33C},  -- beak
  { 16,-15,  3,  3, 1, 0x1F1B18},  -- eye
}
local NPART = 8

-- style() is only ever handed one of these. Per-tick table churn is the failure mode this badge has
-- already shown once, at 17.7 KB of peak heap for 900 bytes.
local SHIDE, SSHOW = {bg_opa = 0}, {bg_opa = 255}
local SMESH, SFLASH = {bg_color = 0xD6D2CB}, {bg_color = 0xFFFFFF}
local SRING = {border_color = 0x1B2E24, border_width = 2}
local SRING_ON = {border_color = 0xFFC24A, border_width = 4}
local SEYE = {[0] = {bg_color = 0x5A1610}, {bg_color = 0xC03420}, {bg_color = 0xFF7A5C}}
local SLIFE = {[0] = {bg_color = 0x2A2420}, {bg_color = 0xFF715B}}
local SSAY = {[0] = {text_color = 0xECE7E1}, {text_color = 0x44DD88},
              {text_color = 0xFF715B}, {text_color = 0xFF3B22}}
local SHINT, SHINT_CUT = {text_color = 0x6E6660}, {text_color = 0xFF3B22}

local ui, PW = {}, {}                   -- PW: the pool of 8 part boxes, reused by both animals
local state, tprev, now_ms = "idle", 0, 0
local score, best, lives, spawned, since_hen = 0, 0, LIVES, 0, 0
local bug_kind, bx, by0, bdir, bvx, bph, bdodge, bbolt, balarm = nil, 0, SY, 1, 0, 0, 0, 0, 0
local bprev_d, card_until, lunge_until, strike_at, pending = 999, 0, 0, 0, false
local led_n, led_c, say_i = -1, -1, -1
local ring_on, robot_stopped = false, false

local function say(txt, colour)
  if ui.say then ui.say:set_text(txt) end
  if colour ~= say_i then ui.say:style(SSAY[colour]); say_i = colour end
end

-- The badge's two robot controls. Each one prints a line on the USB serial console; a laptop running
-- scripts/badge_link.py reads it and presses the matching hotkey on scripts/demo.py over UDP. Serial,
-- not BLE: badge.radio.enable() panics this device with the circuit loaded (badge/SDK_NOTES.md), and
-- badge.radio filters receive to its own LUA1 frames anyway, so a laptop could not join.
-- The badge plays perfectly well with nothing listening. These are announcements, not a dependency.
local function set_lesion(on)
  lesion = on
  if ui.hint then
    ui.hint:set_text(on and "LOBOTOMIZED - it cannot see you" or "")
    ui.hint:style(on and SHINT_CUT or SHINT)
  end
  badge.sys.log(on and "@FLY LOBO 1" or "@FLY LOBO 0")
end

local function set_robot_stop(on)
  robot_stopped = on
  badge.sys.log(on and "@FLY STOP 1" or "@FLY STOP 0")
end

-- The one place a mosquito is allowed to leave. Called under `if gf > 0` and nowhere else.
local function fly_escapes()
  bbolt = (by0 < SY) and -1 or 1
  bvx = bvx + (bvx > 0 and 40 or -40)
end

local function spawn()
  -- at least one chicken in every four, so mashing the badge cannot pay
  local hen = (since_hen >= 3) or (spawned > 0 and badge.sys.random(10) < 3)
  since_hen = hen and 0 or since_hen + 1
  spawned = spawned + 1
  bug_kind = hen and "hen" or "mos"
  bdir = (badge.sys.random(2) == 1) and 1 or -1
  bx = (bdir > 0) and -34 or 354
  by0 = SY - LANE + badge.sys.random(2 * LANE + 1) - 1
  -- starts gentle and creeps up: 84 px/s at nothing, +3 per point, capped so it stays playable
  -- standing at a table. It takes 15 points to get half again as fast.
  local pace = 84 + (score < 18 and score * 3 or 54)
  bvx = bdir * (hen and 62 or pace)
  bph, bdodge, bbolt, balarm, bprev_d = badge.sys.random(600) / 100, 0, 0, 0, 999
  if not hen then reset_brain() end
end

-- ---------------------------------------------------------------- drawing
local function place(t, n)
  -- one set_pos per visible part; the parts that this animal does not use were hidden on the swap
  local cy = by0 + floor(bdodge)
  for k = 1, n do
    local p = t[k]
    PW[k]:set_pos(floor(bx + bdir * p[1] - p[3] / 2), cy + p[2] - floor(p[4] / 2))
  end
end

local function dress(t, n)
  -- called once when an animal appears, never per tick
  for k = 1, NPART do
    if k <= n then
      local p = t[k]
      PW[k]:set_size(p[3], p[4])
      PW[k]:style({bg_color = p[6], radius = p[5], bg_opa = 255, border_width = 0})
    else
      PW[k]:style(SHIDE)
    end
  end
end

function on_enter(root)
  ui.bg = badge.ui.box(root, 320, 240)
  ui.bg:set_pos(0, 0); ui.bg:style({bg_color = 0x07110D, border_width = 0})

  ui.ring = badge.ui.box(root, 2 * (SR + 12), 2 * (SR + 12))
  ui.ring:set_pos(SX - SR - 12, SY - SR - 12)
  ui.ring:style({bg_opa = 0, radius = SR + 12, border_width = 2, border_color = 0x1B2E24})

  for k = 1, NPART do
    PW[k] = badge.ui.box(root, 4, 4)
    PW[k]:style(SHIDE)
  end

  -- the swatter, bolted to the middle
  ui.rim = badge.ui.box(root, 56, 56)
  ui.rim:set_pos(SX - 28, SY - 28); ui.rim:style({bg_color = 0x2A2724, radius = 12, border_width = 0})
  ui.mesh = badge.ui.box(root, 46, 46)
  ui.mesh:set_pos(SX - 23, SY - 23); ui.mesh:style({bg_color = 0xD6D2CB, radius = 10, border_width = 0})
  ui.mh = badge.ui.box(root, 46, 2)
  ui.mh:set_pos(SX - 23, SY - 1); ui.mh:style({bg_color = 0x8E8A84, border_width = 0})
  ui.mv = badge.ui.box(root, 2, 46)
  ui.mv:set_pos(SX - 1, SY - 23); ui.mv:style({bg_color = 0x8E8A84, border_width = 0})
  ui.grip = badge.ui.box(root, 9, 32)
  ui.grip:set_pos(SX + 4, SY + 26); ui.grip:style({bg_color = 0x43372C, radius = 4, border_width = 0})

  ui.score = badge.ui.label(root, "0")
  ui.score:style({text_color = 0xECE7E1, text_font = 24}); ui.score:align("top_left", 10, 6)
  ui.best = badge.ui.label(root, "best 0")
  ui.best:style({text_color = 0x6E6660, text_font = 14}); ui.best:align("top_left", 10, 34)
  for k = 1, LIVES do
    ui["l" .. k] = badge.ui.box(root, 12, 12)
    ui["l" .. k]:set_pos(300 - (k - 1) * 18, 10)
    ui["l" .. k]:style({bg_color = 0xFF715B, radius = 6, border_width = 0})
  end

  ui.alarm = badge.ui.bar(root, 0, 100, 0)
  ui.alarm:set_size(300, 8); ui.alarm:set_pos(10, 224)
  ui.alarm:style({bg_color = 0x0F1A15, radius = 4})

  ui.say = badge.ui.label(root, "SHAKE TO SWAT")
  ui.say:style({text_color = 0xECE7E1, text_font = 22}); ui.say:align("top_left", 68, 150)
  ui.hint = badge.ui.label(root, "mosquitoes yes, chickens no")
  ui.hint:style({text_color = 0x6E6660, text_font = 14}); ui.hint:align("top_left", 66, 180)

  best = badge.store.get_int("swat_best", 0)
  ui.best:set_text("best " .. best)
  tprev = badge.sys.ms()
  -- a fresh entry owns the clock: a lunge_until left in the future from a previous run would eat every
  -- shake until the badge caught up with it
  lunge_until, strike_at, card_until, pending = 0, 0, 0, false
  robot_stopped = false
  bug_kind, bbolt, bdodge, balarm = nil, 0, 0, 0
  led_n, led_c, say_i = -1, -1, -1
  ring_on = false
  state = "idle"
end

local function set_leds(n, r, g, b)
  if n == led_n and r == led_c then return end
  led_n, led_c = n, r
  for k = 1, 6 do
    if k <= n then badge.led.set(k, r, g, b) else badge.led.set(k, 0, 0, 0) end
  end
  badge.led.show()
end

local function go_idle()
  if score > best then best = score; badge.store.set_int("swat_best", best); ui.best:set_text("best " .. best) end
  state = "idle"; bug_kind = nil
  pending, strike_at = false, 0
  ui.mesh:style(SMESH)          -- a run that ended on a strike left the head flashed white
  dress(MOS, 0)
  say("SHAKE TO SWAT", 0)
  ui.hint:set_text("mosquitoes yes, chickens no")
  ui.alarm:set_value(0)
  set_leds(0, 0, 0, 0)
end

local function begin()
  score, lives, spawned, since_hen = 0, LIVES, 0, 0
  -- a fresh run owns the clock, the same reason on_enter does: a lunge_until left in the future eats
  -- the first shakes and reads as a dead button
  lunge_until, strike_at, pending = 0, 0, false
  ui.score:set_text("0")
  for k = 1, LIVES do ui["l" .. k]:style(SLIFE[1]) end
  ui.hint:set_text("")
  say("", 0)
  state = "play"
  spawn()
  dress(bug_kind == "hen" and HEN or MOS, bug_kind == "hen" and 8 or 6)
end

local function card(txt, colour)
  say(txt, colour)
  card_until = now_ms + CARD_MS
  state = "card"
end

-- the strike: resolved STRIKE_MS after you shook, which is the window the mosquito is racing
local function resolve()
  local cy = by0 + floor(bdodge)
  local dx, dy = bx - SX, cy - SY
  local reach = SR + (bug_kind == "hen" and 16 or 18)
  ui.mesh:style(SFLASH)
  if dx * dx + dy * dy > reach * reach then
    -- out of reach. Whether that is your timing or its reflex is the whole lesson, so say which.
    card(bbolt ~= 0 and "IT GOT OUT" or "MISSED", bbolt ~= 0 and 2 or 0)
  elseif bug_kind == "hen" then
    lives = lives - 1
    ui["l" .. (lives + 1)]:style(SLIFE[0])
    dress(HEN, 0); bug_kind = nil
    if lives <= 0 then card("GAME OVER  " .. score, 3); state = "over"; card_until = now_ms + 2000
    else card("THAT WAS A CHICKEN", 3) end
  else
    score = score + 1
    ui.score:set_text(tostring(score))
    dress(MOS, 0); bug_kind = nil
    card("GOT IT", 1)
  end
end

function on_tick()
  now_ms = badge.sys.ms()
  local dt = now_ms - tprev
  if dt < 1 then dt = 1 elseif dt > 100 then dt = 100 end
  tprev = now_ms

  local shook = badge.sensor.shake()

  if state == "idle" then
    if shook then begin() end
    return
  end
  if state == "over" then
    if now_ms >= card_until then go_idle() end
    return
  end
  if state == "card" then
    if now_ms >= card_until then
      ui.mesh:style(SMESH)
      -- a shake in the dying tick of a round must not resolve against the next animal
      pending, strike_at = false, 0
      state = "play"; spawn()
      dress(bug_kind == "hen" and HEN or MOS, bug_kind == "hen" and 8 or 6)
    end
    return
  end

  -- ---- play
  if shook and now_ms >= lunge_until then
    lunge_until = now_ms + LUNGE_MS
    strike_at = now_ms + STRIKE_MS
    pending = true
  end
  local lunging = now_ms < lunge_until

  bx = bx + bvx * dt / 1000
  bph = bph + dt * 36 / 10000
  if bbolt ~= 0 then bdodge = bdodge + bbolt * DART * dt / 1000 end

  if bug_kind == "mos" then
    local cy = by0 + floor(bdodge)
    local dx, dy = bx - SX, cy - SY
    local d2 = dx * dx + dy * dy
    if d2 < 1 then d2 = 1 end
    local d = math.sqrt(d2)
    local closing = (bprev_d - d) * 1000 / dt
    if closing < 0 then closing = 0 end
    bprev_d = d
    -- A still swatter is not a looming object, so the crossing alone stays under threshold. The shake
    -- is the stimulus, and it is louder the nearer the mosquito already is.
    local den = d2 < 500 and 500 or d2
    local cur = floor(380 * SR * SR / den + 160 * closing * SR / den)
    if cur > 1500 then cur = 1500 end
    if lunging then
      local dl = d2 < 300 and 300 or d2
      cur = cur + floor(5200 * SR * SR / dl)
    end
    if cur > CUR_MAX then cur = CUR_MAX end
    local gf = 0
    for _ = 1, STEPS_PER_TICK do gf = gf + step(cur, false) end
    if gf > 0 and bbolt == 0 then fly_escapes() end
    -- THR is the measured firing threshold: 1792 fires nothing anywhere, 1793 fires the column. So the
    -- spike count is all-or-nothing and makes a useless bar. The drive is what is worth watching.
    balarm = floor(balarm * 5 / 10 + cur * 5 / 10)
    local pct = floor(balarm * 100 / THR)
    if pct > 100 then pct = 100 end
    ui.alarm:set_value(pct)
    local lv = pct > 70 and 2 or (pct > 30 and 1 or 0)
    PW[6]:style(SEYE[lv])
    set_leds(floor(pct * 6 / 100), 255, 113, 91)
  else
    ui.alarm:set_value(0)
    set_leds(0, 0, 0, 0)
  end

  if bug_kind == "hen" then place(HEN, 8) else place(MOS, 6) end

  -- The ring is the cue, not scenery: lit exactly when a strike would reach. Written only on change,
  -- so a quiet tick still costs nothing.
  local cy2 = by0 + floor(bdodge)
  local ddx, ddy = bx - SX, cy2 - SY
  local rr = SR + (bug_kind == "hen" and 14 or 12)
  local near = (ddx * ddx + ddy * ddy) <= rr * rr
  if near ~= ring_on then
    ring_on = near
    ui.ring:style(near and SRING_ON or SRING)
  end

  if pending and now_ms >= strike_at then
    pending = false
    resolve()
    return
  end

  -- off the edge
  if bx < -50 or bx > 370 or abs(bdodge) > 140 then
    local was = bug_kind
    dress(bug_kind == "hen" and HEN or MOS, 0)
    bug_kind = nil
    if was == "mos" and bbolt ~= 0 then card("IT GOT AWAY", 2)
    elseif was == "mos" then card("IT FLEW PAST", 0)
    else card("SPARED IT", 1) end
  end
end

function on_button(button, kind)
  if kind ~= badge.input.KIND.PRESSED then return end
  local B = badge.input.BUTTON
  if button == B.A then
    -- the same verb as a shake, for anyone who does not want to wave the badge about
    if state == "idle" then begin()
    elseif state == "play" and now_ms >= lunge_until then
      lunge_until = now_ms + LUNGE_MS; strike_at = now_ms + STRIKE_MS; pending = true
    end
  elseif button == B.B or button == B.LEFT then
    set_lesion(not lesion)
  elseif button == B.RIGHT then
    -- stop and resume the robot's wheels. Its brain keeps running, which is the point of stop rather
    -- than silence, and the badge game carries on regardless.
    set_robot_stop(not robot_stopped)
    say(robot_stopped and "ROBOT STOPPED" or "ROBOT RUNNING", robot_stopped and 3 or 1)
  elseif button == B.START then
    go_idle()
  end
end

function on_exit()
  -- never walk away leaving the robot lesioned or halted because the badge was put down
  if lesion then set_lesion(false) end
  if robot_stopped then set_robot_stop(false) end
  if score > best then badge.store.set_int("swat_best", score) end
  badge.led.clear(); badge.led.show()
end

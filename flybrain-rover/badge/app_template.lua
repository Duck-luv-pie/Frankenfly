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

-- FlyBadge / STARTLE -- a real fruit fly's escape circuit on your badge. Measured anatomy, not a model
-- and not a trained net: a subsample of the male Drosophila CNS connectome (neuPrint male-cns:v1.0,
-- Berg et al. 2026), 54 LC4 and 55 LPLC2 looming detectors of which 106 synapse straight onto the
-- giant fibre (in-degrees 73 and 76), whose spike launches the escape jump.
-- The screen is the fly. A ring whose diameter IS the current going into LC4/LPLC2 closes on it, and
-- contact is the measured threshold: 1792 membrane units fires nothing anywhere, 1793 fires the whole
-- column. Then a bolt runs down the giant fibre, the screen flinches, a shockwave goes out, it leaps.
-- B cuts the eye's output: the ring still closes and the eyes still blaze, and the bolt stops dead on
-- the red slash. UP/DOWN gain (saved), LEFT slow motion, START the SWAT screen.
-- Nothing here can make the fly escape: the renderer only reads spk[] and the count step() returns,
-- and the phase-2 abort tests that count, not the lesion flag. DISPLAY, not biology: the replay is
-- ~12x slow motion of one real volley (really two stages 10 ms apart, both inside one tick) and the
-- bars are leaky rate estimates; LEFT steps the real brain slowly instead of replaying anything.
-- No Bluetooth, it panics the badge. No FPU, so fixed point, checked against badge/sim_fixed.py.

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
local FLASH_MS, JUMP_MS, FLIRT_MS = 160, 260, 3000

local byte, floor, abs = string.byte, math.floor, math.abs
local V, G, REF, SYN = {}, {}, {}, {}
local spk, spk_n = {}, 0
local lesion, gain = false, 3
local loom, last = 0, nil

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

-- STARTLE, the renderer: strictly downstream of step(), never writes V/G/REF/SYN/spk/lesion. WORST is
-- the counted native-call ceiling for one on_tick, enforced by badge/test_visual.lua. It is the phase
-- 6 / phase 11 tick: 3 sensor and clock + 3 ring + 2 eyes + 4 bars + 7 LED + 25 for the leap (19
-- set_pos, 2 wings, 2 shockwave, 2 bolt). Measured across every scenario in the harness: 41.
local THR, CX, CY, JDY, WORST = 1793, 160, 116, 22, 44

-- style() is only ever handed one of these, never a fresh literal: per-tick table churn is the
-- failure mode this badge has already shown once, at 17.7 KB of peak heap for 900 bytes.
local SRB = {[0]={border_color=0x3A1410},{border_color=0x8A2418},{border_color=0xD03320},{border_color=0xFF3B20}}
local SEY = {[0]={bg_color=0x5A1610},{bg_color=0x8A2016},{bg_color=0xC03420},{bg_color=0xFF5A3C},{bg_color=0xFFD2C6}}
local STX = {[0]={bg_color=0x3E3226},{bg_color=0x6B4F28},{bg_color=0xB07A2C},{bg_color=0xE8A33C}}
local STR = {[0]={bg_color=0x2A2320},{bg_color=0xFFC24A},{bg_color=0x3A1512}}
local SOP = {[0]={bg_opa=0},[60]={bg_opa=60},[140]={bg_opa=140},[255]={bg_opa=255}}
local SWO = {[0]={opa=0},[255]={opa=255}}
local SWG = {[0]={bg_color=0x28303A},{bg_color=0x93AEC8}}
local SBG = {[0]={bg_color=0x0A0A0A,border_width=0},{bg_color=0x39332E,border_width=0}}  -- the flinch
-- Bolt height by phase: it starts at y=70 (the eye's outputs) so its tip is 70+h, and the red slash
-- is 86..92, so phase 2 (tip 92) is the frame where the wave crosses the cut or dies on it.
local BH = {[0]=6,12,22,34,50,70}
local LB = {[0]=0,40,96,170,255}                                  -- LED brightness by stage level
local LP = {{1,2,255,113,91},{6,3,232,163,60},{5,4,255,255,255}}  -- pair, pair, colour at full
local LL = {-1,-1,-1}

local ui, JW, JX, JY, JN, TR = {}, {}, {}, {}, 0, {}
local full, mode, slow, slow_c, wd = true, 0, false, 0, 0
local r_eye, r_int, gf_all, flash_until = 0, 0, 0, 0
local P, wn_gf, TOPT = 90, 0, ""                   -- replay phase; 90 = idle and re-armable
local HINT = "B cuts its eye"                      -- one control, not four
local SAY_IDLE, SAY_GO, SAY_BLIND = "SWAT ME", "IT GOT AWAY", "IT CANNOT SEE YOU"
local say_until = 0                                 -- how long the answer stays up
local d_s, dc_s, ev_s, dv_s, be_s, bi_s, bg_s = -1, -1, -1, -1, -1, -1, -1

local function lv(v) return v>=85 and 4 or (v>=55 and 3 or (v>=28 and 2 or (v>=8 and 1 or 0))) end
local function tract(s) local t = STR[s]; for k = 1, #TR do TR[k]:style(t) end end
local function bodyy(dy) for k = 1, JN do JW[k]:set_pos(JX[k], JY[k] + dy) end end

local function wave()
  local d = wd
  if d > 0 then
    d = d + 26
    if d > 260 then wd = 0; ui.wav:style(SWO[0])
    else wd = d; ui.wav:set_size(d, d); ui.wav:set_pos(CX - d // 2, CY - d // 2) end
  end
end

-- One frame of the replay. 0..12 are the escape; 30..31 retract the bolt after an abort.
local function render(p)
  if p <= 5 then
    ui.spike:set_size(10, BH[p])
    if p == 0 then ui.spike:style(SOP[255])
    elseif p == 3 then tract(1)
    elseif p == 4 then ui.thor:style(STX[1])
    elseif p == 5 then ui.thor:style(STX[2]) end
  elseif p == 6 then
    ui.bg:style(SBG[1])
    wd = 104; ui.wav:style(SWO[255]); wave()
    bodyy(-JDY); ui.wingL:style(SWG[1]); ui.wingR:style(SWG[1])
  elseif p == 7 then ui.bg:style(SBG[0]); wave()
  elseif p == 8 then wave(); ui.thor:style(STX[3])
  elseif p == 9 then wave(); ui.spike:style(SOP[140])
  elseif p == 10 then wave(); ui.thor:style(STX[1]); tract(lesion and 2 or 0)
  elseif p == 11 then
    wave(); bodyy(0); ui.wingL:style(SWG[0]); ui.wingR:style(SWG[0])
    ui.spike:set_size(10, 6); ui.spike:style(SOP[0])
  elseif p == 12 then
    if wd > 0 then wd = 0; ui.wav:style(SWO[0]) end
    ui.thor:style(STX[0])
  elseif p == 30 then ui.spike:set_size(10, 12); ui.spike:style(SOP[140])
  elseif p == 31 then ui.spike:set_size(10, 6); ui.spike:style(SOP[0]) end
end

function on_enter(root)
  gain = badge.store.get_int("gain", 3); reset_brain()
  TOPT = C.n .. " real neurons   " .. C.n_edges .. " real synapses"
  -- Heap read BEFORE any widget: ~32 KB is all that is free, so the optional bars go if it is tight.
  local st = badge.sys.stats()
  full = (not st) or (not st.free_heap) or st.free_heap >= 24000

  local o = badge.ui.box(root, 320, 240); o:set_pos(0, 0)
  o:style(SBG[0]); ui.bg = o
  -- radius >= half the size makes a box a circle, so a ring is set_size + set_pos and no style().
  ui.threat = badge.ui.box(root, 300, 300); ui.threat:set_pos(CX - 150, CY - 150)
  ui.threat:style({bg_opa=0, radius=160, border_width=3, border_color=0x3A1410})
  ui.wav = badge.ui.box(root, 2, 2); ui.wav:set_pos(CX, CY)
  ui.wav:style({bg_opa=0, radius=160, border_width=5, border_color=0xFF715B, opa=0})

  -- The fly, back to front. Colour is the legend: dark red eyes are LC4 + LPLC2, the vertical channel
  -- is the giant fibre, the tapering bars above it are 109 eye cells converging on one axon.
  -- name, w, h, x, y, colour, radius, 0 leaps / 1 leaps and is part of the tract
  local PA = {
    {"wingL",76,24,60,92,0x28303A,12,0},{"wingR",76,24,184,92,0x28303A,12,0},
    {"abdo",46,58,137,136,0x3A2E22,22,0},{"band1",46,6,137,152,0x1E1812,3,0},
    {"band2",46,6,137,170,0x1E1812,3,0},{"thor",56,48,132,94,0x3E3226,20,0},
    {"head",30,30,145,42,0x2A2622,14,0},{"eyeL",30,34,120,40,0x5A1610,15,0},
    {"eyeR",30,34,170,40,0x5A1610,15,0},{"trbL",44,4,116,72,0x2A2320,2,1},
    {"trbR",44,4,160,72,0x2A2320,2,1},{"trbl",22,4,138,78,0x2A2320,2,1},
    {"trbr",22,4,160,78,0x2A2320,2,1},{"axon",10,56,155,84,0x241E1A,5,0},
    {"nrv1",22,5,149,98,0x2A2320,2,1},{"nrv2",22,5,149,112,0x2A2320,2,1},
    {"nrv3",22,5,149,126,0x2A2320,2,1},{"spike",10,6,155,70,0xFFFFFF,5,0},
    {"cut",92,6,114,86,0xFF2A18,3,0},
  }
  for k = 1, #PA do
    local p = PA[k]
    o = badge.ui.box(root, p[2], p[3]); o:set_pos(p[4], p[5])
    o:style({bg_color=p[6], radius=p[7], border_width=0})
    ui[p[1]] = o
    JN = JN + 1; JW[JN] = o; JX[JN] = p[4]; JY[JN] = p[5]
    if p[8] == 1 then TR[#TR + 1] = o end
  end
  ui.spike:style(SOP[0]); ui.cut:style(SOP[0])

  -- The drive signal with the measured threshold on it: a redundant marker, no per-tick calls.
  ui.drive = badge.ui.bar(root, 0, CUR_MAX, 0)
  ui.drive:set_pos(12, 200); ui.drive:set_size(296, 6)
  ui.drive:style({bg_color=0x14181A, radius=3})
  o = badge.ui.box(root, 2, 12); o:set_pos(12 + 296 * THR // CUR_MAX, 197)
  o:style({bg_color=0xF2EFEA, border_width=0})

  -- Stage bars, small on purpose, and leaky rates not raw counts: the eye fires all 54 LC4 cells on
  -- alternating 10 ms steps, so a raw bar would strobe at 25 Hz and read as a fault.
  local BA = {{"beye",12,0xFF715B,1},{"bint",114,0xFFC24A,2},{"bgf",216,0xFFFFFF,2}}
  for k = 1, 3 do
    local q = BA[k]
    if q[4] == 1 or full then
      o = badge.ui.bar(root, 0, 100, 0); o:set_pos(q[2], 214); o:set_size(92, 5)
      o:style({bg_color=0x161616, color=q[3]}); ui[q[1]] = o
    end
  end

  ui.top = badge.ui.label(root, TOPT)
  ui.top:style({text_color=0x6E6660, text_font=14}); ui.top:align("top_left", 10, 4)
  ui.hint = badge.ui.label(root, HINT)
  ui.hint:style({text_color=0x4E4844, text_font=14}); ui.hint:align("top_left", 10, 222)
  -- The one thing the first version got wrong: nothing on screen said what to do. This is bigger than
  -- everything else and it changes to answer whatever just happened.
  ui.say = badge.ui.label(root, "SWAT ME")
  ui.say:style({text_color=0xFFFFFF, text_font=24}); ui.say:align("top_left", 10, 26)

  -- SWAT placeholder, three widgets over the top: merging badge/GAME_SPEC.md is deleting this block.
  ui.sw1 = badge.ui.box(root, 320, 240); ui.sw1:set_pos(0, 0)
  ui.sw1:style({bg_color=0x0A0A0A, border_width=0, opa=0})
  ui.sw2 = badge.ui.label(root, "SWAT")
  ui.sw2:style({text_color=0xFF715B, text_font=24, opa=0}); ui.sw2:align("top_left", 20, 92)
  ui.sw3 = badge.ui.label(root, "race the fly's reflex. coming soon.")
  ui.sw3:style({text_color=0x6E6660, text_font=16, opa=0}); ui.sw3:align("top_left", 20, 128)

  badge.led.clear(); badge.led.set(1, 0, 8, 16); badge.led.show()
end

-- One LED pair, written only when its level changed. A lit LED always means neurons fired, so the
-- idle light is teal: never a spike colour, so coral there means the eye really volleyed.
local function ledpair(k, l)
  if l == LL[k] then return false end
  LL[k] = l
  local p = LP[k]
  if k == 1 and l == 0 then
    if lesion then badge.led.set(1, 60, 0, 0) else badge.led.set(1, 0, 8, 16) end
    badge.led.set(2, 0, 0, 0)
  else
    local b = LB[l]
    local r, g, bl = b * p[3] // 255, b * p[4] // 255, b * p[5] // 255
    badge.led.set(p[1], r, g, bl); badge.led.set(p[2], r, g, bl)
  end
  return true
end

function on_tick()
  if mode ~= 0 then return end                 -- SWAT placeholder: zero native calls, brain frozen
  local now = badge.sys.ms()

  -- Looming, from the only sensor this badge has: the swat itself, there is no light sensor.
  local x, y, z = badge.sensor.accel()
  local jerk = 0
  if x then
    if last then jerk = abs(x - last[1]) + abs(y - last[2]) + abs(z - last[3])
    else last = {} end
    last[1], last[2], last[3] = x, y, z
  end
  if badge.sensor.shake() then jerk = jerk + 400 end
  loom = floor(loom * 3 / 4) + jerk             -- leaky, so one swat drives the eye for a few ticks
  local cur = loom * gain
  if cur > CUR_MAX then cur = CUR_MAX end

  -- LEFT steps the brain once per 5 ticks, not twice per tick. The circuit is untouched, just
  -- stepped less often, which makes the real 10 ms ordering inspectable.
  local ns = STEPS_PER_TICK
  if slow then
    slow_c = slow_c + 1
    if slow_c >= 5 then slow_c = 0; ns = 1 else ns = 0 end
  end
  local s_eye, s_int, gf_tick = 0, 0, 0
  for _ = 1, ns do
    gf_tick = gf_tick + step(cur, false)
    for k = 1, spk_n do
      local i = spk[k]
      if i <= C.eye_last then s_eye = s_eye + 1
      elseif i > C.lc10a_last and i < C.gf_first then s_int = s_int + 1 end
    end
  end
  gf_all = gf_all + gf_tick
  if gf_tick > 0 then flash_until = now + FLASH_MS end
  if gf_tick > 0 then ui.say:set_text(SAY_GO); ui.say:style({text_color=0xFF715B}); say_until = now + 1400 end
  if say_until > 0 and now > say_until and not lesion then
    ui.say:set_text(SAY_IDLE); ui.say:style({text_color=0xFFFFFF}); say_until = 0
  end

  -- Only 8 of the 27 intermediates ever fire, at most 7 in one volley, so this divides by 7. All 12
  -- LC10a cells are silent here and drawn nowhere: never draw a neuron that cannot light.
  r_eye = r_eye * (slow and 246 or 192) // 256 + s_eye * 2
  if r_eye > 100 then r_eye = 100 end
  r_int = r_int * (slow and 250 or 214) // 256 + s_int * 14
  if r_int > 100 then r_int = 100 end

  -- The ring reaches the fly's silhouette (d = 170) at exactly cur = THR. Contact is the threshold.
  local d
  if cur <= THR then d = 300 - (cur * 130) // THR
  else d = 170 - ((cur - THR) * 50) // (CUR_MAX - THR) end
  d = d // 4 * 4
  if d ~= d_s then
    d_s = d; ui.threat:set_size(d, d); ui.threat:set_pos(CX - d // 2, CY - d // 2)
  end
  local dc = cur >= THR and 3 or (cur >= 1200 and 2 or (cur >= 600 and 1 or 0))
  if dc ~= dc_s then dc_s = dc; ui.threat:style(SRB[dc]) end

  -- The eyes: a real LC4 cell's sub-threshold membrane potential until it spikes, then the smoothed
  -- rate. Both are plain Lua table reads, neither is a native call.
  local mv = (V[1] - V_REST) * 100 // 1792
  if mv < 0 then mv = 0 end
  if r_eye > mv then mv = r_eye end
  local ev = lv(mv)
  if ev ~= ev_s then ev_s = ev; ui.eyeL:style(SEY[ev]); ui.eyeR:style(SEY[ev]) end

  -- Arms on a real eye volley. The phase-2 abort tests the giant-fibre count step() returned, summed
  -- over phases 0..2 because the giant fibre fires on the step AFTER the eye, maybe in the next tick.
  local adv = ns > 0
  if P >= 90 and s_eye > 0 and adv then P = -1; wn_gf = gf_tick end
  if P < 90 and adv then
    P = P + 1
    if P <= 2 then wn_gf = wn_gf + gf_tick end
    render(P)
    if P == 2 and wn_gf == 0 then P = 29               -- the wave dies on the cut
    elseif P == 12 or P == 31 then P = 90 end
  end

  -- Bars. Two neurons with an all-or-none output get a latch, never a smooth meter.
  local dv = cur // 20
  if dv ~= dv_s then dv_s = dv; ui.drive:set_value(cur) end
  local be = r_eye // 4
  if be ~= be_s then be_s = be; ui.beye:set_value(r_eye) end
  local gf_on = now < flash_until
  if ui.bint then
    local bi = r_int // 4
    if bi ~= bi_s then bi_s = bi; ui.bint:set_value(r_int) end
    local bv = gf_on and 100 or 0
    if bv ~= bg_s then bg_s = bv; ui.bgf:set_value(bv) end
  end

  -- Three pairs down the badge: {1,2} eye, {6,3} intermediates, {5,4} giant fibre, so the light runs
  -- down the badge as the wave runs down the fly. One show(), only if a pair changed.
  local w1 = ledpair(1, lv(r_eye))
  local w2 = ledpair(2, lv(r_int))
  local w3 = ledpair(3, gf_on and 4 or 0)
  if w1 or w2 or w3 then badge.led.show() end
end

function on_button(button, kind)
  if kind ~= badge.input.KIND.PRESSED then return end
  local B = badge.input.BUTTON
  if button == B.B then
    lesion = true
    ui.cut:style(SOP[255]); tract(2)
    ui.top:set_text("LC4 + LPLC2 CUT"); ui.top:style({text_color=0xFF2A18})
    ui.say:set_text(SAY_BLIND); ui.say:style({text_color=0xFF2A18}); say_until = 0
    LL[1] = -1
  elseif button == B.A then
    lesion = false
    ui.cut:style(SOP[0]); tract(0)
    ui.top:set_text(TOPT); ui.top:style({text_color=0x6E6660})
    ui.say:set_text(SAY_IDLE); ui.say:style({text_color=0xFFFFFF}); say_until = 0
    LL[1] = -1
  elseif button == B.LEFT then
    slow = not slow; slow_c = 0
    ui.hint:set_text(slow and "SLOW MOTION   one 10 ms brain step per 100 ms   LEFT back" or HINT)
  elseif button == B.UP then gain = math.min(40, gain + 1)
  elseif button == B.DOWN then gain = math.max(1, gain - 1)
  elseif button == B.START then
    mode = 1 - mode
    local o = mode == 1 and SWO[255] or SWO[0]
    ui.sw1:style(o); ui.sw2:style(o); ui.sw3:style(o)
    badge.led.clear()
    if mode == 1 then badge.led.set(1, 0, 8, 16) end
    badge.led.show()
    LL[1], LL[2], LL[3] = -1, -1, -1
  end
end

function on_exit()
  badge.store.set_int("gain", gain)
  badge.led.clear()
  badge.led.show()
end

-- Test hooks: badge/test_lua.lua checks step() against badge/sim_fixed.py, badge/test_visual.lua
-- drives the whole app and counts native calls. Nothing on the badge reads either.
_G.__flybadge_test = {
  C = C,
  step = function(cur, flirt) return step(cur, flirt) end,
  reset = reset_brain,
  spikes = function() return spk, spk_n end,
  set_lesion = function(v) lesion = v end,
  ui = ui, worst = WORST, bolt_full = BH[5], bolt_cut = BH[2],
  state = function() return P, r_eye, r_int, gf_all, LL[1], LL[2], LL[3], mode end,
}

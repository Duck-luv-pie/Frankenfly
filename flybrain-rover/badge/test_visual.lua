-- test_visual.lua -- drive the whole badge app on a laptop and check what the screen and LEDs did.
--
--   lua badge/test_visual.lua            # run every scenario, print a CSV summary, exit 0 or 1
--
-- badge/test_lua.lua proves the arithmetic. This proves the exhibit: that a swat really does light the
-- giant fibre's widgets, that cutting the eye leaves the eye lit and everything downstream dark, and
-- that the renderer stays inside its native-call budget. The `badge` table here is a stub that records
-- every widget write and counts every native call, because the firmware is not available off the
-- device but the call pattern is, and the call pattern is where a 20 ms tick goes wrong.
--
-- tests/test_visual_app.py runs this and asserts on the same numbers.

local NCALL, NSHOW = 0, 0
local LED = {}
for i = 1, 6 do LED[i] = {0, 0, 0} end

local function widget(kind, w, h, value, text)
  local o = {kind = kind, w = w or 0, h = h or 0, x = 0, y = 0,
             bg_color = -1, bg_opa = 255, opa = 255, value = value or 0, text = text or ""}
  function o:style(t)
    NCALL = NCALL + 1
    if type(t) ~= "table" then error("style() needs a table") end
    if t.bg_color then self.bg_color = t.bg_color end
    if t.bg_opa then self.bg_opa = t.bg_opa end
    if t.opa then self.opa = t.opa end
    if t.color then self.color = t.color end
    self.styles = (self.styles or 0) + 1
    -- record which preallocated tables this widget has ever been handed, so the test can prove the
    -- app never builds a fresh literal per tick: the same table identity must come back every time.
    self.seen = self.seen or {}
    self.seen[t] = (self.seen[t] or 0) + 1
    return self
  end
  function o:set_pos(x, y) NCALL = NCALL + 1; self.x, self.y = x, y; return self end
  function o:set_size(w, h) NCALL = NCALL + 1; self.w, self.h = w, h; return self end
  function o:set_value(v) NCALL = NCALL + 1; self.value = v; return self end
  function o:set_text(t) NCALL = NCALL + 1; self.text = t; return self end
  function o:align(_, x, y) NCALL = NCALL + 1; self.x, self.y = x, y; return self end
  function o:bring_to_front() NCALL = NCALL + 1; return self end
  return o
end

local MS, ACC, SHAKE, HEAP = 0, {0, 0, 1000}, false, 60000
-- `lua badge/test_visual.lua lowheap` reports a badge that is nearly out of heap, which is what
-- on_enter's degrade ladder is for: the optional bars are never created and no code path may assume
-- they exist.
local LOW = (arg and arg[1] == "lowheap")
if LOW then HEAP = 10000 end

badge = {
  ui = {
    box = function(_, w, h) NCALL = NCALL + 1; return widget("box", w, h) end,
    bar = function(_, _, _, v) NCALL = NCALL + 1; return widget("bar", 0, 0, v) end,
    label = function(_, t) NCALL = NCALL + 1; return widget("label", 0, 0, 0, t) end,
  },
  led = {
    set = function(i, r, g, b) NCALL = NCALL + 1; LED[i] = {r, g, b} end,
    set_all = function(r, g, b) NCALL = NCALL + 1; for i = 1, 6 do LED[i] = {r, g, b} end end,
    clear = function() NCALL = NCALL + 1; for i = 1, 6 do LED[i] = {0, 0, 0} end end,
    show = function() NCALL = NCALL + 1; NSHOW = NSHOW + 1 end,
    count = function() return 6 end,
  },
  input = {
    BUTTON = {A = 1, B = 2, HOME = 3, DOWN = 4, LEFT = 5, RIGHT = 6, UP = 7, START = 8},
    KIND = {PRESSED = 1, RELEASED = 0},
    is_down = function() return false end, held = function() return 0 end,
  },
  sensor = {
    accel = function() NCALL = NCALL + 1; return ACC[1], ACC[2], ACC[3] end,
    shake = function() NCALL = NCALL + 1; return SHAKE end,
    tap = function() return false end,
  },
  sys = {
    ms = function() NCALL = NCALL + 1; return MS end,
    uptime = function() return MS end,
    log = function() end,
    random = function(n) return n // 2 end,
    stats = function() NCALL = NCALL + 1; return {lua_used = 0, free_heap = HEAP, widgets = 0} end,
  },
  store = {get_int = function(_, d) return d end, set_int = function() end,
           get = function(_, d) return d end, set = function() end},
}

local f = assert(loadfile("badge/flybadge_app.lua"))
f()
local T = assert(_G.__flybadge_test, "app did not expose its test hook")
local ui = assert(T.ui, "app did not expose its widgets")

-- ---------------------------------------------------------------------------------- the driver ----
local function led_is(i, r, g, b) return LED[i][1] == r and LED[i][2] == g and LED[i][3] == b end

-- One tick. `swat` injects a jerk by moving the accelerometer; the app's own leaky accumulator does
-- the rest, exactly as a real hand would.
local function tick(swat)
  if swat then ACC = {ACC[1] + 1500, ACC[2], ACC[3]} end
  NCALL, NSHOW = 0, 0
  on_tick()
  MS = MS + 20
  return NCALL, NSHOW
end

local function run(n, every, acc)
  for i = 1, n do
    local calls, shows = tick(every and i % every == 1)
    acc.ticks = acc.ticks + 1
    if calls > acc.max_calls then acc.max_calls = calls end
    if shows > acc.max_show then acc.max_show = shows end
    if ui.spike.bg_opa > 0 and ui.spike.h > acc.bolt then acc.bolt = ui.spike.h end
    if ui.bgf and ui.bgf.value > acc.gfbar then acc.gfbar = ui.bgf.value end
    if ui.eyeL.bg_color == 0xFFD2C6 then acc.eye_hot = acc.eye_hot + 1 end
    if ui.nrv1.bg_color == 0xFFC24A then acc.tract_lit = acc.tract_lit + 1 end
    if ui.thor.bg_color == 0xE8A33C then acc.thorax_hot = acc.thorax_hot + 1 end
    if led_is(5, 255, 255, 255) and led_is(4, 255, 255, 255) then acc.led_gf = acc.led_gf + 1 end
    if LED[1][1] > 60 then acc.led_eye = acc.led_eye + 1 end
    if LED[6][2] > 0 then acc.led_int = acc.led_int + 1 end
    if ui.threat.w ~= acc.ring_last then acc.ring_moves = acc.ring_moves + 1; acc.ring_last = ui.threat.w end
    if ui.threat.w < acc.ring_min then acc.ring_min = ui.threat.w end
  end
end

local function fresh()
  return {ticks = 0, max_calls = 0, max_show = 0, bolt = 0, gfbar = 0, eye_hot = 0, tract_lit = 0,
          thorax_hot = 0, led_gf = 0, led_eye = 0, led_int = 0, ring_moves = 0, ring_min = 9999,
          ring_last = -1, gf0 = 0}
end

local function gf_total() local _, _, _, g = T.state(); return g end

-- Quiet ticks with a throwaway accumulator. A replay armed by the previous scenario is still in
-- flight when a button is pressed, and its last frames would otherwise be scored against the next
-- scenario -- which is how a lesion run ends up "seeing" the escape it was supposed to abolish.
local function settle(n)
  local junk = fresh()
  run(n or 30, nil, junk)
end

local FAIL = 0
local function check(name, ok, detail)
  if not ok then FAIL = FAIL + 1; io.write("FAIL  ", name, "  ", detail or "", "\n") end
  return ok
end

-- ------------------------------------------------------------------------------- the scenarios ----
on_enter({})
local enter_calls = NCALL

-- 1. INTACT. Quiet, then a swat, then let the replay run out.
local idle = fresh()
run(20, nil, idle)
local gf_before = gf_total()

local hit = fresh()
run(40, 12, hit)
local gf_intact = gf_total() - gf_before

check("intact: the giant fibre fired", gf_intact > 0, "gf spikes " .. gf_intact)
check("intact: the bolt reached the thorax", hit.bolt >= T.bolt_full, "bolt h " .. hit.bolt)
if LOW then
  check("lowheap: the optional bars were not built", ui.bgf == nil and ui.bint == nil, "bars exist")
else
  check("intact: the giant-fibre bar latched", hit.gfbar == 100, "bar " .. hit.gfbar)
end
check("intact: the bottom LED pair went white", hit.led_gf > 0, "ticks " .. hit.led_gf)
check("intact: the eye lit", hit.eye_hot > 0, "ticks " .. hit.eye_hot)
check("intact: the tract carried spikes", hit.tract_lit > 0, "ticks " .. hit.tract_lit)
check("intact: the thorax flared", hit.thorax_hot > 0, "ticks " .. hit.thorax_hot)
check("idle: nothing fired with no stimulus", idle.bolt == 0 and idle.led_gf == 0, "bolt " .. idle.bolt)
check("idle: the ring still tracks the hand", idle.ring_moves >= 1, "moves " .. idle.ring_moves)

-- 2. LESIONED. Same stimulus for 200 ticks. The eye must still light and nothing downstream may.
on_button(badge.input.BUTTON.B, badge.input.KIND.PRESSED)
settle()
local gf_mark = gf_total()
local cut = fresh()
run(200, 10, cut)
local gf_cut = gf_total() - gf_mark

check("cut: zero giant-fibre spikes over 200 ticks", gf_cut == 0, "gf spikes " .. gf_cut)
check("cut: the eye still lights", cut.eye_hot > 0, "ticks " .. cut.eye_hot)
check("cut: the top LED pair still lights", cut.led_eye > 0, "ticks " .. cut.led_eye)
check("cut: the ring still closes", cut.ring_min <= 170, "min d " .. cut.ring_min)
check("cut: the bolt died on the slash", cut.bolt <= T.bolt_cut, "bolt h " .. cut.bolt)
check("cut: the tract never carried spikes", cut.tract_lit == 0, "ticks " .. cut.tract_lit)
check("cut: the thorax never flared", cut.thorax_hot == 0, "ticks " .. cut.thorax_hot)
check("cut: the giant-fibre bar stayed at zero", cut.gfbar == 0, "bar " .. cut.gfbar)
check("cut: the eye bar still moves", ui.beye.value > 0, "bar " .. ui.beye.value)
check("cut: the middle LED pair stayed dark", cut.led_int == 0, "ticks " .. cut.led_int)
check("cut: the bottom LED pair stayed dark", cut.led_gf == 0, "ticks " .. cut.led_gf)
check("cut: the red slash is drawn", ui.cut.bg_opa == 255, "opa " .. ui.cut.bg_opa)

-- 3. RESTORED. A restores the escape at the same latency, because nothing was destroyed.
on_button(badge.input.BUTTON.A, badge.input.KIND.PRESSED)
settle()
local gf_mark2 = gf_total()
local back = fresh()
run(40, 12, back)
local gf_back = gf_total() - gf_mark2
check("restore: the escape came back", gf_back > 0, "gf spikes " .. gf_back)
check("restore: the slash is gone", ui.cut.bg_opa == 0, "opa " .. ui.cut.bg_opa)

-- 4. SLOW MOTION. LEFT steps the real brain once per 5 ticks, so the same window has ~1/10 the spikes.
on_button(badge.input.BUTTON.LEFT, badge.input.KIND.PRESSED)
settle()
local gf_mark3 = gf_total()
local slow = fresh()
run(40, 12, slow)
local gf_slow = gf_total() - gf_mark3
check("slow: still escapes, more slowly", gf_slow > 0 and gf_slow < gf_back, "slow " .. gf_slow ..
      " vs normal " .. gf_back)
on_button(badge.input.BUTTON.LEFT, badge.input.KIND.PRESSED)
settle()

-- 5. SWAT PLACEHOLDER. START costs nothing per tick and freezes the brain.
on_button(badge.input.BUTTON.START, badge.input.KIND.PRESSED)
local ph = fresh()
run(20, 5, ph)
check("swat screen: zero native calls per tick", ph.max_calls == 0, "calls " .. ph.max_calls)
check("swat screen: it is visible", ui.sw1.opa == 255 and ui.sw2.opa == 255, "opa " .. ui.sw1.opa)
on_button(badge.input.BUTTON.START, badge.input.KIND.PRESSED)
check("swat screen: START goes back", ui.sw1.opa == 0, "opa " .. ui.sw1.opa)

-- 6. THE BUDGET, over every tick of every scenario above.
local worst = math.max(idle.max_calls, hit.max_calls, cut.max_calls, back.max_calls, slow.max_calls)
local shows = math.max(idle.max_show, hit.max_show, cut.max_show, back.max_show, slow.max_show)
check("budget: under the stated worst case", worst <= T.worst, worst .. " > " .. T.worst)
check("budget: at most one led.show() per tick", shows <= 1, "shows " .. shows)

-- 7. NO FRESH STYLE TABLES. Every widget that is restyled must be handed the same few table
--    identities over and over; a fresh {bg_color=...} literal per tick would show up as one
--    distinct table per call.
local function distinct(w)
  local n = 0
  for _ in pairs(w.seen or {}) do n = n + 1 end
  return n
end
check("no per-tick table churn: the eye", distinct(ui.eyeL) <= 6, distinct(ui.eyeL) .. " tables")
check("no per-tick table churn: the bolt", distinct(ui.spike) <= 6, distinct(ui.spike) .. " tables")
check("no per-tick table churn: the ring", distinct(ui.threat) <= 6, distinct(ui.threat) .. " tables")
check("no per-tick table churn: the thorax", distinct(ui.thor) <= 6, distinct(ui.thor) .. " tables")

print("scenario,ticks,gf_spikes,max_calls,max_show,bolt_h,gf_bar,eye_ticks,tract_ticks,led_gf_ticks")
local function row(n, a, g)
  print(string.format("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d", n, a.ticks, g, a.max_calls, a.max_show,
    a.bolt, a.gfbar, a.eye_hot, a.tract_lit, a.led_gf))
end
row("idle", idle, 0)
row("intact_swat", hit, gf_intact)
row("lesioned", cut, gf_cut)
row("restored", back, gf_back)
row("slow_motion", slow, gf_slow)
row("swat_screen", ph, 0)
print(string.format("budget,%d,%d,%d,%d,%d,0,0,0,0", 0, 0, worst, shows, T.worst))
print(string.format("on_enter,1,0,%d,0,0,0,0,0,0", enter_calls))
print((FAIL == 0) and "OK" or ("FAILURES " .. FAIL))
os.exit(FAIL == 0 and 0 or 1)

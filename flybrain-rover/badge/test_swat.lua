-- test_swat.lua -- drive the SWAT badge game on a laptop and print one CSV line per round.
--
--   lua badge/test_swat.lua
--
-- tests/test_swat_app.py runs this and checks the numbers against badge/GAME_SPEC.md section 3. The
-- firmware is not available off the badge, but the game is pure Lua over a stubbed `badge` table, so the
-- whole state machine can run here: the clock is whatever the harness passes to __swat_test.tick(),
-- badge.sys.random() returns a fixed value, the store is a table, widgets remember their text, size and
-- position, and the LED stub counts writes so "one show() per tick" is checked.
--
-- The widget stub defines only the six methods badge/SDK_NOTES.md documents (align, set_pos, set_size,
-- set_text, set_value, style). Anything else the app calls is a nil method here and a crash on the badge,
-- so the harness fails on it instead of masking it.
--
-- Scenarios, one CSV line each: scenario,difficulty,result,press_ms,gf_ms
--   a  every difficulty, never press: the fly escapes at the measured latency
--   b  NORMAL, press A 100 ms after looming starts: HIT with press_ms 100
--   c  NORMAL, press A while ARMED: FALSE_START
--   d  lesion on (B), INSANE, never press, 20 rounds: the 3000 ms timeout ends each round, gf_ms nil
--   e  lesion off (B), INSANE: the escape is back at the same latency as in (a)
-- Between rounds the disc must be back to the spec's 10x10 dot at (155,113): asserted on every IDLE and
-- ARMED tick, so a disc that kept the previous round's size fails here.

local widgets = {}
local function widget(text, w, h)
  local x = { text = text, w = w, h = h, st = {} }
  x.set_text = function(self, s) self.text = s end
  x.set_size = function(self, sw, sh) self.w, self.h = sw, sh end
  x.set_pos = function(self, px, py) self.x, self.y = px, py end
  x.align = function(self, anchor, ax, ay) self.anchor, self.ax, self.ay = anchor, ax, ay end
  x.style = function(self, s) for k, v in pairs(s) do self.st[k] = v end end
  x.set_value = function(self, v) self.value = v end
  widgets[#widgets + 1] = x
  return x
end

local store = {}
local led = { writes = 0, shows = 0 }

badge = {
  ui = {
    screen_width = 320, screen_height = 240,
    label = function(_, text) return widget(text) end,
    box = function(_, w, h) return widget("", w, h) end,
    bar = function() return widget("") end,
  },
  led = {
    set = function() led.writes = led.writes + 1 end,
    set_all = function() led.writes = led.writes + 1 end,
    clear = function() led.writes = led.writes + 1 end,
    show = function() led.shows = led.shows + 1 end,
    count = function() return 6 end,
  },
  input = {
    BUTTON = { A = 1, B = 2, HOME = 3, DOWN = 4, LEFT = 5, RIGHT = 6, UP = 7, AUX1 = 8, START = 9 },
    KIND = { PRESSED = 1, RELEASED = 0 },
  },
  sensor = { accel = function() return 0, 0, 1000 end, shake = function() return false end },
  sys = {
    ms = function() return 0 end,
    random = function() return 200 end,           -- arm delay is then exactly 800 + 200 ms
    stats = function() return { lua_used = 0 } end,
    log = function() end,
  },
  store = {
    get_int = function(k, d) return store[k] or d end,
    set_int = function(k, v) store[k] = v end,
    get = function(k, d) return store[k] or d end,
    set = function(k, v) store[k] = v end,
  },
  radio = { enable = function() return false end, disable = function() end,
            on_recv = function() end, send = function() end },
}

local f = assert(loadfile("badge/swat_app.lua"))
f()
local T = assert(_G.__swat_test, "app did not expose its test hook")

local TICK = 20
local t = 0

local function state() return T.state() end

-- the looming disc is the one box painted 0x2A0E0A; its size and position are what the spec fixes
local function disc()
  for _, w in ipairs(widgets) do
    if w.st.bg_color == 0x2A0E0A then return w end
  end
  error("no disc widget")
end

local function tick()
  local shows = led.shows
  t = t + TICK
  T.tick(t, { 0, 0, 1000 })
  assert(led.shows - shows <= 1, "more than one badge.led.show() in a tick")
  local s = state()
  if s == "IDLE" or s == "ARMED" then
    local d = disc()
    assert(d.w == 10 and d.h == 10 and d.x == 155 and d.y == 113,
      string.format("%s at %d ms: disc is %sx%s at (%s,%s), spec says 10x10 at (155,113)",
        s, t, tostring(d.w), tostring(d.h), tostring(d.x), tostring(d.y)))
  end
end

local function sub_text()
  for _, w in ipairs(widgets) do
    if w.text:find("neurons", 1, true) or w.text:find("CUT", 1, true) then return w.text end
  end
  error("no subtitle widget")
end

local function assert_sub(name)
  assert(sub_text():find(name, 1, true), "subtitle does not show " .. name .. ": " .. sub_text())
end

-- UP cycles SLOW -> NORMAL -> FAST -> INSANE -> SLOW (spec section 7), so any level is at most three
-- presses away. The harness tracks the index itself because the subtitle shows the lesion, not the
-- difficulty, while LC4 and LPLC2 are cut.
local DIFF_INDEX = { SLOW = 1, NORMAL = 2, FAST = 3, INSANE = 4 }
local diff_idx = 2                               -- the app boots on NORMAL; asserted below
local function set_difficulty(name)
  for _ = 1, (DIFF_INDEX[name] - diff_idx) % 4 do T.button("UP") end
  diff_idx = DIFF_INDEX[name]
  if sub_text():find("neurons", 1, true) then assert_sub(name) end
end

local function is_result(s)
  return s == "HIT" or s == "ESCAPED" or s == "FALSE_START" or s == "TIMEOUT"
end

-- One round from IDLE back to IDLE. press_at: ms after looming starts to press A (nil = never).
-- false_start: press A while ARMED instead. Returns result, press_ms, gf_ms.
local function round(press_at, false_start)
  assert(state() == "IDLE", "round must start from IDLE, got " .. state())
  T.button("A")
  assert(state() == "ARMED", "A in IDLE must arm")
  local t_end                                    -- clock when the round ended
  if false_start then T.button("A"); t_end = t end
  local loom_t0
  for _ = 1, 400 do                              -- 8 s guard
    if is_result(state()) then break end
    tick()
    local s = state()
    if s == "LOOMING" and not loom_t0 then loom_t0 = t end
    if is_result(s) then t_end = t; break end
    if press_at and loom_t0 and t - loom_t0 == press_at then T.button("A") end
  end
  local s, _, _, _, _, gf_ms, press_ms = state()
  assert(is_result(s), "round never ended, state " .. s)
  if s == "TIMEOUT" then
    assert(t - loom_t0 == 3000, "timeout fired at " .. (t - loom_t0) .. " ms, not 3000")
  end
  for _ = 1, 100 do
    if state() == "IDLE" then break end
    tick()
  end
  assert(state() == "IDLE", "result screen never returned to IDLE")
  assert(t - t_end == 1400, "result screen lasted " .. (t - t_end) .. " ms, not 1400")
  return s, press_ms, gf_ms
end

local function row(scenario, difficulty, result, press_ms, gf_ms)
  print(table.concat({ scenario, difficulty, result, tostring(press_ms or ""), tostring(gf_ms or "") }, ","))
end

on_enter({})
tick()
assert(state() == "IDLE")
assert_sub("NORMAL")

-- difficulty cycles both ways: DOWN from SLOW lands on INSANE, UP from INSANE on SLOW
T.button("DOWN"); assert_sub("SLOW")
T.button("DOWN"); assert_sub("INSANE")
T.button("UP");   assert_sub("SLOW")
T.button("UP");   assert_sub("NORMAL")

print("scenario,difficulty,result,press_ms,gf_ms")

-- (a) never press: the fly escapes at the latency the ramp produces
for _, name in ipairs({ "SLOW", "NORMAL", "FAST", "INSANE" }) do
  set_difficulty(name)
  local result, press_ms, gf_ms = round(nil)
  row("a", name, result, press_ms, gf_ms)
end

-- (b) press 100 ms after looming starts, before the fly at 250 ms
set_difficulty("NORMAL")
do
  local result, press_ms, gf_ms = round(100)
  row("b", "NORMAL", result, press_ms, gf_ms)
end

-- (c) press while still ARMED
do
  local result, press_ms, gf_ms = round(nil, true)
  row("c", "NORMAL", result, press_ms, gf_ms)
end

-- (d) lesion on, INSANE, 20 rounds without pressing
T.button("B")
assert(sub_text() == "LC4 + LPLC2 CUT · it cannot see you", "lesion subtitle is " .. sub_text())
set_difficulty("INSANE")
for _ = 1, 20 do
  local result, press_ms, gf_ms = round(nil)
  row("d", "INSANE", result, press_ms, gf_ms)
end

-- (e) lesion off again: the escape comes back
T.button("B")
assert_sub("INSANE")
do
  local result, press_ms, gf_ms = round(nil)
  row("e", "INSANE", result, press_ms, gf_ms)
end

-- persistence: on_exit writes what on_enter reads
on_exit()
assert(store.diff == 4 and store.lesion == 0, "on_exit did not persist difficulty and lesion")

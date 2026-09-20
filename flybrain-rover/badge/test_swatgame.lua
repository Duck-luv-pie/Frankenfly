-- test_swatgame.lua -- play the badge game on a laptop and check what actually happened.
--
--   lua badge/test_swatgame.lua           # every scenario, a CSV summary, exit 0 or 1
--
-- The `badge` table here is a stub that records every widget write and counts every native call,
-- because the firmware is not available off the device but the call pattern is, and the call pattern is
-- where a 20 ms tick goes wrong. tests/test_swatgame_app.py runs this and asserts on the same numbers.
--
-- Nothing here reaches inside the app. It reads the screen: widgets come back in creation order, so the
-- banner, the score, the lives and the alarm bar are all observable exactly as a player sees them. The
-- app carries no test-only accessor.
--
-- The claim this file exists to defend: nothing but a real giant-fibre spike can make a mosquito
-- escape. Every scenario below is timed from the bug's own constant speed, never from a result, so a
-- hidden `if elapsed > N` could not make these pass.

local NCALL, NSHOW, NW = 0, 0, 0
local W = {}                       -- every widget, in creation order
local LED = {}
for i = 1, 6 do LED[i] = {0, 0, 0} end

local function widget(kind, w, h, value, text)
  local o = {kind = kind, w = w or 0, h = h or 0, x = 0, y = 0, bg_opa = 255,
             bg_color = -1, value = value or 0, text = text or "", styles = 0}
  function o:style(t)
    NCALL = NCALL + 1
    if type(t) ~= "table" then error("style() needs a table") end
    if t.bg_color then self.bg_color = t.bg_color end
    if t.bg_opa ~= nil then self.bg_opa = t.bg_opa end
    if t.text_color then self.text_color = t.text_color end
    self.styles = self.styles + 1
    self.seen = self.seen or {}
    self.seen[t] = (self.seen[t] or 0) + 1
    return self
  end
  function o:set_pos(x, y) NCALL = NCALL + 1; self.x, self.y = x, y; return self end
  function o:set_size(w, h) NCALL = NCALL + 1; self.w, self.h = w, h; return self end
  function o:set_value(v) NCALL = NCALL + 1; self.value = v; return self end
  function o:set_text(t) NCALL = NCALL + 1; self.text = t; return self end
  function o:align(_, x, y) NCALL = NCALL + 1; self.x, self.y = x, y; return self end
  NW = NW + 1; W[NW] = o
  return o
end

local MS, SHAKE, STORE, RQ = 0, false, {}, {}

badge = {
  ui = {
    box = function(_, w, h) NCALL = NCALL + 1; return widget("box", w, h) end,
    bar = function(_, _, _, v) NCALL = NCALL + 1; return widget("bar", 0, 0, v) end,
    label = function(_, t) NCALL = NCALL + 1; return widget("label", 0, 0, 0, t) end,
    screen_width = 320, screen_height = 240,
  },
  led = {
    set = function(i, r, g, b) NCALL = NCALL + 1; LED[i] = {r, g, b} end,
    set_all = function(r, g, b) NCALL = NCALL + 1; for i = 1, 6 do LED[i] = {r, g, b} end end,
    clear = function() NCALL = NCALL + 1; for i = 1, 6 do LED[i] = {0, 0, 0} end end,
    show = function() NCALL = NCALL + 1; NSHOW = NSHOW + 1 end,
  },
  input = {BUTTON = {A = 1, B = 2, HOME = 3, START = 4}, KIND = {PRESSED = 1, RELEASED = 2},
           is_down = function() NCALL = NCALL + 1; return false end},
  sensor = {
    accel = function() NCALL = NCALL + 1; return 0, 0, 1000 end,
    shake = function() NCALL = NCALL + 1; local s = SHAKE; SHAKE = false; return s end,
    tap = function() NCALL = NCALL + 1; return false end,
  },
  sys = {
    ms = function() NCALL = NCALL + 1; return MS end,
    uptime = function() NCALL = NCALL + 1; return MS end,
    log = function() NCALL = NCALL + 1 end,
    -- a queue the scenario fills, so every spawn is reproducible; 1 when it runs dry
    random = function(n)
      NCALL = NCALL + 1
      local v = table.remove(RQ, 1) or 1
      return ((v - 1) % n) + 1
    end,
    stats = function() NCALL = NCALL + 1; return {free_heap = 60000, widgets = NW} end,
  },
  store = {
    get_int = function(k, d) NCALL = NCALL + 1; return STORE[k] or d end,
    set_int = function(k, v) NCALL = NCALL + 1; STORE[k] = v end,
    get = function(k, d) NCALL = NCALL + 1; return STORE[k] or d end,
    set = function(k, v) NCALL = NCALL + 1; STORE[k] = v end,
  },
}

dofile("badge/swatgame_app.lua")

-- ---------------------------------------------------------------- the screen, by creation order
-- on_enter builds: bg, ring, PW[1..8], rim, mesh, mh, mv, grip, score, best, life1..3, alarm, say, hint
local BG, RING, PW1, PW8 = 1, 2, 3, 10
local MESH, SCORE, BEST, LIFE1, ALARM, SAY, HINT = 12, 16, 17, 18, 21, 22, 23
local function banner() return W[SAY].text end
local function score_n() return tonumber(W[SCORE].text) or -1 end
local function lives_n()
  local n = 0
  for k = 0, 2 do if W[LIFE1 + k].bg_color == 0xFF715B then n = n + 1 end end
  return n
end
local function alarm_pct() return W[ALARM].value end
local function bug_visible()
  for k = PW1, PW8 do if W[k].bg_opa ~= 0 then return true end end
  return false
end

-- ---------------------------------------------------------------- driving it
local TICK, WORST_SEEN, WORST_WHERE = 20, 0, ""
local function tick(label)
  local before = NCALL
  MS = MS + TICK
  on_tick()
  local used = NCALL - before
  if used > WORST_SEEN then WORST_SEEN, WORST_WHERE = used, label or "?" end
  return used
end

-- Spawn arguments, in the order spawn() consumes them.
--   first spawn of a run:  dir, lane, phase
--   every spawn after:     hen, dir, lane, phase
local function q_first(dir, lane) RQ = {dir, lane, 1} end
local function q_next(hen, dir, lane) for _, v in ipairs({hen, dir, lane, 1}) do RQ[#RQ + 1] = v end end

local function boot(dir, lane)
  -- MS is never rewound. The app's timers are absolute badge.sys.ms() values, so winding the clock
  -- back would leave a lunge_until in the future and silently swallow every shake.
  NCALL, NSHOW, NW, SHAKE, STORE, W = 0, 0, 0, false, {}, {}
  MS = MS + 5000
  on_enter({})
  q_first(dir or 1, lane or 6)       -- lane 6 of 11 -> dead centre
  SHAKE = true; tick("start")        -- leave idle; begin() spawns
end

-- A bug born at x=-34 moving right at 100 px/s reaches the swatter's centre (160) after 1.94 s.
-- Everything below is timed from that, never from a result.
local SX_T = 160
local function is_hen() return W[PW1 + 1].w == 26 end     -- the chicken's body box; the mosquito's is 17

-- Where the animal's centre is, from the box the app actually drew. PW[2] is the chicken's body
-- (offset 0) and the mosquito's abdomen (offset -6, width 17).
local function bug_centre(dir)
  local p = W[PW1 + 1]
  if p.w == 26 then return p.x + 13 end
  return p.x + 8 + dir * 6
end

-- Tick until the animal on screen is within `near` px of the swatter, and report which one it was.
-- Direction is taken from the sprite's own movement, so this works whichever way it entered.
local function approach(near, maxt)
  local dir = 1
  for i = 1, (maxt or 600) do
    local before = W[PW1 + 1].x
    tick("approach")
    local after = W[PW1 + 1].x
    if after ~= before then dir = (after > before) and 1 or -1 end
    if math.abs(bug_centre(dir) - SX_T) <= near then return true, i end
  end
  return false
end

local function ticks_at(px, speed) return math.floor((px + 34) / speed * 1000 / TICK + 0.5) end
local function ticks_to_reach(px) return ticks_at(px, 100) end       -- a mosquito at score 0
local CENTRE = ticks_to_reach(160)

local rows, fails = {}, 0
local function row(...) local t = {} for i, v in ipairs({...}) do t[i] = tostring(v) end
  rows[#rows + 1] = table.concat(t, ",") end
local function check(name, cond, detail)
  if not cond then fails = fails + 1
    io.stderr:write("FAIL  " .. name .. (detail and ("  -- " .. detail) or "") .. "\n") end
end

-- ================================================================ scenarios

-- 1. A mosquito crossing an untouched swatter must never bolt. If it does, the baseline looming is
--    over threshold and the game is unplayable -- which is exactly the bug the browser build had.
do
  boot(1, 6)
  local peak, bolted = 0, false
  for i = 1, CENTRE + 60 do
    tick("quiet")
    if alarm_pct() > peak then peak = alarm_pct() end
    if banner() == "IT GOT AWAY" then bolted = true end
  end
  row("quiet_crossing", "mos", bolted and "BOLTED" or "crossed", "peak_alarm=" .. peak)
  check("an untouched crossing never bolts", not bolted, "baseline looming is over threshold")
end

-- 2. Shake with the mosquito dead centre: it has nowhere to go and you get it.
do
  boot(1, 6)
  check("a mosquito came first", not is_hen())
  approach(5)
  SHAKE = true
  for i = 1, 6 do tick("strike") end            -- the strike lands 60 ms (3 ticks) after the shake
  row("centre_hit", "mos", banner(), "score=" .. score_n())
  check("shaking at the centre gets it", banner() == "GOT IT", "banner was " .. banner())
  check("a hit scores", score_n() == 1, "score " .. score_n())
  check("the mosquito is gone after a hit", not bug_visible())
end

-- 3. Shake while it is still 55 px out: it sees the lunge and clears the head, or you simply miss.
--    Either way it must not be GOT IT.
do
  boot(1, 6)
  approach(55)
  SHAKE = true
  for i = 1, 6 do tick("early_strike") end
  row("early_shake", "mos", banner(), "score=" .. score_n())
  check("shaking early does not get it", banner() ~= "GOT IT", "banner was " .. banner())
  check("an early shake scores nothing", score_n() == 0, "score " .. score_n())
end

-- 4. Swatting a chicken costs a life, whatever the circuit is doing.
do
  boot(1, 6)
  local saw_hen, before_lives = false, lives_n()
  for r = 1, 6 do                                        -- rounds spawn themselves; wait for a chicken
    if approach(5, 600) and is_hen() then saw_hen = true; break end
    for i = 1, 60 do tick("gap") end
  end
  if saw_hen then
    SHAKE = true
    for i = 1, 6 do tick("hen_strike") end
    row("chicken_swatted", "hen", banner(), "lives " .. before_lives .. "->" .. lives_n())
    check("swatting a chicken costs a life", lives_n() < before_lives,
          "lives " .. before_lives .. " -> " .. lives_n())
  else
    row("chicken_swatted", "hen", "NO CHICKEN APPEARED", "")
    check("a chicken turns up within four rounds", false, "since_hen forcing did not produce one")
  end
end

-- 4b. A chicken you leave alone walks off, and costs nothing.
do
  boot(1, 6)
  local before_lives, before_score = lives_n(), score_n()
  local found = false
  for r = 1, 6 do
    if approach(5, 600) and is_hen() then found = true; break end
    for i = 1, 60 do tick("gap") end
  end
  check("a chicken turned up to spare", found)
  for i = 1, 240 do tick("hen_spared") end               -- let it walk all the way off
  row("chicken_spared", "hen", banner(), "lives=" .. lives_n() .. " score=" .. score_n())
  check("sparing a chicken costs nothing", lives_n() == before_lives, "lost a life for doing nothing")
  check("sparing a chicken is acknowledged", banner() == "SPARED IT", "banner was " .. banner())
end

-- 5. With the eye cut the mosquito can never bolt, however loudly you shake at it.
do
  boot(1, 6)
  on_button(badge.input.BUTTON.B, badge.input.KIND.PRESSED)     -- lesion on
  local bolted, peak = false, 0
  for r = 1, 3 do
    for i = 1, CENTRE - 20 do
      tick("lesioned")
      if alarm_pct() > peak then peak = alarm_pct() end
      if banner() == "IT GOT AWAY" or banner() == "IT GOT OUT" then bolted = true end
    end
    SHAKE = true                                               -- shake early, the spooking distance
    for i = 1, 40 do
      tick("lesioned_shake")
      if banner() == "IT GOT AWAY" or banner() == "IT GOT OUT" then bolted = true end
    end
    q_next(9, 1, 6)
    for i = 1, 40 do tick("lesioned_gap") end
  end
  row("lesioned", "mos", bolted and "BOLTED" or "never bolted", "peak_alarm=" .. peak)
  check("a cut eye can never bolt", not bolted, "it escaped with the pathway cut")
end

-- 6. The tick budget. 20 ms is the tick; native calls are the documented laptop proxy for it.
row("budget", "none", "worst=" .. WORST_SEEN, "in=" .. WORST_WHERE)
check("the worst tick stays inside its budget", WORST_SEEN <= 26,
      WORST_SEEN .. " native calls in " .. WORST_WHERE)
row("widgets", "none", NW, "")
check("widget count is sane", NW > 15 and NW < 60, tostring(NW))

io.stderr:write("worst tick " .. WORST_SEEN .. " native calls, in: " .. WORST_WHERE .. "\n")
print("scenario,animal,result,detail")
print(table.concat(rows, "\n"))
if fails > 0 then io.stderr:write(fails .. " FAILED\n"); os.exit(1) end
print("OK")

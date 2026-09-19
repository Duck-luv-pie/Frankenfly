-- test_lua.lua -- run the badge app's integer LIF on a laptop and print a spike trace.
--
--   lua badge/test_lua.lua                  # trace to stdout, as CSV
--   lua badge/test_lua.lua lesion           # same stimulus, LC4/LPLC2 outgoing synapses zeroed
--
-- tests/test_badge_app.py runs this and requires the trace to match badge/sim_fixed.py step for step.
-- That is the only way to know the Lua is right without a badge in hand: the firmware is not available
-- off the device, but the arithmetic is, and the arithmetic is where a fixed-point port goes wrong.
--
-- The `badge` table here is a stub. It exists only so the app file loads; the test drives step() directly.

badge = {
  ui = setmetatable({}, { __index = function() return function() return setmetatable({}, {
    __index = function() return function() end end }) end end }),
  led = { set = function() end, set_all = function() end, clear = function() end,
          show = function() end, count = function() return 6 end },
  input = { BUTTON = { A = 1, B = 2, UP = 3, DOWN = 4, START = 5 }, KIND = { PRESSED = 1, RELEASED = 0 } },
  sensor = { accel = function() return 0, 0, 1000 end, shake = function() return false end },
  sys = { ms = function() return 0 end, stats = function() return { lua_used = 0 } end,
          log = function() end },
  store = { get_int = function(_, d) return d end, set_int = function() end },
  radio = { enable = function() return false end, disable = function() end,
            on_recv = function() end, send = function() end },
}

local f = assert(loadfile("badge/flybadge_app.lua"))
f()
local T = assert(_G.__flybadge_test, "app did not expose its test hook")
local C = T.C

local lesion = (arg[1] == "lesion")
local STEPS = tonumber(arg[2]) or 300
T.reset()
T.set_lesion(lesion)

-- identical ramp to sim_fixed.looming()
local function looming(s, steps)
  if s < steps // 4 then return 0 end
  local t = (s - steps // 4) / math.max(1, steps - steps // 4)
  return math.floor(2600 * t)
end

print("step,cur,fired,gf")
for s = 0, STEPS - 1 do
  local cur = looming(s, STEPS)
  local gf = T.step(cur, false)
  local _, n = T.spikes()
  print(string.format("%d,%d,%d,%d", s, cur, n, gf))
end

--[==[badge-app
slug=fly_pi_probe
name=Fly Pi Probe
icon=RF
api=2
heap_kb=48
wake_lock=1
version=0.1
author=Hunting Fly
]==]

-- Transport probe only. It never controls the robot.
-- A broadcasts a future "lobotomize" test message.
-- B broadcasts a future "restore" test message.

local RETRIES = 6
local RETRY_MS = 220

local status_label
local detail_label
local sequence_label
local radio_ok = false
local sequence = 0
local pending = nil
local pending_action = nil
local attempts = 0
local next_send = 0

local function lights(r, g, b)
  badge.led.set_all(r, g, b)
  badge.led.show()
end

local function queue_probe(action)
  if not radio_ok then
    status_label:set_text("Radio unavailable")
    detail_label:set_text("Cannot queue a probe")
    lights(120, 0, 0)
    return
  end

  sequence = (sequence + 1) % 10000
  pending_action = action
  pending = string.format("FBT1|%s|%04d", action, sequence)
  attempts = 0
  next_send = 0
  status_label:set_text(action == "A" and "A: lobotomize TEST" or "B: restore TEST")
  detail_label:set_text("Sending probe - not confirmed")
  sequence_label:set_text("Sequence " .. string.format("%04d", sequence))
  lights(100, 45, 0)
end

function on_enter(root)
  sequence = badge.sys.random(10000)

  local title = badge.ui.label(root, "Fly -> Pi Radio Probe")
  title:style({text_font = 22, text_color = 0xffffff})
  title:align("top_mid", 0, 18)

  status_label = badge.ui.label(root, "Starting radio...")
  status_label:style({text_font = 20, text_color = 0x66aaff})
  status_label:align("center", 0, -34)

  detail_label = badge.ui.label(root, "This app cannot control the robot")
  detail_label:style({text_font = 16, text_color = 0xc8d0dc})
  detail_label:align("center", 0, 4)

  sequence_label = badge.ui.label(root, "No probe sent")
  sequence_label:style({text_font = 16, text_color = 0x8b96a8})
  sequence_label:align("center", 0, 34)

  local controls = badge.ui.label(root, "A lobotomize test   B restore test")
  controls:style({text_font = 16, text_color = 0xffffff})
  controls:align("bottom_mid", 0, -34)

  local warning = badge.ui.label(root, "QUEUED is not Pi confirmation")
  warning:style({text_font = 14, text_color = 0xffb347})
  warning:align("bottom_mid", 0, -12)

  local ok, err = badge.radio.enable()
  radio_ok = ok == true
  if radio_ok then
    status_label:set_text("Radio ready")
    detail_label:set_text("Run the receiver on the Pi")
    lights(0, 18, 70)
  else
    status_label:set_text("Radio failed")
    detail_label:set_text(tostring(err or "enable returned false"))
    lights(120, 0, 0)
  end
end

function on_tick()
  if not pending then return end
  local now = badge.sys.ms()
  if now < next_send then return end

  local ok, err = badge.radio.send(pending)
  attempts = attempts + 1
  next_send = now + RETRY_MS

  if not ok then
    status_label:set_text("Radio send failed")
    detail_label:set_text(tostring(err or "send returned false"))
    pending = nil
    lights(120, 0, 0)
    return
  end

  detail_label:set_text("Queued " .. attempts .. "/" .. RETRIES .. " - check Pi")
  if attempts >= RETRIES then
    local word = pending_action == "A" and "lobotomize" or "restore"
    status_label:set_text(word .. " TEST queued")
    detail_label:set_text("Sent 6 copies - Pi not confirmed")
    pending = nil
    lights(100, 45, 0)
  end
end

function on_button(button, kind)
  if kind ~= badge.input.KIND.PRESSED then return end
  local B = badge.input.BUTTON
  if button == B.A then
    queue_probe("A")
  elseif button == B.B then
    queue_probe("B")
  end
end

function on_exit()
  pending = nil
  if radio_ok then badge.radio.disable() end
  badge.led.clear()
  badge.led.show()
end

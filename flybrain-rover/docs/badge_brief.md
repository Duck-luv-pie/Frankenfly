# Hacker Badge 2026 — app brief (saved from Taka's paste, Sat morning)

Key constraints for anything we put on the badge (full brief on badge.hackthenorth.com/ide, "Download guide"):
- Lua app, single file: `--[==[badge-app ... ]==]` header (slug, name, icon, api=2, heap_kb=48) then `main.lua`. Import app → Connect → Push in the IDE (Chrome/Edge, USB data cable, badge on, don't hold Start).
- 320x240 screen via `badge.ui` widgets only (label/box/bar/arc/line/...), physical buttons via `on_button` (A B HOME DOWN LEFT RIGHT UP AUX1 START), 20 ms `on_tick` (budget 250 ms), `on_enter` 3 s, `on_button` 1 s, `on_exit` 1 s.
- 6 RGB LEDs (`badge.led.set(i, r, g, b)`, indices 1-6: 1 upper-left, 2 upper-right, 3 mid-right, 4 bottom-right, 5 bottom-left, 6 mid-left; stage then `show()`).
- Accelerometer (`badge.sensor.accel()` mg, `shake()`, `tap()`, `orientation()`), 48 KiB Lua heap (96 max), 512 widgets, 64 KiB main.lua.
- Radio: badge-to-badge only, 44-byte payloads, `LUA1` prefix, BLE; NO Wi-Fi/HTTP, so the badge cannot receive the laptop's live brain feed.
- Storage: `badge.store` (32 keys) and `badge.fs` (64 KiB). Identity: `badge.me.name()/role_name()/color()/badge_id()`.
- No pcall/coroutine/os/io; no sleep; integer coordinates; ASCII text only (no em dashes/emoji in labels).
Manual: power switch on top, QR on boot is the ID, Scanner app for stickers, Connect app to bump badges, Sync to back up, Badge Help Desk in the QNX Makerspace, badge workshop Sat 11:30.

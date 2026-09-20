#pragma once
#include <Arduino.h>

// S-Bus output to the RoboMaster S1's motion controller (docs/wiring.md, "RoboMaster S1"): the brain
// sends the seven stick / switch channels in its packet ("s1": {"ch": [...]}, 1024 +/- 672); this class
// streams them as inverted 100000-baud 8E2 frames every 14 ms on PIN_SBUS_TX. No packet for
// SBUS_TIMEOUT_MS -> channels 1-4 centred and the frame-lost flag set: the rover stops.
class SBusOut {
 public:
  void begin();
  void set(const uint16_t *ch, size_t n);   // channels 1..n (n <= 16), the rest stay centred
  void tick();                              // call from loop(); sends a frame when 14 ms have passed
  bool live() const { return lastSet_ && millis() - lastSet_ < SBUS_TIMEOUT_MS; }
  uint32_t frames() const { return frames_; }
  bool streaming() const { return haveAny_; }
  const uint16_t *channels() const { return ch_; }

 private:
  static const uint32_t SBUS_TIMEOUT_MS = 500;
  void frame(uint8_t *out, bool lost) const;
  HardwareSerial ser_{1};
  uint16_t ch_[16];
  uint32_t lastSet_ = 0, lastFrame_ = 0, frames_ = 0;
  bool haveAny_ = false;
};

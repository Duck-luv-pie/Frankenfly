#pragma once
#include <Arduino.h>

class Pir {
 public:
  void begin();
  void poll();
  bool rose() { bool r = rose_; rose_ = false; return r; }   // true once per rising edge
  int state() const { return state_; }
  bool ready() const { return millis() > warmupUntil_; }

 private:
  int state_ = 0;
  bool rose_ = false;
  uint32_t lastChange_ = 0, warmupUntil_ = 0;
};

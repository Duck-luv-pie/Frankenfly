#pragma once
#include <Arduino.h>
#include <Arduino_GFX_Library.h>

struct EyeParams {
  float px = 0, py = 0;       // pupil position -1..1
  float pr = 0.45f;           // pupil radius 0..1
  float ut = 0.85f, lt = 0.85f;  // upper / lower lid openness 0..1
  float tilt = 0.0f;         // upper-lid slope -0.6..0.6
  uint8_t r = 90, g = 200, b = 255;  // iris tint
};

class Eye {
 public:
  Eye(Arduino_GFX *gfx) : gfx_(gfx) {}
  void begin();
  void setTarget(const EyeParams &p) { target_ = p; }
  void blink() { blink_phase_ = 1; }
  void tick();            // ease toward target and redraw what changed
  EyeParams current() const { return cur_; }
  void testPattern();

 private:
  void draw(bool full);
  Arduino_GFX *gfx_;
  EyeParams cur_, target_, drawn_;
  bool first_ = true;
  int blink_phase_ = 0;   // 0 none, 1 closing, 2 opening
  float blink_ = 0;       // 1 = fully closed
};

class Eyes {
 public:
  void begin();
  void setTargets(const EyeParams &l, const EyeParams &r) { left_->setTarget(l); right_->setTarget(r); }
  void blink() { left_->blink(); right_->blink(); }
  void tick();
  void idle(uint32_t now_ms);       // built-in wander + random blinks when the brain is away
  void startle();                    // local PIR reflex: eyes wide
  void testPattern();

 private:
  Eye *left_ = nullptr, *right_ = nullptr;
  uint32_t last_frame_ = 0, next_blink_ = 0, next_wander_ = 0;
  float wx_ = 0, wy_ = 0;
};

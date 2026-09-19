#include "pir.h"
#include "config.h"

void Pir::begin() {
  pinMode(PIN_PIR, INPUT);
  warmupUntil_ = millis() + PIR_WARMUP_MS;
}

void Pir::poll() {
  if (!ready()) return;
  int s = digitalRead(PIN_PIR);
  uint32_t now = millis();
  if (s != state_ && now - lastChange_ > PIR_DEBOUNCE_MS) {
    lastChange_ = now;
    if (s && !state_) rose_ = true;
    state_ = s;
    Serial.printf("PIR %d\n", s);
  }
}

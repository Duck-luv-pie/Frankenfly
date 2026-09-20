#include "sbus.h"
#include "config.h"

static const uint16_t CENTER = 1024;

void SBusOut::begin() {
  for (auto &c : ch_) c = CENTER;
  ser_.begin(100000, SERIAL_8E2, -1, PIN_SBUS_TX, true);   // inverted TX = S-Bus, no RX
}

void SBusOut::set(const uint16_t *ch, size_t n) {
  for (size_t i = 0; i < 16; i++) ch_[i] = i < n ? (ch[i] & 0x7FF) : CENTER;
  lastSet_ = millis();
  haveAny_ = true;
}

void SBusOut::frame(uint8_t *out, bool lost) const {
  memset(out, 0, 25);
  out[0] = 0x0F;
  uint32_t bits = 0; int nb = 0; int idx = 1;
  for (int c = 0; c < 16; c++) {
    uint16_t v = ch_[c];
    if (lost && c < 4) v = CENTER;                     // sticks centred, switches (mode, release) as they were
    bits |= (uint32_t)(v & 0x7FF) << nb; nb += 11;
    while (nb >= 8) { out[idx++] = bits & 0xFF; bits >>= 8; nb -= 8; }
  }
  out[23] = lost ? 0x04 : 0x00;                        // frame-lost flag
  out[24] = 0x00;
}

void SBusOut::tick() {
  if (!haveAny_) return;                               // nothing until the brain has asked for the rover once
  uint32_t now = millis();
  if (now - lastFrame_ < 14) return;
  lastFrame_ = now;
  uint8_t f[25];
  frame(f, !live());
  ser_.write(f, 25);
  frames_++;
}

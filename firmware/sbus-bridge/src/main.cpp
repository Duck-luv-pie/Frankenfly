// S-Bus inverter bridge. USB serial in (UART0, 100000 8E2, non-inverted, from the Pi) -> GPIO17 out
// (UART2, 100000 8E2, inverted = real S-Bus) to the S1 motion controller's S-Bus Signal pin.
// Failsafe: no complete frame for 500 ms -> keep the last frame's mode switches, centre channels 1-4,
// set the frame-lost flag, 70 Hz. The blue LED is on while frames are flowing.
#include <Arduino.h>

static const int PIN_SBUS_TX = 17;
static const int PIN_LED = 2;
static const uint32_t TIMEOUT_MS = 500;
static const uint32_t FRAME_MS = 14;
static const uint16_t CENTER = 1024;

HardwareSerial sbus(2);
static uint8_t frame[25];
static uint8_t last[25];
static size_t n = 0;
static bool haveLast = false;
static uint32_t lastFrameMs = 0, lastSafeMs = 0, lastReportMs = 0;
static uint32_t frames = 0, bad = 0;

static void unpack(const uint8_t *f, uint16_t *ch) {
  uint32_t bits = 0; int nb = 0; int idx = 1;
  for (int c = 0; c < 16; c++) {
    while (nb < 11) { bits |= (uint32_t)f[idx++] << nb; nb += 8; }
    ch[c] = bits & 0x7FF; bits >>= 11; nb -= 11;
  }
}

static void pack(const uint16_t *ch, uint8_t flags, uint8_t *f) {
  memset(f, 0, 25); f[0] = 0x0F;
  uint32_t bits = 0; int nb = 0; int idx = 1;
  for (int c = 0; c < 16; c++) {
    bits |= (uint32_t)(ch[c] & 0x7FF) << nb; nb += 11;
    while (nb >= 8) { f[idx++] = bits & 0xFF; bits >>= 8; nb -= 8; }
  }
  f[23] = flags; f[24] = 0;
}

static void sendSafe() {
  uint16_t ch[16];
  if (haveLast) unpack(last, ch); else for (int i = 0; i < 16; i++) ch[i] = CENTER;
  for (int i = 0; i < 4; i++) ch[i] = CENTER;          // sticks centred, switches as they were
  uint8_t f[25];
  pack(ch, 0x04, f);                                    // frame-lost flag
  sbus.write(f, 25);
}

void setup() {
  pinMode(PIN_LED, OUTPUT);
  Serial.begin(100000, SERIAL_8E2);                     // from the Pi, over USB
  sbus.begin(100000, SERIAL_8E2, -1, PIN_SBUS_TX, true); // inverted TX = S-Bus
  lastFrameMs = lastSafeMs = millis();
}

void loop() {
  while (Serial.available()) {
    uint8_t b = Serial.read();
    if (n == 0 && b != 0x0F) continue;                  // sync on the header
    frame[n++] = b;
    if (n == 25) {
      n = 0;
      if (frame[24] != 0x00) { bad++; continue; }       // bad footer: resync
      sbus.write(frame, 25);
      memcpy(last, frame, 25);
      haveLast = true;
      frames++;
      lastFrameMs = millis();
    }
  }
  uint32_t now = millis();
  bool flowing = (now - lastFrameMs) < TIMEOUT_MS;
  if (now - lastReportMs >= 1000) {                     // status back up the USB cable, once a second
    lastReportMs = now;
    Serial.printf("bridge frames=%lu bad=%lu flowing=%d\n", (unsigned long)frames, (unsigned long)bad, flowing ? 1 : 0);
  }
  digitalWrite(PIN_LED, flowing ? HIGH : LOW);
  if (!flowing && now - lastSafeMs >= FRAME_MS) { sendSafe(); lastSafeMs = now; }
}

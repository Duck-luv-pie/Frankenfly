// Companion "body": ESP32 DevKit driving two GC9A01 eyes, a DFPlayer Mini, a PIR sensor and, on the
// robot, the RoboMaster S1 over S-Bus, expressing whatever the fly brain decides. Wiring: docs/wiring.md
#include <Arduino.h>
#include "config.h"
#include "eyes.h"
#include "audio.h"
#include "pir.h"
#include "link.h"
#include "sbus.h"

static Eyes eyes;
static Audio audio;
static Pir pir;
static Link brainLink;
static SBusOut sbus;   // the RoboMaster S1, when the brain sends "s1" channels
static BrainPacket brain;
static uint32_t lastHeartbeat = 0;
static uint32_t startleUntil = 0;

void setup() {
  Serial.setRxBufferSize(8192);       // brain packets over USB are ~450 bytes at 20 Hz; the default 256 would chop them
  Serial.begin(BODY_SERIAL_BAUD);
  Serial.println("\ncompanion body");
  eyes.begin();
  eyes.testPattern();
  delay(800);
  audio.begin();
  pir.begin();
  brainLink.begin();
  sbus.begin();
  audio.play(1, DF_DEFAULT_VOLUME);  // boot sound: confirms the SD card works
  Serial.println("ready");
}

void loop() {
  uint32_t now = millis();

  bool fresh = brainLink.poll(brain);
  if (brainLink.pollSerial(brain)) fresh = true;     // the same packets over USB (bench: tools/eyes_demo.py)
  if (fresh) {
    eyes.setTargets(brain.l, brain.r);
    if (brain.blink) eyes.blink();
    if (brain.track > 0) audio.play(brain.track, brain.volume);
    if (brain.hasS1) sbus.set(brain.s1, brain.s1n);
  }
  sbus.tick();

  pir.poll();
  if (pir.rose()) {
    startleUntil = now + 400;   // local reflex: widen immediately, don't wait for the brain
    eyes.startle();
    brainLink.send(pir.state(), audio.busy() ? 1 : 0);
  }

  if (!brainLink.hostAlive() && now > startleUntil) eyes.idle(now);
  eyes.setLeftEnabled(!brainLink.hostAlive());   // DEBUG: left eye off while the brain's packets are arriving, on when idling
  eyes.tick();

  if (now - lastHeartbeat > HEARTBEAT_MS) {
    lastHeartbeat = now;
    brainLink.send(pir.state(), audio.busy() ? 1 : 0);
  }
  static uint32_t lastSbusLog = 0;
  if (now - lastSbusLog > 5000) {                 // the S-Bus side of the wire, in numbers (tools/pi_body_log.sh)
    lastSbusLog = now;
    if (sbus.streaming()) {
      const uint16_t *c = sbus.channels();
      Serial.printf("[sbus] GPIO %d: %lu frames, brain %s, ch1-7 = %u %u %u %u %u %u %u\n", PIN_SBUS_TX, (unsigned long)sbus.frames(),
                    sbus.live() ? "live" : "LOST (sticks centred)", c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    } else {
      Serial.printf("[sbus] GPIO %d: idle, no s1 channels received yet (brain %s)\n", PIN_SBUS_TX, brainLink.hostAlive() ? "alive" : "absent");
    }
  }
}

#pragma once
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "eyes.h"

struct BrainPacket {
  uint32_t t = 0;
  char state[16] = "idle";
  EyeParams l, r;
  bool blink = false;
  int track = 0;
  int volume = -1;
  float armL = 0, armR = 0;   // reserved for future arms
  bool hasS1 = false;         // "s1": {"ch": [...]}: S-Bus channels for the RoboMaster S1 (1024 +/- 672)
  uint16_t s1[16];
  size_t s1n = 0;
};

class Link {
 public:
  void begin();
  bool poll(BrainPacket &out);            // true when a new brain packet arrived (UDP)
  bool pollSerial(BrainPacket &out);      // the same packets as JSON lines over USB serial (bench: tools/eyes_demo.py)
  void send(int pir, int busy);           // status packet to the brain
  bool hostAlive() const { return lastRx_ && millis() - lastRx_ < 2000; }
  uint32_t lastRx() const { return lastRx_; }

 private:
  WiFiUDP udp_;
  IPAddress host_;
  bool haveHost_ = false;
  uint32_t lastRx_ = 0;
  bool parse(const char *buf, size_t len, BrainPacket &out);
  void retryWifi();
  char buf_[1024];
  char sbuf_[1024];
  size_t slen_ = 0;
  uint32_t lastWifiTry_ = 0;
  uint32_t serialOk_ = 0;
};

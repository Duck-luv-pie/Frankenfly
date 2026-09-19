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
};

class Link {
 public:
  void begin();
  bool poll(BrainPacket &out);            // true when a new brain packet arrived
  void send(int pir, int busy);           // status packet to the brain
  bool hostAlive() const { return lastRx_ && millis() - lastRx_ < 2000; }
  uint32_t lastRx() const { return lastRx_; }

 private:
  WiFiUDP udp_;
  IPAddress host_;
  bool haveHost_ = false;
  uint32_t lastRx_ = 0;
  char buf_[1024];
};

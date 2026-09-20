#pragma once
#include <Arduino.h>

class Audio {
 public:
  void begin();
  void play(int track, int volume);   // track 0 = no-op; repeated requests for the playing track are ignored
  bool busy() const;
  bool online() const { return online_; }

 private:
  bool online_ = false;
  int lastTrack_ = 0;
  uint32_t lastStart_ = 0;
  int volume_ = -1;
};

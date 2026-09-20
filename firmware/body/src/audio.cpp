#include "audio.h"
#include "config.h"
#include <DFRobotDFPlayerMini.h>

static DFRobotDFPlayerMini player;

void Audio::begin() {
  pinMode(PIN_DF_BUSY, INPUT);
  Serial2.begin(9600, SERIAL_8N1, PIN_DF_RX, PIN_DF_TX);
  delay(300);  // DFPlayer boot
  online_ = player.begin(Serial2, /*isACK=*/true, /*doReset=*/true);
  if (online_) {
    player.volume(DF_DEFAULT_VOLUME);
    volume_ = DF_DEFAULT_VOLUME;
    Serial.println("DFPlayer online");
  } else {
    Serial.println("DFPlayer not responding (check SD card, 5V, RX/TX swap)");
  }
}

bool Audio::busy() const { return online_ && digitalRead(PIN_DF_BUSY) == LOW; }

void Audio::play(int track, int volume) {
  if (!online_ || track <= 0) return;
  if (volume >= 0 && volume != volume_) {
    player.volume(constrain(volume, 0, 30));
    volume_ = volume;
  }
  uint32_t now = millis();
  if (track == lastTrack_ && (busy() || now - lastStart_ < 300)) return;  // already playing it
  player.playMp3Folder(track);   // /mp3/0001.mp3 ...
  lastTrack_ = track;
  lastStart_ = now;
}

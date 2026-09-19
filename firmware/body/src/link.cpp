#include "link.h"
#include "config.h"
#include <ArduinoJson.h>
#include <ESPmDNS.h>

#ifndef WIFI_SSID
#error "Create firmware/secrets.ini from secrets.example.ini with WIFI_SSID / WIFI_PASS"
#endif

void Link::begin() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setHostname(HOSTNAME);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("connecting to %s", WIFI_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(250);
    Serial.print('.');
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nIP %s\n", WiFi.localIP().toString().c_str());
    if (MDNS.begin(HOSTNAME)) MDNS.addService("companion", "udp", UDP_LISTEN_PORT);
  } else {
    Serial.println("\nWiFi not connected; running with local idle animation only");
  }
  udp_.begin(UDP_LISTEN_PORT);
}

static void readEye(JsonObjectConst o, EyeParams &e) {
  if (o.isNull()) return;
  e.px = o["px"] | e.px;
  e.py = o["py"] | e.py;
  e.pr = o["pr"] | e.pr;
  e.ut = o["ut"] | e.ut;
  e.lt = o["lt"] | e.lt;
  JsonArrayConst t = o["tint"];
  if (t.size() == 3) { e.r = t[0]; e.g = t[1]; e.b = t[2]; }
}

bool Link::poll(BrainPacket &out) {
  bool got = false;
  int n;
  while ((n = udp_.parsePacket()) > 0) {
    int len = udp_.read(buf_, sizeof(buf_) - 1);
    if (len <= 0) continue;
    buf_[len] = 0;
    host_ = udp_.remoteIP();
    haveHost_ = true;
    JsonDocument doc;
    if (deserializeJson(doc, buf_, len)) continue;
    out.t = doc["t"] | 0;
    strlcpy(out.state, doc["state"] | "idle", sizeof(out.state));
    readEye(doc["eyes"]["l"], out.l);
    readEye(doc["eyes"]["r"], out.r);
    out.blink = doc["blink"] | false;
    out.track = doc["sound"]["track"] | 0;
    out.volume = doc["sound"]["vol"] | -1;
    out.armL = doc["arms"]["l"] | 0.0f;
    out.armR = doc["arms"]["r"] | 0.0f;
    lastRx_ = millis();
    got = true;
  }
  return got;
}

void Link::send(int pir, int busy) {
  if (!haveHost_ || WiFi.status() != WL_CONNECTED) return;
  char msg[96];
  snprintf(msg, sizeof(msg), "{\"t\":%lu,\"pir\":%d,\"busy\":%d,\"rssi\":%d}", (unsigned long)millis(), pir, busy, WiFi.RSSI());
  udp_.beginPacket(host_, UDP_REPLY_PORT);
  udp_.write((const uint8_t *)msg, strlen(msg));
  udp_.endPacket();
}

#include "link.h"
#include "config.h"
#include <ArduinoJson.h>
#include <ESPmDNS.h>

#ifndef WIFI_SSID
#error "Create firmware/secrets.ini from secrets.example.ini with WIFI_SSID / WIFI_PASS"
#endif

struct KnownNet { const char *ssid; const char *pass; };
static const KnownNet KNOWN[] = {
  {WIFI_SSID, WIFI_PASS},
#ifdef WIFI_SSID2
  {WIFI_SSID2, WIFI_PASS2},
#endif
#ifdef WIFI_SSID3
  {WIFI_SSID3, WIFI_PASS3},
#endif
};

static bool joinKnown(uint32_t waitMs) {
  WiFi.disconnect(true, false);
  delay(100);
  int n = WiFi.scanNetworks(false, false, false, 300);
  const KnownNet *chosen = nullptr;
  int best = -1000;
  for (int i = 0; i < n; i++)
    for (const KnownNet &k : KNOWN)
      if (WiFi.SSID(i) == k.ssid && WiFi.RSSI(i) > best) { best = WiFi.RSSI(i); chosen = &k; }
  WiFi.scanDelete();
  if (!chosen) { Serial.printf("scan: %d networks, none known\n", n); return false; }
  Serial.printf("joining %s (%d dBm)", chosen->ssid, best);
  WiFi.begin(chosen->ssid, chosen->pass);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < waitMs) { delay(250); Serial.print('.'); }
  Serial.println();
  return WiFi.status() == WL_CONNECTED;
}

void Link::begin() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setHostname(HOSTNAME);
  // Every network the body may meet (secrets.ini: WIFI_SSID / WIFI_PASS, then WIFI_SSID2 / 3: a phone
  // hotspot, the Pi's own hotspot "companion" on the robot): scan, join the strongest known one.
  joinKnown(10000);                       // one try now; retried every 30 s from poll() (the eyes work meanwhile)
  lastWifiTry_ = millis();
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
  e.tilt = constrain(o["tilt"] | 0.0f, -0.6f, 0.6f);
  JsonArrayConst t = o["tint"];
  if (t.size() == 3) { e.r = t[0]; e.g = t[1]; e.b = t[2]; }
}

void Link::retryWifi() {
  if (WiFi.status() == WL_CONNECTED || millis() - lastWifiTry_ < 30000) return;
  lastWifiTry_ = millis();
  if (joinKnown(8000)) {
    Serial.printf("IP %s\n", WiFi.localIP().toString().c_str());
    if (MDNS.begin(HOSTNAME)) MDNS.addService("companion", "udp", UDP_LISTEN_PORT);
    udp_.begin(UDP_LISTEN_PORT);
  }
}

bool Link::pollSerial(BrainPacket &out) {
  bool got = false;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (slen_ > 0 && sbuf_[0] == '{') {
        sbuf_[slen_] = 0;
        if (parse(sbuf_, slen_, out)) { got = true; if (++serialOk_ % 100 == 1) Serial.printf("[link] serial packet #%lu ok (%u bytes)\n", (unsigned long)serialOk_, (unsigned)slen_); }
        else { Serial.printf("[link] serial packet rejected (%u bytes): %.40s\n", (unsigned)slen_, sbuf_); }
      } else if (slen_ > 0) {
        Serial.printf("[link] serial line ignored (%u bytes, starts '%c')\n", (unsigned)slen_, sbuf_[0]);
      }
      slen_ = 0;
    } else if (slen_ < sizeof(sbuf_) - 1) {
      sbuf_[slen_++] = c;
    } else {
      slen_ = 0;                           // overlong line: drop it
    }
  }
  return got;
}

bool Link::parse(const char *buf, size_t len, BrainPacket &out) {
  JsonDocument doc;
  if (deserializeJson(doc, buf, len)) return false;
  out.t = doc["t"] | 0;
  strlcpy(out.state, doc["state"] | "idle", sizeof(out.state));
  readEye(doc["eyes"]["l"], out.l);
  readEye(doc["eyes"]["r"], out.r);
  out.blink = doc["blink"] | false;
  out.track = doc["sound"]["track"] | 0;
  out.volume = doc["sound"]["vol"] | -1;
  out.armL = doc["arms"]["l"] | 0.0f;
  out.armR = doc["arms"]["r"] | 0.0f;
  JsonArrayConst ch = doc["s1"]["ch"];
  out.hasS1 = !ch.isNull() && ch.size() > 0;
  out.s1n = 0;
  if (out.hasS1) for (JsonVariantConst v : ch) { if (out.s1n < 16) out.s1[out.s1n++] = (uint16_t)(v.as<int>()); }
  lastRx_ = millis();
  return true;
}

bool Link::poll(BrainPacket &out) {
  retryWifi();
  bool got = false;
  int n;
  while ((n = udp_.parsePacket()) > 0) {
    int len = udp_.read(buf_, sizeof(buf_) - 1);
    if (len <= 0) continue;
    buf_[len] = 0;
    host_ = udp_.remoteIP();
    haveHost_ = true;
    if (parse(buf_, len, out)) got = true;
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

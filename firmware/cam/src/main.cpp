// Companion "cam": the fly's retina.
// Serves the OV2640 as an MJPEG stream on http://companion-cam.local:81/stream
// and a JSON status on http://companion-cam.local/status.
// Board: AI-Thinker ESP32-CAM on its USB carrier. Nothing else is wired to this board.

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <ESPmDNS.h>
#include <esp_camera.h>
#include <esp_http_server.h>
#include <esp_task_wdt.h>
#include <esp_wifi.h>

#ifndef WIFI_SSID
#error "Create firmware/secrets.ini from secrets.example.ini with WIFI_SSID / WIFI_PASS"
#endif

static const char *HOSTNAME = "companion-cam";
static WiFiMulti wifiMulti;

// AI-Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define FLASH_LED_GPIO 4

static httpd_handle_t stream_httpd = nullptr;
static httpd_handle_t status_httpd = nullptr;
static volatile uint32_t frames_served = 0;
static uint32_t boot_ms = 0;

static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";

static esp_err_t stream_handler(httpd_req_t *req) {
  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  // One TCP write per frame. Sending boundary, part header and JPEG as three small writes
  // trips Nagle's algorithm against the client's delayed ACK and stalls the stream for ~200 ms.
  static uint8_t *packet = nullptr;
  static size_t packet_cap = 0;
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("camera capture failed");
      res = ESP_FAIL;
      break;
    }
    char part[96];
    size_t hlen = snprintf(part, sizeof(part), "\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    size_t need = hlen + fb->len;
    if (need > packet_cap) {
      free(packet);
      packet = (uint8_t *)malloc(need + 4096);   // internal RAM: small frames, no PSRAM cache surprises
      packet_cap = packet ? need + 4096 : 0;
    }
    if (packet) {
      memcpy(packet, part, hlen);
      memcpy(packet + hlen, fb->buf, fb->len);
      esp_camera_fb_return(fb);
      res = httpd_resp_send_chunk(req, (const char *)packet, need);
    } else {
      res = httpd_resp_send_chunk(req, part, hlen);
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      esp_camera_fb_return(fb);
    }
    if (res != ESP_OK) break;  // client went away
    frames_served++;
  }
  return res;
}

static esp_err_t status_handler(httpd_req_t *req) {
  char buf[200];
  snprintf(buf, sizeof(buf),
           "{\"uptime_s\":%lu,\"frames\":%lu,\"heap\":%u,\"psram\":%u,\"rssi\":%d,\"ip\":\"%s\"}",
           (unsigned long)((millis() - boot_ms) / 1000), (unsigned long)frames_served, ESP.getFreeHeap(),
           ESP.getFreePsram(), WiFi.RSSI(), WiFi.localIP().toString().c_str());
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, buf, strlen(buf));
}

static void start_servers() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.lru_purge_enable = true;
  config.recv_wait_timeout = 5;   // seconds; a client that stops reading is dropped, not waited on forever
  config.send_wait_timeout = 5;
  httpd_uri_t status_uri = {.uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = nullptr};
  if (httpd_start(&status_httpd, &config) == ESP_OK) httpd_register_uri_handler(status_httpd, &status_uri);

  config.server_port = 81;
  config.ctrl_port = 32769;
  httpd_uri_t stream_uri = {.uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = nullptr};
  if (httpd_start(&stream_httpd, &config) == ESP_OK) httpd_register_uri_handler(stream_httpd, &stream_uri);
}

static bool init_camera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_QVGA;   // 320x240: the person detector wants this much; the optic lobe resizes to 80x60
  c.jpeg_quality = 16;   // smaller frames stream faster; the brain only uses 80x60 of it
  c.fb_count = psramFound() ? 3 : 1;
  c.grab_mode = CAMERA_GRAB_LATEST;  // always serve the freshest frame (low latency)
  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_framesize(s, FRAMESIZE_QVGA);
    s->set_vflip(s, 0);
    s->set_hmirror(s, 0);
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  boot_ms = millis();
  esp_task_wdt_init(15, true);        // if loop() stops being called for 15 s, reboot
  esp_task_wdt_add(nullptr);
  esp_task_wdt_reset();
  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);  // keep the bright flash LED off

  if (!init_camera()) {
    Serial.println("camera init failed - check the ribbon cable, then press RST");
    while (true) delay(1000);
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setHostname(HOSTNAME);
  // Every network the camera may meet (secrets.ini: WIFI_SSID / WIFI_PASS, then WIFI_SSID2 / 3 for a phone
  // hotspot or the Pi's own hotspot); WiFiMulti joins whichever is in range, strongest first.
  wifiMulti.addAP(WIFI_SSID, WIFI_PASS);
#ifdef WIFI_SSID2
  wifiMulti.addAP(WIFI_SSID2, WIFI_PASS2);
#endif
#ifdef WIFI_SSID3
  wifiMulti.addAP(WIFI_SSID3, WIFI_PASS3);
#endif
  Serial.print("connecting to a known WiFi");
  uint32_t t0 = millis();
  while (wifiMulti.run(8000) != WL_CONNECTED) {
    esp_task_wdt_reset();                 // joining can take a while; don't let the watchdog reboot us mid-connect
    Serial.print('.');
    if (millis() - t0 > 90000) { Serial.println("\nno WiFi after 90 s, rebooting"); ESP.restart(); }
  }
  WiFi.setSleep(false);
  Serial.printf("\njoined %s, IP %s\n", WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());
  if (MDNS.begin(HOSTNAME)) {
    MDNS.addService("http", "tcp", 81);
    Serial.printf("stream: http://%s.local:81/stream\n", HOSTNAME);
  }
  start_servers();
}

void loop() {
  static uint32_t last = 0, wifi_lost_since = 0;
  esp_task_wdt_reset();
  if (WiFi.status() != WL_CONNECTED) {
    if (!wifi_lost_since) wifi_lost_since = millis();
    else if (millis() - wifi_lost_since > 20000) { Serial.println("WiFi lost for 20 s, rebooting"); ESP.restart(); }
  } else {
    wifi_lost_since = 0;
  }
  if (millis() - last > 10000) {
    last = millis();
    Serial.printf("up %lus frames %lu heap %u rssi %d\n", (unsigned long)(last / 1000),
                  (unsigned long)frames_served, ESP.getFreeHeap(), WiFi.RSSI());
    if (WiFi.status() != WL_CONNECTED) wifiMulti.run(8000);
  }
  delay(100);
}

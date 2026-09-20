// Companion body: pins, ports and timing. Wiring tables: docs/wiring.md
#pragma once

// --- Eyes: two GC9A01 240x240 round TFTs on the shared VSPI bus ---------------------------
#define PIN_TFT_SCK   18   // both displays: SCL
#define PIN_TFT_MOSI  23   // both displays: SDA
#define PIN_TFT_DC    27   // both displays: DC
#define PIN_TFT_RST   26   // both displays: RES
#define PIN_TFT_CS_L   5   // left eye CS
#define PIN_TFT_CS_R  25   // right eye CS
#define PIN_TFT_BL    -1   // backlight PWM (set to 32 if BLK is wired there; -1 = tied to 3V3)
#define TFT_SPI_HZ    40000000
#define EYE_SIZE      240

// --- DFPlayer Mini on UART2 ---------------------------------------------------------------
#define PIN_DF_RX     16   // ESP32 RX2  <- DFPlayer TX
#define PIN_DF_TX     17   // ESP32 TX2  -> 1 kOhm -> DFPlayer RX
#define PIN_DF_BUSY   35   // DFPlayer BUSY (low while playing), input-only pin
#define DF_DEFAULT_VOLUME 20   // 0..30

// --- RoboMaster S1 over S-Bus (docs/wiring.md, "RoboMaster S1") ------------------------------
#define PIN_SBUS_TX    4   // -> S1 motion controller S-Bus "Signal" pin (inverted UART1 TX); GND to GND, 5 V pin unused

// --- HC-SR501 PIR ---------------------------------------------------------------------------
#define PIN_PIR       34   // input-only pin
#define PIR_WARMUP_MS 60000
#define PIR_DEBOUNCE_MS 200

// --- Network --------------------------------------------------------------------------------
#define HOSTNAME        "companion-body"
#define UDP_LISTEN_PORT 4210   // brain -> body
#define UDP_REPLY_PORT  4211   // body  -> brain
#define HEARTBEAT_MS    500
#define HOST_TIMEOUT_MS 2000   // no packet for this long -> local idle animation

// --- Animation --------------------------------------------------------------------------------
#define EYE_FPS        30
#define EASE_PER_FRAME 0.18f   // fraction of remaining distance closed per frame (~0.2 s glide at 30 fps)

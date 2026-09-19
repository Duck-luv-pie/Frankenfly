// Eye rendering for GC9A01 without a frame buffer: the eye is a few filled circles and two
// lid rectangles. Each frame we erase the previous pupil/lids by redrawing the iris/sclera
// under them, then draw the new ones. Cheap enough for 30 fps on two displays.
#include "eyes.h"
#include "config.h"

static const uint16_t SCLERA = 0xFFFF;     // white
static const uint16_t BG = 0x0000;         // outside the eye
static const uint16_t PUPIL = 0x0000;
static const uint16_t LID = 0x1082;        // dark grey skin

static inline uint16_t rgb(uint8_t r, uint8_t g, uint8_t b) { return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3); }
static inline float ease(float cur, float tgt) { return cur + (tgt - cur) * EASE_PER_FRAME; }

void Eye::begin() {
  gfx_->begin(TFT_SPI_HZ);
  gfx_->fillScreen(BG);
  first_ = true;
}

void Eye::testPattern() {
  gfx_->fillScreen(BG);
  gfx_->fillCircle(EYE_SIZE / 2, EYE_SIZE / 2, EYE_SIZE / 2 - 2, SCLERA);
  gfx_->fillCircle(EYE_SIZE / 2, EYE_SIZE / 2, 60, rgb(90, 200, 255));
  gfx_->fillCircle(EYE_SIZE / 2, EYE_SIZE / 2, 30, PUPIL);
}

void Eye::tick() {
  cur_.px = ease(cur_.px, target_.px);
  cur_.py = ease(cur_.py, target_.py);
  cur_.pr = ease(cur_.pr, target_.pr);
  cur_.ut = ease(cur_.ut, target_.ut);
  cur_.lt = ease(cur_.lt, target_.lt);
  cur_.tilt = ease(cur_.tilt, target_.tilt);
  cur_.r = (uint8_t)ease(cur_.r, target_.r);
  cur_.g = (uint8_t)ease(cur_.g, target_.g);
  cur_.b = (uint8_t)ease(cur_.b, target_.b);
  if (blink_phase_ == 1) { blink_ += 0.35f; if (blink_ >= 1) { blink_ = 1; blink_phase_ = 2; } }
  else if (blink_phase_ == 2) { blink_ -= 0.25f; if (blink_ <= 0) { blink_ = 0; blink_phase_ = 0; } }
  draw(first_);
  first_ = false;
}

void Eye::draw(bool full) {
  const int c = EYE_SIZE / 2, R = EYE_SIZE / 2 - 2;
  const int irisR = R * 0.55f;
  const int maxOff = R - irisR - 4;
  const int ix = c + (int)(cur_.px * maxOff), iy = c + (int)(cur_.py * maxOff);
  const int pupR = (int)(irisR * (0.25f + 0.65f * cur_.pr));
  const uint16_t iris = rgb(cur_.r, cur_.g, cur_.b);
  const float ut = cur_.ut * (1 - blink_), lt = cur_.lt * (1 - blink_);
  const int topLid = c - (int)(R * ut);       // y below which the eye is visible
  const int botLid = c + (int)(R * lt);       // y above which the eye is visible

  bool irisMoved = full || ix != (c + (int)(drawn_.px * maxOff)) || iy != (c + (int)(drawn_.py * maxOff)) ||
                   rgb(drawn_.r, drawn_.g, drawn_.b) != iris;
  bool pupilChanged = irisMoved || pupR != (int)(irisR * (0.25f + 0.65f * drawn_.pr));
  int oldTop = c - (int)(R * drawn_.ut), oldBot = c + (int)(R * drawn_.lt);
  const int slope = (int)(cur_.tilt * R * (1 - blink_));
  bool lidsChanged = full || oldTop != topLid || oldBot != botLid ||
                     slope != (int)(drawn_.tilt * R);

  if (full) {
    gfx_->fillScreen(BG);
    gfx_->fillCircle(c, c, R, SCLERA);
  } else if (irisMoved || lidsChanged) {
    // restore sclera under the previous iris and previous lids
    const int oix = c + (int)(drawn_.px * maxOff), oiy = c + (int)(drawn_.py * maxOff);
    gfx_->fillCircle(oix, oiy, irisR + 1, SCLERA);
    if (lidsChanged) gfx_->fillCircle(c, c, R, SCLERA);
  }
  if (irisMoved || lidsChanged) {
    gfx_->fillCircle(ix, iy, irisR, iris);
    gfx_->fillCircle(ix, iy, pupR, PUPIL);
    gfx_->fillCircle(ix - irisR / 3, iy - irisR / 3, irisR / 6, SCLERA);  // highlight
  } else if (pupilChanged) {
    gfx_->fillCircle(ix, iy, irisR, iris);
    gfx_->fillCircle(ix, iy, pupR, PUPIL);
    gfx_->fillCircle(ix - irisR / 3, iy - irisR / 3, irisR / 6, SCLERA);
  }
  // lids: fill the parts of the eye disc above topLid and below botLid
  if (lidsChanged || irisMoved || pupilChanged) {
    const int leftTop = constrain(topLid - slope, 0, EYE_SIZE);
    const int rightTop = constrain(topLid + slope, 0, EYE_SIZE);
    gfx_->fillTriangle(0, 0, EYE_SIZE - 1, 0, 0, leftTop, LID);
    gfx_->fillTriangle(EYE_SIZE - 1, 0, EYE_SIZE - 1, rightTop, 0, leftTop, LID);
    if (botLid < c + R) gfx_->fillRect(0, botLid, EYE_SIZE, EYE_SIZE - botLid, LID);
    // round the corners back to black outside the eye disc
    gfx_->drawCircle(c, c, R + 1, BG);
  }
  drawn_ = cur_;
  drawn_.ut = ut;
  drawn_.lt = lt;
  drawn_.tilt = cur_.tilt * (1 - blink_);
}

// ------------------------------------------------------------------------------------------
void Eyes::begin() {
  Arduino_DataBus *busL = new Arduino_ESP32SPI(PIN_TFT_DC, PIN_TFT_CS_L, PIN_TFT_SCK, PIN_TFT_MOSI, GFX_NOT_DEFINED, VSPI);
  Arduino_DataBus *busR = new Arduino_ESP32SPI(PIN_TFT_DC, PIN_TFT_CS_R, PIN_TFT_SCK, PIN_TFT_MOSI, GFX_NOT_DEFINED, VSPI);
  // shared reset: only the first display object drives RST; the second is reset with it
  Arduino_GFX *gL = new Arduino_GC9A01(busL, PIN_TFT_RST, 0, true);
  Arduino_GFX *gR = new Arduino_GC9A01(busR, GFX_NOT_DEFINED, 0, true);
  left_ = new Eye(gL);
  right_ = new Eye(gR);
  left_->begin();
  right_->begin();
#if PIN_TFT_BL >= 0
  pinMode(PIN_TFT_BL, OUTPUT);
  digitalWrite(PIN_TFT_BL, HIGH);
#endif
}

void Eyes::testPattern() {
  left_->testPattern();
  right_->testPattern();
}

void Eyes::tick() {
  uint32_t now = millis();
  if (now - last_frame_ < 1000 / EYE_FPS) return;
  last_frame_ = now;
  left_->tick();
  right_->tick();
}

void Eyes::idle(uint32_t now) {
  if (now > next_wander_) {
    next_wander_ = now + 1500 + random(2500);
    wx_ = (random(200) - 100) / 250.0f;
    wy_ = (random(200) - 100) / 400.0f;
  }
  if (now > next_blink_) {
    next_blink_ = now + 2500 + random(4000);
    blink();
  }
  EyeParams p;
  p.px = wx_; p.py = wy_; p.pr = 0.45f; p.ut = 0.85f; p.lt = 0.85f;
  setTargets(p, p);
}

void Eyes::startle() {
  EyeParams p = left_->current();
  p.pr = 0.2f; p.ut = 1.0f; p.lt = 1.0f;
  setTargets(p, p);
}

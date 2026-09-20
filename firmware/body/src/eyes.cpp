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

void Eye::setEnabled(bool on) {
  if (on == enabled_) return;
  enabled_ = on;
  if (!on) gfx_->fillScreen(BG);
  else first_ = true;                      // coming back: repaint everything
}

void Eye::tick() {
  if (!enabled_) return;
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

// Row-span renderer: a row is painted once, left to right, in its final state (lid, sclera, iris, pupil,
// highlight), and only the rows that changed since the last frame are painted. Nothing is ever cleared to
// white first, so transitions never flash.
struct EyeGeom {
  int c, R, irisR, ix, iy, pupR, topLid, botLid, slope;
  uint16_t iris;
};

static EyeGeom geom(const EyeParams &p, float blink) {
  EyeGeom g;
  g.c = EYE_SIZE / 2; g.R = EYE_SIZE / 2 - 2;
  g.irisR = g.R * 0.55f;
  const int maxOff = g.R - g.irisR - 4;
  g.ix = g.c + (int)(p.px * maxOff); g.iy = g.c + (int)(p.py * maxOff);
  g.pupR = (int)(g.irisR * (0.25f + 0.65f * p.pr));
  const float ut = p.ut * (1 - blink), lt = p.lt * (1 - blink);
  g.topLid = g.c - (int)(g.R * ut);
  g.botLid = g.c + (int)(g.R * lt);
  g.slope = (int)(p.tilt * g.R * (1 - blink));
  g.iris = rgb(p.r, p.g, p.b);
  return g;
}

static inline int chord(int r, int dy) { int q = r * r - dy * dy; return q > 0 ? (int)sqrtf((float)q) : -1; }

void Eye::paintRows(const EyeGeom &g, int y0, int y1, bool withBg) {
  y0 = constrain(y0, 0, EYE_SIZE); y1 = constrain(y1, 0, EYE_SIZE);
  if (y1 <= y0) return;
  gfx_->startWrite();
  for (int y = y0; y < y1; y++) {
    const int half = chord(g.R, y - g.c);
    if (half < 0) { if (withBg) gfx_->writeFastHLine(0, y, EYE_SIZE, BG); continue; }
    int x0 = g.c - half, x1 = g.c + half;                    // the eye disc on this row, inclusive
    if (withBg) { if (x0 > 0) gfx_->writeFastHLine(0, y, x0, BG); if (x1 < EYE_SIZE - 1) gfx_->writeFastHLine(x1 + 1, y, EYE_SIZE - 1 - x1, BG); }
    if (y >= g.botLid) { gfx_->writeFastHLine(x0, y, x1 - x0 + 1, LID); continue; }
    // the upper lid's edge runs from (0, topLid - slope) to (EYE_SIZE - 1, topLid + slope): covered where y < edge(x)
    int vx0 = x0, vx1 = x1;                                  // the visible (not lid) part of the disc row
    if (g.slope == 0) {
      if (y < g.topLid) { gfx_->writeFastHLine(x0, y, x1 - x0 + 1, LID); continue; }
    } else {
      // edge(x) = topLid + slope * (x - c) / c  ->  covered iff slope*(x - c) > (y - topLid) * c
      const long t = (long)(y - g.topLid) * g.c;
      if (g.slope > 0) {                                     // covered for x > c + t/slope (the right side)
        long xs = g.c + t / g.slope;
        if (xs < x0) { gfx_->writeFastHLine(x0, y, x1 - x0 + 1, LID); continue; }
        if (xs < x1) { gfx_->writeFastHLine((int)xs + 1, y, x1 - (int)xs, LID); vx1 = (int)xs; }
      } else {                                               // covered for x < c + t/slope (the left side)
        long xs = g.c + t / g.slope;
        if (xs > x1) { gfx_->writeFastHLine(x0, y, x1 - x0 + 1, LID); continue; }
        if (xs > x0) { gfx_->writeFastHLine(x0, y, (int)xs - x0, LID); vx0 = (int)xs; }
      }
    }
    if (vx1 < vx0) continue;
    // sclera, then the iris / pupil / highlight spans clipped to the visible part
    gfx_->writeFastHLine(vx0, y, vx1 - vx0 + 1, SCLERA);
    const int ih = chord(g.irisR, y - g.iy);
    if (ih >= 0) {
      int a = max(vx0, g.ix - ih), b = min(vx1, g.ix + ih);
      if (b >= a) gfx_->writeFastHLine(a, y, b - a + 1, g.iris);
      const int ph = chord(g.pupR, y - g.iy);
      if (ph >= 0) { a = max(vx0, g.ix - ph); b = min(vx1, g.ix + ph); if (b >= a) gfx_->writeFastHLine(a, y, b - a + 1, PUPIL); }
      const int hx = g.ix - g.irisR / 3, hy = g.iy - g.irisR / 3, hh = chord(g.irisR / 6, y - hy);
      if (hh >= 0) { a = max(vx0, hx - hh); b = min(vx1, hx + hh); if (b >= a) gfx_->writeFastHLine(a, y, b - a + 1, SCLERA); }
    }
  }
  gfx_->endWrite();
}

void Eye::draw(bool full) {
  const EyeGeom g = geom(cur_, blink_);
  const EyeGeom o = geom(drawn_, 0.0f);                       // drawn_ already has the blink folded in
  if (full) { paintRows(g, 0, EYE_SIZE, true); }
  else {
    // dirty bands: where the lids moved, and where the iris was / is (if it moved, resized or changed colour)
    int bands[3][2]; int nb = 0;
    const int oT = o.topLid, nT = g.topLid, oS = abs(o.slope), nS = abs(g.slope);
    if (oT != nT || o.slope != g.slope) { bands[nb][0] = min(oT - oS, nT - nS) - 1; bands[nb][1] = max(oT + oS, nT + nS) + 1; nb++; }
    if (o.botLid != g.botLid) { bands[nb][0] = min(o.botLid, g.botLid) - 1; bands[nb][1] = max(o.botLid, g.botLid) + 1; nb++; }
    if (o.ix != g.ix || o.iy != g.iy || o.pupR != g.pupR || o.iris != g.iris) {
      bands[nb][0] = min(o.iy, g.iy) - g.irisR - 1; bands[nb][1] = max(o.iy, g.iy) + g.irisR + 2; nb++;
    }
    // merge overlapping bands, then paint
    for (int i = 0; i < nb; i++) for (int j = i + 1; j < nb; j++)
      if (bands[j][0] <= bands[i][1] && bands[i][0] <= bands[j][1]) { bands[i][0] = min(bands[i][0], bands[j][0]); bands[i][1] = max(bands[i][1], bands[j][1]); bands[j][0] = bands[j][1] = -1; }
    for (int i = 0; i < nb; i++) if (bands[i][1] > bands[i][0]) paintRows(g, bands[i][0], bands[i][1], false);
  }
  drawn_ = cur_;
  drawn_.ut = cur_.ut * (1 - blink_);
  drawn_.lt = cur_.lt * (1 - blink_);
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

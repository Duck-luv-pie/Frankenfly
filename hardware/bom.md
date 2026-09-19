# Bill of materials

| Qty | Part | Notes |
|---|---|---|
| 1 | ESP32 DevKit V1 (ESP-WROOM-32, 30-pin) | the "body" |
| 1 | ESP32-CAM (AI-Thinker, OV2640) with USB carrier board | the "cam"; carrier gives 5 V + serial |
| 2 | GC9A01 1.28" round IPS TFT, 240×240, SPI, 7-pin | the eyes; 3.3 V modules |
| 1 | DFPlayer Mini MP3 module | UART, plays from micro SD |
| 1 | micro SD card, FAT32, ≤ 32 GB | holds `/mp3/000N.mp3` |
| 1 | 8 Ω speaker, 0.5–3 W | wired to SPK_1 / SPK_2 |
| 1 | HC-SR501 PIR motion sensor | 5 V, 3.3 V logic output |
| 1 | 1 kΩ resistor | in series with DFPlayer RX |
| 2 | 5 V USB power supplies, ≥ 1 A each | one per board |
| — | jumper wires, breadboard or perfboard | |
| — | later: servos + driver for arms | `arms` field in the protocol is reserved |

Nothing else is needed: the brain runs on the Mac.

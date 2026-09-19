# Sounds for the DFPlayer Mini

The DFPlayer plays MP3s from a FAT32 micro SD card. Track numbers in `brain/configs/default.yaml`
(`decode.sounds`) refer to files in the `/mp3/` folder with 4-digit names:

```
SD card
└── mp3/
    ├── 0001.mp3   escape  - wing buzz
    ├── 0002.mp3   backward - descending chirp
    ├── 0003.mp3   groom   - brushing noise
    ├── 0004.mp3   threat  - harsh tremolo buzz
    └── 0005.mp3   social  - courtship pulse song (35 ms inter-pulse interval)
```

Generate them with:

```sh
uv run --project brain python tools/make_sounds.py
```

This writes WAVs here and, if `ffmpeg` is installed (`brew install ffmpeg`), the MP3s too.
Replace any of them with your own recordings; only the file names matter. Keep the card small
(≤ 32 GB) and copy the files in order, since the DFPlayer indexes by copy order on some firmware
versions and by name in the `mp3/` folder on others.

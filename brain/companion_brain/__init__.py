"""Companion brain package."""
import os

# OpenCV's ffmpeg reader probes an unknown stream for several seconds before delivering a frame;
# the ESP32-CAM is plain MJPEG, so tell it not to (must be set before the first capture is opened).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "analyzeduration;200000|probesize;65536|fflags;nobuffer|flags;low_delay|max_delay;0")

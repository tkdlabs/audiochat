#!/usr/bin/env python3
"""Persistent, length-prefixed Piper TTS server.

Reads a length-prefixed text request from stdin and writes a length-prefixed
raw-PCM reply to stdout, keeping the voice model loaded for the lifetime of the
process. This lets the Rust client synthesize many chunks without paying the
model-load cost on every call.

Protocol (all integers little-endian u32):
  startup:  <sample_rate>          (once, before any reply)
  request:  <text_len><text bytes> (utf-8)
  reply:    <pcm_len><pcm bytes>   (16-bit mono at <sample_rate>)
"""

import struct
import sys

from piper import PiperVoice


def main() -> None:
    model = sys.argv[1]
    voice = PiperVoice.load(model)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    stdout.write(struct.pack("<I", voice.config.sample_rate))
    stdout.flush()

    while True:
        header = stdin.read(4)
        if len(header) < 4:
            break
        (text_len,) = struct.unpack("<I", header)
        if text_len == 0:
            stdout.write(struct.pack("<I", 0))
            stdout.flush()
            continue
        data = stdin.read(text_len)
        if len(data) < text_len:
            break
        text = data.decode("utf-8").strip()
        if not text:
            stdout.write(struct.pack("<I", 0))
            stdout.flush()
            continue

        pcm = bytearray()
        for chunk in voice.synthesize(text):
            pcm.extend(chunk.audio_int16_bytes)

        stdout.write(struct.pack("<I", len(pcm)))
        stdout.write(pcm)
        stdout.flush()


if __name__ == "__main__":
    main()

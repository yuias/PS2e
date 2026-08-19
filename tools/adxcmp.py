"""Compare the emulator's SPU2 output against a reference decode of the
ADX track the game streams through AutoDMA.

    python tools/adxcmp.py shots/x.wav [--track N] [--iso assets/SLPS-25918.iso]

Decodes the first seconds of every MUSIC.AFS track (or just --track),
finds the one the WAV contains by cross-correlation, aligns it and prints
the gain and residual per window. A clean SPU2/IOP path shows a constant
gain (0x3D80/0x8000 * BVOL/MVOL) and a residual of ~1 LSB; the ADMA ring
bug showed up here as 0.5-0.9 correlation with lags jumping by 256.

CRI type-8 encryption: the scale words are XORed with an LCG key stream
(one step per 18-byte frame across channels). Amagami's key is the
vgmstream entry "mituba".
"""

import argparse
import math
import struct
import sys
import wave

import numpy as np

KEY = (0x5A17, 0x509F, 0x5BFD)
MUSIC_AFS = (728729, 484880384)  # LBA, size of /SOUND/MUSIC.AFS


class Afs:
    def __init__(self, iso, lba, size):
        self.f = open(iso, "rb")
        self.base = lba * 2048
        self.f.seek(self.base)
        hdr = self.f.read(8)
        assert hdr[:4] == b"AFS\0", hdr
        n = struct.unpack("<I", hdr[4:8])[0]
        tab = self.f.read(8 * n)
        self.entries = [struct.unpack("<II", tab[i * 8 : i * 8 + 8]) for i in range(n)]

    def read(self, i, limit=None):
        off, size = self.entries[i]
        if limit:
            size = min(size, limit)
        self.f.seek(self.base + off)
        return self.f.read(size)


def adx_info(d):
    if d[0] != 0x80 or d[1] != 0x00:
        raise ValueError("not ADX")
    off = struct.unpack(">H", d[2:4])[0]
    enc, blk, bits, ch = d[4], d[5], d[6], d[7]
    rate, total = struct.unpack(">II", d[8:16])
    return dict(off=off + 4, enc=enc, blk=blk, bits=bits, ch=ch, rate=rate, total=total,
                cutoff=struct.unpack(">H", d[16:18])[0], flags=d[0x13])


def adx_decode(d, max_samples, key=KEY):
    info = adx_info(d)
    assert info["enc"] == 3 and info["bits"] == 4 and info["blk"] == 18, info
    ch, rate, cutoff = info["ch"], info["rate"], info["cutoff"]
    a = math.sqrt(2) - math.cos(2 * math.pi * cutoff / rate)
    b = math.sqrt(2) - 1
    c = (a - math.sqrt((a + b) * (a - b))) / b
    c1, c2 = int(c * 2 * 4096), int(-(c * c) * 4096)
    total = min(info["total"], max_samples)
    nblk = (total + 31) // 32
    out = np.zeros((nblk * 32, ch), dtype=np.int32)
    data = d[info["off"] :]
    xor = key[0] if info["flags"] & 8 else None
    for bi in range(nblk):
        for cc in range(ch):
            base = (bi * ch + cc) * 18
            blk = data[base : base + 18]
            if len(blk) < 18:
                break
            scale = struct.unpack(">H", blk[:2])[0]
            if xor is not None:
                scale = ((scale ^ xor) & 0x1FFF) + 1
                xor = (xor * key[1] + key[2]) & 0x7FFF
            h1 = out[bi * 32 - 1, cc] if bi else 0
            h2 = out[bi * 32 - 2, cc] if bi else 0
            for i in range(32):
                byte = blk[2 + i // 2]
                nib = (byte >> 4) if i % 2 == 0 else (byte & 0xF)
                if nib >= 8:
                    nib -= 16
                s = nib * scale + ((c1 * h1 + c2 * h2) >> 12)
                s = max(-32768, min(32767, s))
                out[bi * 32 + i, cc] = s
                h2, h1 = h1, s
    return info, out[:total].astype(np.int16)


def load_wav(path):
    w = wave.open(path)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return d.reshape(-1, w.getnchannels()).astype(np.float64)


def best_match(seg, ref):
    """Lag of `ref` inside `seg` with the highest normalised correlation."""
    n = 1 << int(np.ceil(np.log2(len(seg) + len(ref))))
    xc = np.fft.irfft(np.fft.rfft(seg, n) * np.fft.rfft(ref[::-1], n), n)
    valid = xc[len(ref) - 1 : len(seg)]
    k = int(np.argmax(valid))
    sub = seg[k : k + len(ref)]
    return k, valid[k] / (np.linalg.norm(ref) * np.linalg.norm(sub) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--iso", default="assets/SLPS-25918.iso")
    ap.add_argument("--track", type=int)
    ap.add_argument("--seconds", type=float, default=6.0, help="reference length to decode")
    args = ap.parse_args()

    out = load_wav(args.wav)[:, 0]
    out -= out.mean()
    afs = Afs(args.iso, *MUSIC_AFS)
    tracks = [args.track] if args.track is not None else range(len(afs.entries))
    best = None
    for i in tracks:
        raw = afs.read(i, int(args.seconds * 48000 * 36 / 32) + 65536)
        try:
            _, pcm = adx_decode(raw, int(args.seconds * 48000))
        except (ValueError, AssertionError):
            continue
        ref = pcm[:, 0].astype(np.float64)
        nz = int(np.argmax(np.abs(ref) > 0))
        ref = ref[nz:]
        ref -= ref.mean()
        k, corr = best_match(out, ref)
        print(f"track {i:3d}: corr {corr:.3f} at {k / 48000:.2f}s", flush=True)
        if best is None or corr > best[0]:
            best = (corr, i, k, ref)
    corr, i, k, ref = best
    print(f"\nbest: track {i} corr {corr:.3f} starting at wav sample {k} ({k / 48000:.3f}s)")
    sub = out[k : k + len(ref)]
    W = 24000
    for w in range(0, len(ref) - W, W):
        r, s = ref[w : w + W], sub[w : w + W]
        g = (r * s).sum() / ((r * r).sum() + 1e-9)
        res = s - g * r
        c = (r * s).sum() / (np.linalg.norm(r) * np.linalg.norm(s) + 1e-9)
        print(f"  {(k + w) / 48000:7.2f}s gain {g:.4f} corr {c:.4f} resid rms {res.std():7.2f}")


if __name__ == "__main__":
    sys.exit(main())

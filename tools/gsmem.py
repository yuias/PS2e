"""Decode GS VRAM dumps (`--dump <dir>/gs_vram.bin`) laid out the way the
hardware stores them (see crates/ps2-core/src/gs/layout.rs).

    from gsmem import Vram
    v = Vram('dumps/run/gs_vram.bin')
    v.image32(bp=0, bw=10, w=640, h=448).save('fb.png')       # PSMCT32 buffer
    v.image_indexed(bp=4480, bw=8, w=512, h=512, bits=8, cbp=10108).save('tex.png')
    print([hex(c) for c in v.clut32(10108)[:16]])
"""

import struct

from PIL import Image

BLOCK32 = [
    [0, 1, 4, 5, 16, 17, 20, 21],
    [2, 3, 6, 7, 18, 19, 22, 23],
    [8, 9, 12, 13, 24, 25, 28, 29],
    [10, 11, 14, 15, 26, 27, 30, 31],
]
BLOCK16 = [
    [0, 2, 8, 10], [1, 3, 9, 11], [4, 6, 12, 14], [5, 7, 13, 15],
    [16, 18, 24, 26], [17, 19, 25, 27], [20, 22, 28, 30], [21, 23, 29, 31],
]
BLOCK16S = [
    [0, 2, 16, 18], [1, 3, 17, 19], [8, 10, 24, 26], [9, 11, 25, 27],
    [4, 6, 20, 22], [5, 7, 21, 23], [12, 14, 28, 30], [13, 15, 29, 31],
]
VRAM = 4 * 1024 * 1024


def addr32(bp, bw, x, y, z=False):
    bw = max(bw, 1)
    block = bp + ((y >> 5) * bw + (x >> 6)) * 32 + (BLOCK32[(y >> 3) & 3][(x >> 3) & 7] ^ (24 if z else 0))
    col = ((y & 7) >> 1) * 16 + ((x & 7) >> 1) * 4 + (y & 1) * 2 + (x & 1)
    return (block * 256 + col * 4) & (VRAM - 1)


def addr16(bp, bw, x, y, s=False, z=False):
    bw = max(bw, 1)
    table = BLOCK16S if s else BLOCK16
    block = bp + ((y >> 6) * bw + (x >> 6)) * 32 + (table[(y >> 3) & 7][(x >> 4) & 3] ^ (24 if z else 0))
    col = ((y & 7) >> 1) * 32 + (y & 1) * 4 + ((x & 7) >> 1) * 8 + (x & 1) * 2 + ((x & 15) >> 3)
    return (block * 256 + col * 2) & (VRAM - 1)


def addr8(bp, bw, x, y):
    bw = max(bw, 1)
    block = bp + ((y >> 6) * (bw >> 1) + (x >> 7)) * 32 + BLOCK32[(y >> 4) & 3][(x >> 4) & 7]
    c, ry = (y & 15) >> 2, y & 3
    swap = ((ry >> 1) ^ (c & 1)) & 1
    xs = (x & 15) ^ (swap << 2)
    col = c * 64 + (ry & 1) * 8 + (ry >> 1) + ((xs >> 1) & 3) * 16 + (xs & 1) * 4 + (xs >> 3) * 2
    return (block * 256 + col) & (VRAM - 1)


def addr4(bp, bw, x, y):
    """Nibble address."""
    bw = max(bw, 1)
    block = bp + ((y >> 7) * (bw >> 1) + (x >> 7)) * 32 + BLOCK16[(y >> 4) & 7][(x >> 5) & 3]
    c, ry = (y & 15) >> 2, y & 3
    swap = ((ry >> 1) ^ (c & 1)) & 1
    xs = (x & 31) ^ (swap << 2)
    col = c * 128 + (ry & 1) * 16 + (ry >> 1) + ((xs >> 1) & 3) * 32 + (xs & 1) * 8 + ((xs >> 3) & 3) * 2
    return (block * 512 + col) & (VRAM * 2 - 1)


class Vram:
    def __init__(self, path):
        self.v = open(path, 'rb').read()

    def rd32(self, bp, bw, x, y, z=False):
        return struct.unpack_from('<I', self.v, addr32(bp, bw, x, y, z))[0]

    def rd16(self, bp, bw, x, y, s=False, z=False):
        return struct.unpack_from('<H', self.v, addr16(bp, bw, x, y, s, z))[0]

    def rd8(self, bp, bw, x, y):
        return self.v[addr8(bp, bw, x, y)]

    def rd4(self, bp, bw, x, y):
        a = addr4(bp, bw, x, y)
        b = self.v[a >> 1]
        return b & 0xF if a & 1 == 0 else b >> 4

    def clut32(self, cbp):
        """256 RGBA8 entries of a CSM1 32-bit CLUT (8x2-entry tiles)."""
        out = []
        for e in range(256):
            t = (e & 0xE7) | ((e & 8) << 1) | ((e & 0x10) >> 1)
            out.append(self.rd32(cbp, 1, t & 0xF, t >> 4))
        return out

    def image32(self, bp, bw, w, h, z=False, alpha=False):
        im = Image.new('RGBA' if alpha else 'RGB', (w, h))
        px = im.load()
        for y in range(h):
            for x in range(w):
                c = self.rd32(bp, bw, x, y, z)
                px[x, y] = (c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF) + ((min(255, (c >> 24) * 2),) if alpha else ())
        return im

    def image16(self, bp, bw, w, h, s=False):
        im = Image.new('RGB', (w, h))
        px = im.load()
        for y in range(h):
            for x in range(w):
                c = self.rd16(bp, bw, x, y, s)
                px[x, y] = ((c & 0x1F) << 3, ((c >> 5) & 0x1F) << 3, ((c >> 10) & 0x1F) << 3)
        return im

    def image_indexed(self, bp, bw, w, h, bits=8, cbp=None, alpha=False):
        """PSMT8/PSMT4 texture, through `cbp`'s CLUT or as grey indices."""
        clut = self.clut32(cbp) if cbp is not None else None
        im = Image.new('RGBA' if alpha else 'RGB', (w, h))
        px = im.load()
        for y in range(h):
            for x in range(w):
                i = self.rd8(bp, bw, x, y) if bits == 8 else self.rd4(bp, bw, x, y)
                if clut is None:
                    g = i if bits == 8 else i * 17
                    px[x, y] = (g, g, g) + ((255,) if alpha else ())
                else:
                    c = clut[i]
                    px[x, y] = (c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF) + ((min(255, (c >> 24) * 2),) if alpha else ())
        return im

    def alpha_plane(self, bp, bw, w, h):
        """PSMT8H view: the upper byte of each 32-bit pixel, as grey."""
        im = Image.new('L', (w, h))
        px = im.load()
        for y in range(h):
            for x in range(w):
                px[x, y] = self.rd32(bp, bw, x, y) >> 24
        return im

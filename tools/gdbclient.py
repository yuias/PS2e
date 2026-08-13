"""Minimal scriptable gdb-remote client for the ps2-debug stub."""
import socket
import struct
import time


class Gdb:
    def __init__(self, port, timeout=600):
        for _ in range(100):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=2)
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("cannot connect")
        self.s.settimeout(timeout)

    def _send(self, payload: str):
        p = payload.encode()
        self.s.sendall(b"$" + p + b"#" + b"%02x" % (sum(p) & 0xFF))

    def _recv(self) -> str:
        while True:
            b = self.s.recv(1)
            if not b:
                raise RuntimeError("closed")
            if b == b"$":
                break
        buf = b""
        while True:
            b = self.s.recv(1)
            if b == b"#":
                break
            buf += b
        self.s.recv(2)
        return buf.decode()

    def cmd(self, payload: str) -> str:
        self._send(payload)
        return self._recv()

    # -- registers (EE wire layout: gpr 64-bit, pc index 0x25 32-bit) --

    def reg(self, i: int) -> int:
        r = self.cmd("p%x" % i)
        if r.startswith("E"):
            raise RuntimeError(r)
        return int.from_bytes(bytes.fromhex(r), "little")

    def set_reg(self, i: int, v: int, size: int):
        r = self.cmd("P%x=%s" % (i, v.to_bytes(size, "little").hex()))
        assert r == "OK", r

    def pc(self) -> int:
        return self.reg(0x25)

    def regs(self) -> dict:
        names = ("zero at v0 v1 a0 a1 a2 a3 t0 t1 t2 t3 t4 t5 t6 t7 "
                 "s0 s1 s2 s3 s4 s5 s6 s7 t8 t9 k0 k1 gp sp s8 ra").split()
        out = {n: self.reg(i) for i, n in enumerate(names)}
        out["pc"] = self.pc()
        return out

    # -- memory --

    def read(self, addr: int, length: int) -> bytes:
        out = b""
        while length:
            n = min(length, 1024)
            r = self.cmd("m%x,%x" % (addr, n))
            if r.startswith("E"):
                raise RuntimeError("read %#x: %s" % (addr, r))
            chunk = bytes.fromhex(r)
            out += chunk
            if len(chunk) < n:
                break
            addr += n
            length -= n
        return out

    def read32(self, addr: int) -> int:
        return struct.unpack("<I", self.read(addr, 4))[0]

    def write(self, addr: int, data: bytes):
        r = self.cmd("M%x,%x:%s" % (addr, len(data), data.hex()))
        assert r == "OK", r

    # -- execution --

    def bp(self, addr: int, set=True):
        r = self.cmd("%s0,%x,4" % ("Z" if set else "z", addr))
        assert r == "OK", r

    def watch(self, addr: int, length: int, set=True):
        r = self.cmd("%s2,%x,%x" % ("Z" if set else "z", addr, length))
        assert r == "OK", r

    def cont(self) -> str:
        """Continue and block until the stop reply."""
        self._send("c")
        return self._recv()

    def step(self) -> str:
        return self.cmd("s")

    def interrupt(self) -> str:
        self.s.sendall(b"\x03")
        return self._recv()

    def detach(self):
        self.cmd("D")
        self.s.close()


def disasm(data: bytes, addr: int, mode64=True):
    import capstone
    md = capstone.Cs(
        capstone.CS_ARCH_MIPS,
        (capstone.CS_MODE_MIPS64 if mode64 else capstone.CS_MODE_MIPS32)
        | capstone.CS_MODE_LITTLE_ENDIAN,
    )
    lines = []
    for i in range(0, len(data), 4):
        word = data[i : i + 4]
        ins = list(md.disasm(word, addr + i))
        if ins:
            lines.append("%08x: %08x  %s %s" % (addr + i, struct.unpack("<I", word)[0], ins[0].mnemonic, ins[0].op_str))
        else:
            lines.append("%08x: %08x  ??" % (addr + i, struct.unpack("<I", word)[0]))
    return "\n".join(lines)

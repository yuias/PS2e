//! EE (R5900) and IOP (R3000A) register presentation for gdb-remote clients.
//!
//! Both targets follow GDB's classic raw MIPS numbering so plain GDB works
//! unmodified: r0..r31, sr, lo, hi, badvaddr, cause, pc (38 registers). The
//! IOP presents everything as 32-bit, exactly like a PS1. The EE presents
//! GPRs and LO/HI as 64-bit — the low half of the 128-bit registers; the MMI
//! upper halves, LO1/HI1 and SA are not exposed — with 32-bit control
//! registers and pc. Per-register sizes are described in `target.xml` and
//! `qRegisterInfo` so clients size the `g` packet correctly.

use ps2_core::Ps2System;
use ps2_core::ee::cop0 as ee_cop0;

/// COP0 BadVAddr index (same on both cores).
const BADVADDR: usize = 8;
/// IOP COP0 indices (not exported by ps2-core).
const IOP_STATUS: usize = 12;
const IOP_CAUSE: usize = 13;

/// Which core a debug connection talks to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    Ee,
    Iop,
}

impl Target {
    /// LLDB keys its architecture (and disassembler) off this triple.
    pub fn triple(self) -> &'static str {
        match self {
            // mips64 so doubleword ops (ld/sd/daddu...) disassemble.
            Target::Ee => "mips64el-unknown-unknown",
            Target::Iop => "mipsel-unknown-unknown",
        }
    }

    pub fn hostname(self) -> &'static str {
        match self {
            Target::Ee => "ps2e-ee",
            Target::Iop => "ps2e-iop",
        }
    }

    /// `<architecture>` element for `target.xml` (plain-GDB clients).
    pub fn arch(self) -> &'static str {
        match self {
            Target::Ee => "mips64",
            Target::Iop => "mips",
        }
    }
}

/// Number of registers in the `g` packet.
pub const NUM_REGS: usize = 38;

/// ABI names for r0..r31, used as alt-names and in `qRegisterInfo`.
pub const ABI_NAMES: [&str; 32] = [
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", // 0-7
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", // 8-15
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", // 16-23
    "t8", "t9", "k0", "k1", "gp", "sp", "s8", "ra", // 24-31
];

/// Wire size of register `i` in bytes.
pub fn size(t: Target, i: usize) -> usize {
    match t {
        Target::Iop => 4,
        // GPRs and LO/HI are 64-bit; sr/badvaddr/cause/pc stay 32-bit.
        Target::Ee => match i {
            0..=31 | 33 | 34 => 8,
            _ => 4,
        },
    }
}

/// Byte offset of register `i` in the `g` packet.
pub fn offset(t: Target, i: usize) -> usize {
    (0..i).map(|j| size(t, j)).sum()
}

/// Total `g` packet payload size in bytes.
pub fn g_len(t: Target) -> usize {
    offset(t, NUM_REGS)
}

pub fn read(sys: &Ps2System, t: Target, i: usize) -> u64 {
    match t {
        Target::Ee => {
            let c = &sys.ee;
            match i {
                0..=31 => c.gpr[i][0],
                32 => c.cop0.regs[ee_cop0::STATUS] as u64,
                33 => c.lo[0],
                34 => c.hi[0],
                35 => c.cop0.regs[BADVADDR] as u64,
                36 => c.cop0.regs[ee_cop0::CAUSE] as u64,
                37 => c.pc as u64,
                _ => 0,
            }
        }
        Target::Iop => {
            let c = &sys.iop;
            (match i {
                0..=31 => c.gpr[i],
                32 => c.cop0[IOP_STATUS],
                33 => c.lo,
                34 => c.hi,
                35 => c.cop0[BADVADDR],
                36 => c.cop0[IOP_CAUSE],
                37 => c.pc,
                _ => 0,
            }) as u64
        }
    }
}

pub fn write(sys: &mut Ps2System, t: Target, i: usize, v: u64) {
    match t {
        Target::Ee => {
            let c = &mut sys.ee;
            match i {
                // r0 is hardwired to zero; only the low 64 bits are exposed,
                // the MMI upper half is left untouched.
                1..=31 => c.gpr[i][0] = v,
                32 => c.cop0.regs[ee_cop0::STATUS] = v as u32,
                33 => c.lo[0] = v,
                34 => c.hi[0] = v,
                35 => c.cop0.regs[BADVADDR] = v as u32,
                36 => c.cop0.regs[ee_cop0::CAUSE] = v as u32,
                // Redirecting pc must also cancel any in-flight branch target.
                37 => {
                    c.pc = v as u32;
                    c.next_pc = (v as u32).wrapping_add(4);
                    c.idle = false;
                }
                _ => {}
            }
        }
        Target::Iop => {
            let c = &mut sys.iop;
            let v = v as u32;
            match i {
                1..=31 => c.gpr[i] = v,
                32 => c.cop0[IOP_STATUS] = v,
                33 => c.lo = v,
                34 => c.hi = v,
                35 => c.cop0[BADVADDR] = v,
                36 => c.cop0[IOP_CAUSE] = v,
                37 => {
                    c.pc = v;
                    c.next_pc = v.wrapping_add(4);
                    c.idle = false;
                }
                _ => {}
            }
        }
    }
}

/// The `target.xml` document served via `qXfer:features:read`.
pub fn target_xml(t: Target) -> String {
    let mut xml = format!(
        r#"<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target version="1.0">
  <architecture>{}</architecture>
  <feature name="org.gnu.gdb.mips.cpu">
"#,
        t.arch()
    );
    for (i, abi) in ABI_NAMES.iter().enumerate() {
        let generic = match i {
            29 => r#" generic="sp""#,
            30 => r#" generic="fp""#,
            31 => r#" generic="ra""#,
            _ => "",
        };
        xml.push_str(&format!(
            "    <reg name=\"r{i}\" altname=\"{abi}\" bitsize=\"{}\" regnum=\"{i}\" \
             dwarf_regnum=\"{i}\"{generic}/>\n",
            size(t, i) * 8
        ));
    }
    xml.push_str(&format!(
        r#"    <reg name="lo" bitsize="{}" regnum="33"/>
    <reg name="hi" bitsize="{}" regnum="34"/>
    <reg name="pc" bitsize="32" regnum="37" type="code_ptr" generic="pc"/>
  </feature>
  <feature name="org.gnu.gdb.mips.cp0">
    <reg name="status" bitsize="32" regnum="32"/>
    <reg name="badvaddr" bitsize="32" regnum="35"/>
    <reg name="cause" bitsize="32" regnum="36"/>
  </feature>
</target>
"#,
        size(t, 33) * 8,
        size(t, 34) * 8
    ));
    xml
}

/// Reply to LLDB's `qRegisterInfo<n>`: one register description per query,
/// `E45` past the end. Field reference: lldb docs/lldb-gdb-remote.txt.
pub fn register_info(t: Target, i: usize) -> Option<String> {
    if i >= NUM_REGS {
        return None;
    }
    let (name, set, generic): (&str, &str, Option<&str>) = match i {
        0..=31 => (
            ABI_NAMES[i],
            "General Purpose Registers",
            match i {
                4..=7 => Some(["arg1", "arg2", "arg3", "arg4"][i - 4]),
                29 => Some("sp"),
                30 => Some("fp"),
                31 => Some("ra"),
                _ => None,
            },
        ),
        32 => ("sr", "Control Registers", None),
        33 => ("lo", "General Purpose Registers", None),
        34 => ("hi", "General Purpose Registers", None),
        35 => ("badvaddr", "Control Registers", None),
        36 => ("cause", "Control Registers", None),
        37 => ("pc", "General Purpose Registers", Some("pc")),
        _ => unreachable!(),
    };
    let mut out = format!(
        "name:{name};bitsize:{};offset:{};encoding:uint;format:hex;set:{set};",
        size(t, i) * 8,
        offset(t, i)
    );
    if i <= 31 {
        out.push_str(&format!("alt-name:r{i};dwarf:{i};gcc:{i};"));
    }
    if let Some(g) = generic {
        out.push_str(&format!("generic:{g};"));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys() -> Ps2System {
        Ps2System::new(vec![0; 4 * 1024 * 1024]).unwrap()
    }

    #[test]
    fn r0_stays_zero() {
        let mut s = sys();
        for t in [Target::Ee, Target::Iop] {
            write(&mut s, t, 0, 0xdead_beef);
            assert_eq!(read(&s, t, 0), 0);
        }
    }

    #[test]
    fn pc_write_cancels_pending_branch() {
        let mut s = sys();
        write(&mut s, Target::Ee, 37, 0x8010_0000);
        assert_eq!(s.ee.pc, 0x8010_0000);
        assert_eq!(s.ee.next_pc, 0x8010_0004);
        write(&mut s, Target::Iop, 37, 0x0010_0000);
        assert_eq!(s.iop.pc, 0x0010_0000);
        assert_eq!(s.iop.next_pc, 0x0010_0004);
    }

    #[test]
    fn ee_gpr_is_64_bit() {
        let mut s = sys();
        write(&mut s, Target::Ee, 1, 0x1_0000_0000);
        assert_eq!(read(&s, Target::Ee, 1), 0x1_0000_0000);
        assert_eq!(size(Target::Ee, 1), 8);
        assert_eq!(size(Target::Ee, 37), 4);
    }

    #[test]
    fn g_packet_layout() {
        // EE: 32 x 8 (gpr) + 4 (sr) + 8 + 8 (lo/hi) + 4 + 4 + 4.
        assert_eq!(g_len(Target::Ee), 288);
        assert_eq!(offset(Target::Ee, 37), 284);
        // IOP: 38 x 4, the classic PS1 layout.
        assert_eq!(g_len(Target::Iop), 152);
    }

    #[test]
    fn register_info_covers_all_and_ends() {
        for t in [Target::Ee, Target::Iop] {
            for i in 0..NUM_REGS {
                let info = register_info(t, i).unwrap();
                assert!(info.contains("bitsize:"), "{info}");
            }
            assert!(register_info(t, NUM_REGS).is_none());
        }
    }

    #[test]
    fn target_xml_names_all_gdb_registers() {
        for t in [Target::Ee, Target::Iop] {
            let xml = target_xml(t);
            for name in ["r0", "r31", "lo", "hi", "pc", "status", "badvaddr", "cause"] {
                assert!(xml.contains(&format!("name=\"{name}\"")), "{name} missing");
            }
        }
        assert!(target_xml(Target::Ee).contains("bitsize=\"64\""));
        assert!(!target_xml(Target::Iop).contains("bitsize=\"64\""));
    }
}

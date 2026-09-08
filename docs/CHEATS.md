# Cheats

PS2e applies pnach patches once per frame, at the start of vertical blank.
Writes go through the debugger's poke path rather than straight into RAM,
so the EE and IOP recompilers are told about pages they hold translated
code for.

Codes are read from a `.pnach` file beside the disc image — `Game.iso`
takes its cheats from `Game.pnach`. The format is the one PCSX2
documents at <https://pcsx2.net/docs/advanced/writing-patches/>:
`[Name]` section headers, `patch=place,cpu,address,type,data` lines,
`//` comments, and `gametitle=` / `author=` / `comment=` /
`description=` metadata, which is read past. Anything the parser rejects
is skipped with its line number, and the rest of the file still runs.

## The Cheats page

The side pane's Cheats page lists one checkbox per `[Name]` section of
the disc's pnach, with the lines that section lost shown under it.
Nothing applies until "Apply cheats" at the top is on; that switch is
`cheats` in `config.toml` and defaults to off, so a `.pnach` left beside
an image does not change how a game runs until it is asked for. Headless
runs have `--cheats` for the same thing, and no per-section state: they
run everything the file holds.

"Reload" re-reads the file, for editing it with the emulator running.
A reload re-arms the one-shot (`place` 0 and 3) codes; switching a single
cheat on or off does not, so the rest of the table keeps whatever it has
already done.

Which sections are switched **off** is remembered per disc in
`cheats.toml`, next to `config.toml`:

```toml
["SLPS-25418"]
disabled = ["Infinite Health"]
```

The key is the disc's boot serial, read from SYSTEM.CNF, so the entry
survives renaming or moving the image; an image with no readable
SYSTEM.CNF falls back to the pnach's file name. It records what is off
rather than what is on, so a disc with no entry runs everything its file
holds, and a section added to the pnach later is live without an edit
here. The list is kept out of `config.toml` because saving that file
rewrites it whole, losing the commented template it ships with.

The boot serial only arrives once the worker has read the disc, so for a
frame or two after a disc goes in the list runs under the image's file
name instead, with everything on; it switches over and is re-applied as
soon as the serial is known.

## From the memory scanner

The pane's Memory page has the usual find-a-value-and-narrow-it-down
scanner. Each hit has a "+" beside it that writes a cheat holding that
value at that address into the disc's pnach:

```
[Scan EE 0033A1C0 word]
patch=1,EE,0033A1C0,word,00000063
```

`place` 1 is "every frame", which is what holds a value against the game
writing its own. The section is named after the CPU, the address and the
width, so pressing "+" again on the same hit rewrites that section with
the current value rather than stacking a second one — the list has no
delete, and pressing again is what a user does when the value has moved
on. From then on it is an ordinary entry, switched on and off from the
Cheats page like any other.

Only that one section's lines are rewritten. The parser keeps no
`comment=` lines and no formatting, so writing the whole file back from
the parsed model would quietly destroy a hand-written pnach.

## Code types

`patch=place,cpu,address,type,data`. `place` is 0 or 3 for "once, at
start" and 1 or 2 for "every frame"; `cpu` is `EE` or `IOP`; `type` is
`byte`, `short`, `word` or `double`, the same four prefixed with `be`
for a big-endian value, `bytes` for a run of hex bytes, or `extended`.

`extended` codes are the RAW types of the PCSX2 page, where the top
nibble of the address field is the code type and a code may continue onto
the following `patch=` lines:

| Type | Effect |
| --- | --- |
| `0` / `1` / `2` | 8-, 16- and 32-bit write |
| `3` | Increment / decrement, 8- to 32-bit |
| `4` | `count` 32-bit writes, `stride` words apart, value stepping each time |
| `5` | Copy `len` bytes |
| `6` | Follow a pointer chain, then write at the end of it |
| `7` | OR / AND / XOR with the value in memory |
| `D` | Compare, and skip the following codes when the test fails |

Types the page does not list are reported and skipped. A code that
touches an address outside RAM is dropped after one warning rather than
retried every frame.

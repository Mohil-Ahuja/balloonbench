# Caveat — vendored handwriting font, licence and provenance

## What this is

`Caveat.ttf` is the handwriting face BalloonBench uses for the *semantic clutter* stage of
its degradation pipeline: margin notes, red-pen corrections, and the numerals of a previous
ballooning attempt already drawn on the sheet.

That clutter is not decoration either. A drawing arriving at a real shop has usually been
marked up by a person, and a previous hand-ballooning is one of the single most confusing
things a vision model can encounter — it looks exactly like the output it is being asked to
produce. Rendering it convincingly needs a face that is visibly *not* the technical
lettering used for the drawing itself, which is why a second font is vendored rather than
osifont being reused with jitter.

## Provenance

| | |
|---|---|
| Upstream | https://github.com/google/fonts/tree/main/ofl/caveat |
| Files taken | `Caveat[wght].ttf` (kept here as `Caveat.ttf`), `OFL.txt` |
| Retrieved | 2026-09-03 |
| Modified | **No.** The binary is byte-for-byte upstream; only the filename differs, because a `[wght]` in a path is awkward to quote across shells. |

Caveat is a variable font with a single `wght` axis. Pillow renders its default instance
without any configuration, and BalloonBench uses that default; the axis is available if a
future profile wants a heavier pen.

## Licence

**SIL Open Font License, Version 1.1.** Full text in [`OFL.txt`](OFL.txt).

The OFL is the most permissive of the common font licences for this use. It allows
redistribution of the font with the repository and imposes no condition at all on documents
that embed or display it, so BalloonBench's generated and degraded images carry only the
project's own licence. Unlike the GPL-family font we use for the drawing lettering, no font
exception is needed here — the OFL simply does not reach the output.

The two conditions that do bind us, and how we meet them:

- **The font may not be sold on its own.** BalloonBench does not sell it, and redistributes
  it as part of a larger work.
- **A derivative may not use the reserved font name.** We do not modify the binary, so no
  derivative exists and the question does not arise. If a glyph is ever genuinely needed
  that Caveat lacks, add an overlay font rather than editing this one — and rename it if it
  is a modification of Caveat.

## Coverage

`scripts/check_env.py` asserts the font is present and covers the ASCII range the clutter
stage draws with. As with the drawing font, a missing glyph does not raise — it renders as a
tofu box — so absence is checked rather than assumed.

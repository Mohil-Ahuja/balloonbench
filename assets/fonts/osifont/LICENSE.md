# osifont — vendored font, licence and provenance

## What this is

`osifont-lgpl3fe.ttf` and `osifont-italic.ttf` are the drawing font BalloonBench renders
every sheet with. SPEC.md section 3 requires a vendored font rather than a system one, for
two reasons: output must be byte-reproducible across machines, and the font must actually
contain the GD&T glyphs — most system fonts do not, and a missing glyph renders as a
tofu box that would silently corrupt ground truth in the rendered image.

osifont is a free font built to conform to ISO 3098, the drawing-lettering standard, which
is why it is the right choice rather than a general-purpose sans.

## Provenance

| | |
|---|---|
| Upstream | https://github.com/hikikomori82/osifont |
| Files taken | `osifont-lgpl3fe.ttf`, `osifont-italic.ttf`, `MANUAL`, `README.md` (kept here as `UPSTREAM-README.md`) |
| Retrieved | 2026-09-02 |
| Modified | **No.** The binaries are byte-for-byte upstream. |

## Licence

Upstream publishes osifont under three licences, as separate builds. **We vendor the
LGPL-3.0-with-font-exception build** (`osifont-lgpl3fe.ttf`) deliberately — it is the most
permissive of the three, and the only one that sits comfortably alongside Apache-2.0 code.

- Full LGPL-3.0 text: [`COPYING.LESSER`](COPYING.LESSER)
- LGPL-3.0 incorporates GPL-3.0 by reference; that text is in [`COPYING`](COPYING)

`osifont-italic.ttf` is included for the italic style. Upstream ships only one italic
build; treat it as carrying the same GPL-family terms and the same font exception.

### Why this does not make BalloonBench's output GPL

This is the point that matters for a benchmark whose whole product is rendered documents.

The **GPL font exception** exists precisely for this case. Embedding the font in a document
does not place the document under the GPL. BalloonBench's PDFs embed osifont, so without
the exception every generated drawing would arguably inherit the font's licence — which
would make the dataset undistributable on the terms the project promises. With it, the
rendered drawings, the PNGs and the dataset carry only BalloonBench's own licence.

### What this does and does not constrain

- **BalloonBench's source code stays Apache-2.0.** The font is data we redistribute, not
  code we link against. A repository may distribute files under different licences.
- **The font files themselves remain under LGPL-3.0-with-font-exception.** Anyone
  redistributing this repository redistributes them on those terms, which is why this
  notice and both licence texts are checked in beside them.
- **Do not modify the font binaries.** They are unmodified upstream builds, and keeping
  them that way avoids the LGPL obligations that attach to a derived work. If a glyph is
  ever genuinely missing, add it in a separate overlay font rather than editing these.

## Glyph coverage

`scripts/check_env.py` asserts on every run that the vendored font actually contains all
26 codepoints the renderer needs — the geometric characteristic symbols, the material
condition modifiers, and the dimensioning symbols. A font swap or a truncated download
fails the environment gate rather than producing tofu boxes in a rendered drawing that
nobody notices until the ground truth is already wrong.

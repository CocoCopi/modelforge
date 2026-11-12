# ModelForge design system

Concept: a maker's notebook and workshop bench. Warm paper, ink, and one ember
accent. Calm, precise, editorial. Light mode is the only theme for v2.

## Rules

- No pure white backgrounds, no pure black text, no gradients, no glassmorphism,
  no drop shadows. Use hairline borders and tonal layers.
- At most one ember primary action per screen. Ember is for fills and large
  active marks; `emberText` is the darker token for text and small icons.
- Left-aligned editorial layouts. No centered hero cards, no default Material
  cards, no pill buttons.
- 8dp grid. Generous whitespace. Touch targets are at least 48dp.
- Corners are 4-6dp or square, always with a 1dp hairline where a boundary is
  needed.
- Every text/background pair must meet WCAG AA: 4.5:1 for body, 3:1 for large
  text and icons. Verify with `scripts/check_contrast.py`; fix tokens, never
  ignore a failure.
- No Roboto, Inter, system default fonts, emoji icons, stock Material icon look,
  stock illustrations, or identical card grids everywhere.
- Respect the system reduce-motion setting; replace spring/slide motion with
  short fades when it is enabled.

## Color tokens (defined once in `Theme.kt`)

| Token | Hex | Use |
| --- | --- | --- |
| paper | `#F4EFE6` | app background |
| surface | `#FBF8F2` | cards, sheets, fields |
| ink | `#1B1A17` | primary text and ink fills |
| inkMuted | `#6B655A` | secondary text |
| hairline | `#DDD5C6` | borders, dividers, grid |
| ember | `#D9481C` | ember fills, large active marks |
| emberText | `#B83A12` | accent text and small icons |
| verdigris | `#1F5C54` | success, "ready", secondary accent |
| ochre | `#B8860B` | large warning marks only; fails AA as small text |
| ochreInk | `#7D5906` | warning text/icons (derived to pass AA on ochreTint) |
| ochreTint | `#F3E3B3` | warning surface |
| error | `#A8281E` | errors (on errorTint) |
| errorTint | `#F6D9D4` | error surface |
| inkGhost | `#6F685C` | disabled text (darkened from #8C8578 for AA) |
| hairlineSoft | `#E9E2D5` | inner dividers |

Contrast decisions from the AA audit:

- `ember` on paper is 3.75:1, so it is only allowed for large text (>=18pt or
  14pt bold), icons and large marks, never body text. Use `emberText` for body.
- Paper text on an `ember` fill is 3.75:1 and fails body AA. Primary button
  fills therefore use `emberText` (5.02:1 with paper) while `ember` remains the
  large active mark / chart endpoint colour.
- `ochre` on `ochreTint` is 2.55:1, so warning text uses `ochreInk` (4.98:1).
  `ochre` is kept for non-text highlight marks.
- `inkGhost` was darkened so disabled labels still clear AA on paper.

## Typography

- **Fraunces** (static 400/600 instances, optical size 72): display titles and
  large changing numbers. Editorial, high contrast, generous line height.
- **Instrument Sans** (400/500/600): body and UI.
- **JetBrains Mono** (400/500): data, logs, code, tabular numbers.

Type scale (sp):

| Name | Font | Size / line | Use |
| --- | --- | ---: | --- |
| displayXl | Fraunces 600 | 52 / 56 | parameter count, empty-state phrase |
| displayL | Fraunces 600 | 38 / 44 | screen title |
| displayM | Fraunces 400 | 28 / 36 | section hero |
| title | Fraunces 600 | 21 / 28 | card title |
| bodyL | Instrument 400 | 17 / 26 | reading text |
| body | Instrument 400 | 15 / 23 | default text |
| bodyStrong | Instrument 600 | 15 / 23 | emphasis |
| label | Instrument 600 | 11 / 16, 0.12em tracking, uppercase | small caps labels |
| mono | JetBrains Mono 400 | 13 / 20 | stats and logs |
| monoLarge | JetBrains Mono 500 | 20 / 26 | ledger numbers |

All changing numbers use tabular figures (`FontFeatureSettings("tnum")`).

## Shape, spacing, layout

- Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64.
- Radius: 4dp controls, 6dp containers, 0 for rulers and ledger lines.
- Hairline 1dp borders. Tonal layering uses paper -> surface -> hairlineSoft.
- Buttons: rectangular, ink fill for secondary/navigation and one ember fill for
  the primary action. Labels are small caps in Instrument Sans 600.
- Cards: `NotebookCard` only; no Material `Card` on new screens.
- Widths: readable column max 560dp; full-bleed rulers and charts may exceed it.

## Signature components

- `RulerSlider`: log-scale horizontal ruler, decade ticks (1M, 10M, 100M...),
  draggable index mark, huge Fraunces count, haptic ticks at decades.
- `BitGrainSelector`: stepped precision choices; each step shows a Canvas grain
  pattern (fine for 32-bit, coarse for 2-bit) and the real estimated size.
- `WizardRuler`: thin ruler along the top, one notch per step, current step in
  small caps ("STEP 04 / 14").
- `LedgerChart`: hairline ink strokes, light grid, ember latest point, direct
  labels, event annotations instead of a legend.
- `LedgerStat`: monospaced label/value rows, tabular figures, hairline separators.
- `ThinIcon`: 1.5dp stroke vector icons drawn in code; no Material icons.
- Empty states: large Fraunces phrase plus one Instrument Sans guidance line.

## Motion

- Spring-based, 150-300ms, purposeful. Wizard content slides in the direction of
  navigation; back reverses it. Shared-element transitions for the parameter
  count and bit selector.
- Haptic ticks on slider decades and selector changes.
- Reduce motion: cross-fade only, 120ms.

## Screenshot tests

`app/src/test/.../ScreenshotTest.kt` renders the gallery, home, every wizard
step, training and test screens with Paparazzi into
`app/build/reports/paparazzi/images`. CI uploads that folder as
`modelforge-screenshots` so the design can be reviewed without installing the
APK.

## Anti-generic checklist

The build fails review if any of these are true: Material 3 purple or dynamic
color, gradient background, glass blur, neon-on-dark, centered rounded hero
card, emoji icon, Roboto/Inter/system font, identical card grids for everything,
or a stock illustration.

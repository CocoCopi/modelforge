#!/usr/bin/env python3
"""WCAG AA check for every semantic text/background pair used by ModelForge."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    name: str
    hex: str


PAPER = Token("paper", "#F4EFE6")
SURFACE = Token("surface", "#FBF8F2")
INK = Token("ink", "#1B1A17")
INK_MUTED = Token("inkMuted", "#6B655A")
HAIRLINE = Token("hairline", "#DDD5C6")
EMBER = Token("ember", "#D9481C")
EMBER_TEXT = Token("emberText", "#B83A12")
VERDIGRIS = Token("verdigris", "#1F5C54")
OCHRE = Token("ochre", "#B8860B")
OCHRE_INK = Token("ochreInk", "#7D5906")
OCHRE_TINT = Token("ochreTint", "#F3E3B3")
ERROR = Token("error", "#A8281E")
ERROR_TINT = Token("errorTint", "#F6D9D4")
INK_GHOST = Token("inkGhost", "#6F685C")

# (foreground, background, minimum ratio, usage)
PAIRS = [
    (INK, PAPER, 4.5, "body"),
    (INK, SURFACE, 4.5, "body"),
    (INK, OCHRE_TINT, 4.5, "warning body"),
    (INK, ERROR_TINT, 4.5, "error body"),
    (INK_MUTED, PAPER, 4.5, "secondary body"),
    (INK_MUTED, SURFACE, 4.5, "secondary body"),
    (INK_GHOST, PAPER, 4.5, "disabled label"),
    (INK_GHOST, SURFACE, 4.5, "disabled label"),
    (EMBER_TEXT, PAPER, 4.5, "accent body/text"),
    (EMBER_TEXT, SURFACE, 4.5, "accent body/text"),
    (EMBER_TEXT, OCHRE_TINT, 4.5, "warning accent"),
    (PAPER, EMBER_TEXT, 4.5, "primary button label"),
    (PAPER, INK, 4.5, "ink button label"),
    (VERDIGRIS, PAPER, 4.5, "success body"),
    (VERDIGRIS, SURFACE, 4.5, "success body"),
    (ERROR, ERROR_TINT, 4.5, "error text"),
    (OCHRE_INK, OCHRE_TINT, 4.5, "warning text"),
    (OCHRE_INK, PAPER, 4.5, "warning text"),
    (EMBER, PAPER, 3.0, "large accent/icon"),
    (EMBER, SURFACE, 3.0, "large accent/icon"),
]


def _linear(channel: int) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def ratio(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def main() -> int:
    failures = 0
    for fg, bg, minimum, usage in PAIRS:
        if usage == "n/a":
            continue
        value = ratio(fg.hex, bg.hex)
        status = "PASS" if value >= minimum else "FAIL"
        print(f"{status} {value:5.2f} (min {minimum:.1f}) {fg.name} on {bg.name} - {usage}")
        if value < minimum:
            failures += 1
    if failures:
        print(f"\n{failures} contrast pair(s) failed WCAG AA")
        return 1
    print("\nAll semantic pairs meet WCAG AA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

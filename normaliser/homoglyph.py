"""
normaliser/homoglyph.py
------------------------
Normalises homoglyph substitution attacks.

Catches lookalike character substitutions where attacker replaces
Latin characters with visually identical characters from other
unicode blocks to evade pattern matching.

Common attack:
    "іgnore" — Cyrillic 'і' (U+0456) substituted for Latin 'i'

Coverage: Common Cyrillic, Greek, and other script lookalikes.
Full coverage of all unicode homoglyphs is out of scope —
this catches the most common practical attacks.
"""

import unicodedata
from .base import Decoder

# Mapping of common homoglyphs to their Latin ASCII equivalents
# Key: unicode character, Value: ASCII replacement
_HOMOGLYPHS: dict[str, str] = {
    # Cyrillic lookalikes
    "\u0430": "a",   # а → a
    "\u0435": "e",   # е → e
    "\u0456": "i",   # і → i
    "\u043e": "o",   # о → o
    "\u0440": "r",   # р → r
    "\u0441": "c",   # с → c
    "\u0445": "x",   # х → x
    "\u0443": "y",   # у → y
    "\u0410": "A",   # А → A
    "\u0412": "B",   # В → B
    "\u0415": "E",   # Е → E
    "\u041c": "M",   # М → M
    "\u041d": "H",   # Н → H
    "\u041e": "O",   # О → O
    "\u0420": "P",   # Р → P
    "\u0421": "C",   # С → C
    "\u0422": "T",   # Т → T
    "\u0425": "X",   # Х → X
    # Greek lookalikes
    "\u03b1": "a",   # α → a
    "\u03b2": "b",   # β → b
    "\u03b5": "e",   # ε → e
    "\u03b9": "i",   # ι → i
    "\u03bf": "o",   # ο → o
    "\u03c1": "p",   # ρ → p
    "\u03c5": "u",   # υ → u
    "\u0391": "A",   # Α → A
    "\u0392": "B",   # Β → B
    "\u0395": "E",   # Ε → E
    "\u0396": "Z",   # Ζ → Z
    "\u0397": "H",   # Η → H
    "\u0399": "I",   # Ι → I
    "\u039a": "K",   # Κ → K
    "\u039c": "M",   # Μ → M
    "\u039d": "N",   # Ν → N
    "\u039f": "O",   # Ο → O
    "\u03a1": "P",   # Ρ → P
    "\u03a4": "T",   # Τ → T
    "\u03a5": "Y",   # Υ → Y
    "\u03a7": "X",   # Χ → X
    # Full-width latin (common in East Asian text attacks)
    "\uff41": "a", "\uff42": "b", "\uff43": "c", "\uff44": "d",
    "\uff45": "e", "\uff46": "f", "\uff47": "g", "\uff48": "h",
    "\uff49": "i", "\uff4a": "j", "\uff4b": "k", "\uff4c": "l",
    "\uff4d": "m", "\uff4e": "n", "\uff4f": "o", "\uff50": "p",
    "\uff51": "q", "\uff52": "r", "\uff53": "s", "\uff54": "t",
    "\uff55": "u", "\uff56": "v", "\uff57": "w", "\uff58": "x",
    "\uff59": "y", "\uff5a": "z",
    "\uff21": "A", "\uff22": "B", "\uff23": "C", "\uff24": "D",
    "\uff25": "E", "\uff26": "F", "\uff27": "G", "\uff28": "H",
    "\uff29": "I", "\uff2a": "J", "\uff2b": "K", "\uff2c": "L",
    "\uff2d": "M", "\uff2e": "N", "\uff2f": "O", "\uff30": "P",
    "\uff31": "Q", "\uff32": "R", "\uff33": "S", "\uff34": "T",
    "\uff35": "U", "\uff36": "V", "\uff37": "W", "\uff38": "X",
    "\uff39": "Y", "\uff3a": "Z",
}


class HomoglyphDecoder(Decoder):
    """Replace common unicode homoglyphs with ASCII equivalents."""

    name = "homoglyph"

    def decode(self, text: str) -> str:
        # First apply NFKC normalisation — handles many common cases
        text = unicodedata.normalize("NFKC", text)
        # Then apply manual homoglyph table for remainder
        return "".join(_HOMOGLYPHS.get(c, c) for c in text)

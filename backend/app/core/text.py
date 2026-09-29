"""Text normalization for card names and set codes.

These functions are shared by the database layer (values are normalized before
persistence) and by the resolver (OCR output is normalized before lookup), so
both sides always agree on the canonical form.

Design rules
------------
* ``normalize_card_name`` / ``normalize_set_code`` are *conservative*: they only
  canonicalize case, whitespace, punctuation and character set.  They never
  "fix" OCR confusions.
* ``set_code_variants`` generates *candidate* corrections for the well-known OCR
  confusions (O/0, I/1, L/1, S/5, B/8, ...).  The resolver tries them against the
  database; a variant is only accepted if it actually exists.  This is how we
  avoid blindly rewriting every character.

Variant ordering (most to least trustworthy)
--------------------------------------------
1. the plain normalized input (as-is);
2. the input with a dropped hyphen re-inserted (``JOTLEN045`` -> ``JOTL-EN045``,
   ``LOB001`` -> ``LOB-001``);
3. a set-code-shaped token extracted from a longer reading
   (``1st Edition JOTL-EN045`` -> ``JOTL-EN045``);
4. the suffix coerced to the modern ``<2 letters><3 digits>`` shape;
5. prefix alternates (fewest substitutions first) combined with the coerced
   suffix, then with the raw suffix.
"""

from __future__ import annotations

import re
import unicodedata
from itertools import product

# Canonical set-code shape: PREFIX-SUFFIX, e.g. JOTL-EN045, LOB-001, DUEA-ENSE1, MP14-EN095.
SET_CODE_RE = re.compile(r"^(?P<prefix>[A-Z0-9]{2,5})-(?P<suffix>[A-Z0-9]{3,6})$")
# The usual modern suffix: two region letters followed by three digits (EN045).
MODERN_SUFFIX_RE = re.compile(r"^(?P<region>[A-Z]{1,2})(?P<number>[0-9]{3})$")
# A code whose hyphen was dropped by OCR: JOTLEN045 -> JOTL-EN045 (region letters + 3 digits).
HYPHENLESS_RE = re.compile(r"^(?P<prefix>[A-Z0-9]{2,5})(?P<region>[A-Z]{2})(?P<number>[0-9OILSB]{3})$")
# A legacy (pre-2004, no region) code whose hyphen was dropped: LOB001 -> LOB-001, MRD071 -> MRD-071.
HYPHENLESS_LEGACY_RE = re.compile(r"^(?P<prefix>[A-Z][A-Z0-9]{1,3})(?P<number>[0-9]{3})$")

# Common OCR confusions.  Direction matters: use the map for the *expected* class.
DIGIT_FOR_LETTER: dict[str, str] = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "|": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}
LETTER_FOR_DIGIT: dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}
# Bidirectional map used for the prefix, where either class is plausible.
PREFIX_ALTERNATES: dict[str, tuple[str, ...]] = {
    "0": ("O",), "O": ("0",),
    "1": ("I", "L"), "I": ("1",), "L": ("1",),
    "5": ("S",), "S": ("5",),
    "8": ("B",), "B": ("8",),
    "2": ("Z",), "Z": ("2",),
    "6": ("G",), "G": ("6",),
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DASHES = {"‐", "‑", "‒", "–", "—", "―", "−", "_", "­"}


def strip_accents(text: str) -> str:
    """Return ``text`` with combining accents removed (NFKD fold)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_card_name(text: str | None) -> str:
    """Canonical comparison form of a card name.

    * accents folded, lower-cased
    * ``&`` treated as the word ``and``
    * every run of non-alphanumeric characters collapsed to a single space

    ``"Armades, Keeper of Boundaries"`` -> ``"armades keeper of boundaries"``.
    The function is idempotent.
    """
    if not text:
        return ""
    folded = strip_accents(text).lower().replace("&", " and ")
    return _NON_ALNUM_RE.sub(" ", folded).strip()


def normalize_set_code(text: str | None) -> str:
    """Canonical form of a set code: upper-case, no whitespace, only ``A-Z 0-9 -``.

    Unicode dashes (and underscores) are mapped to ``-``; repeated/leading/trailing
    hyphens are removed.  No OCR-confusion substitution happens here.
    """
    if not text:
        return ""
    folded = strip_accents(unicodedata.normalize("NFKC", text)).upper()
    folded = "".join("-" if ch in _DASHES else ch for ch in folded)
    folded = re.sub(r"[^A-Z0-9-]", "", folded)
    folded = re.sub(r"-{2,}", "-", folded).strip("-")
    return folded


def looks_like_set_code(text: str | None) -> bool:
    """True if ``text`` (after normalization) has a plausible set-code shape."""
    return bool(SET_CODE_RE.match(normalize_set_code(text)))


def split_set_code(code: str) -> tuple[str, str] | None:
    """Split a normalized code into (prefix, suffix) or None if it does not parse."""
    match = SET_CODE_RE.match(code)
    if not match:
        return None
    return match.group("prefix"), match.group("suffix")


def _coerce_suffix(suffix: str) -> str:
    """Coerce a suffix towards the modern shape ``LL DDD``.

    Region letters that look like digits are mapped to letters, number digits that
    look like letters are mapped to digits.  Suffixes that do not have length 5
    (e.g. ``001`` on old sets, ``E001``) are left untouched.
    """
    if len(suffix) != 5:
        return suffix
    region = "".join(LETTER_FOR_DIGIT.get(ch, ch) for ch in suffix[:2])
    number = "".join(DIGIT_FOR_LETTER.get(ch, ch) for ch in suffix[2:])
    return region + number


def _substitutions(original: str, candidate: str) -> int:
    """Number of positions at which two equal-length strings differ."""
    return sum(1 for a, b in zip(original, candidate) if a != b)


def _prefix_alternatives(prefix: str, limit: int) -> list[str]:
    """All combinations of per-character alternates for the prefix.

    The original comes first, then alternates with fewer substitutions before
    those with more (stable within the same count).
    """
    options = [(ch, *PREFIX_ALTERNATES.get(ch, ())) for ch in prefix]
    combos = ["".join(combo) for combo in product(*options)]
    combos.sort(key=lambda candidate: _substitutions(prefix, candidate))
    return combos[:limit]


def _reinsert_hyphen(code: str) -> str | None:
    """Return ``code`` with a dropped hyphen re-inserted, or None when it does not apply."""
    if "-" in code:
        return None
    modern = HYPHENLESS_RE.match(code)
    if modern:
        return f"{modern.group('prefix')}-{modern.group('region')}{modern.group('number')}"
    legacy = HYPHENLESS_LEGACY_RE.match(code)
    if legacy:
        return f"{legacy.group('prefix')}-{legacy.group('number')}"
    return None


def _set_code_tokens(text: str) -> list[str]:
    """Normalized whitespace-separated tokens of ``text`` that look like set codes."""
    tokens: list[str] = []
    for token in text.split():
        normalized = normalize_set_code(token)
        if SET_CODE_RE.match(normalized):
            tokens.append(normalized)
        else:
            rehyphenated = _reinsert_hyphen(normalized)
            if rehyphenated:
                tokens.append(rehyphenated)
    return tokens


def set_code_variants(text: str | None, *, limit: int = 48) -> list[str]:
    """Ordered candidate spellings for an OCR'd set code.

    The first entry is always the plain normalized input (when it has a set-code
    shape).  Subsequent entries are increasingly speculative and MUST be validated
    against the database before being trusted.  Only shapes that pass
    :data:`SET_CODE_RE` are returned; the list is empty when nothing plausible can
    be derived from ``text``.
    """
    if not text:
        return []
    normalized = normalize_set_code(text)
    if not normalized:
        return []

    ordered: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        if candidate and candidate not in seen and SET_CODE_RE.match(candidate):
            seen.add(candidate)
            ordered.append(candidate)

    # 1. As-is.
    add(normalized)

    # 2. Re-insert a dropped hyphen (modern or legacy shape).
    rehyphenated = _reinsert_hyphen(normalized)
    if rehyphenated:
        add(rehyphenated)

    # 3. A set-code-shaped token inside a longer reading ("1st Edition JOTL-EN045"),
    #    only when the reading as a whole is not already a set code.
    if not ordered:
        for token in _set_code_tokens(text):
            add(token)

    # 4./5. Character-confusion corrections for every base shape found so far.
    for base in list(ordered):
        parts = split_set_code(base)
        if parts is None:
            continue
        prefix, suffix = parts
        coerced_suffix = _coerce_suffix(suffix)
        add(f"{prefix}-{coerced_suffix}")
        for alt_prefix in _prefix_alternatives(prefix, limit):
            add(f"{alt_prefix}-{coerced_suffix}")
            if len(ordered) >= limit:
                break
        if coerced_suffix != suffix:
            for alt_prefix in _prefix_alternatives(prefix, limit):
                add(f"{alt_prefix}-{suffix}")
                if len(ordered) >= limit:
                    break
        if len(ordered) >= limit:
            break

    return ordered[:limit]


def set_code_prefix(code: str) -> str:
    """Return the set prefix (``JOTL`` for ``JOTL-EN045``) or the whole code if unparseable."""
    parts = split_set_code(normalize_set_code(code))
    return parts[0] if parts else normalize_set_code(code)

"""Semantic canonicalization and matching for tool-call arguments.

Exact equality on tool-call arguments is brittle: ``"$1,000.00"`` and ``1000``
mean the same amount, ``"01/05/2021"`` and ``"2021-01-05"`` the same date,
``"A-100"`` and ``"a100"`` the same id, ``["b", "a"]`` and ``["a", "b"]`` the
same set of tags. This module canonicalizes values by *kind* so the comparison
reflects intent, adds fuzzy string equivalence for the cases exact matching is
too strict for, and parses ``name(arg=value)`` call expressions via Python's
``ast`` for AST-style function-call validation.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any

# Date input formats tried in order; the canonical form is always ISO (yyyy-mm-dd).
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%B %d %Y",
    "%b %d %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)


def canonical_date(value: Any) -> str | None:
    """Canonicalize a date to ISO ``yyyy-mm-dd``; ``None`` if unparseable."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def canonical_amount(value: Any) -> float | None:
    """Canonicalize a monetary/numeric amount to a float; ``None`` if not numeric.

    Strips currency symbols, thousands separators, and surrounding whitespace.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    # Drop everything but digits, sign, and the decimal point.
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def canonical_id(value: Any) -> str:
    """Canonicalize an identifier: uppercase, alphanumerics only.

    ``"a-100"``, ``"A 100"``, and ``"A100"`` all canonicalize to ``"A100"``.
    """
    return re.sub(r"[^0-9A-Za-z]", "", str(value)).upper()


def canonical_text(value: Any) -> str:
    """Casefold and collapse whitespace for case-insensitive text equality."""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def canonical_alias(value: Any, aliases: dict[str, Any]) -> Any:
    """Map a value through an alias table (case-insensitive keys).

    Useful for enums: ``{"yes": True, "y": True, "no": False}``. Unknown values
    pass through casefolded so equal raw values still compare equal.
    """
    folded = {str(k).casefold(): v for k, v in aliases.items()}
    key = str(value).casefold()
    return folded.get(key, key)


def canonical_sorted(value: Any) -> Any:
    """Order-insensitive canonical form for list/tuple values.

    Elements are canonicalized as text then sorted, so ``["b","a"]`` and
    ``["a","b"]`` match. Non-sequences are returned unchanged.
    """
    if isinstance(value, list | tuple):
        return sorted(canonical_text(v) for v in value)
    return value


# kind -> canonicalizer. ``alias`` is handled separately (needs a table).
_CANONICALIZERS = {
    "date": canonical_date,
    "amount": canonical_amount,
    "id": canonical_id,
    "text": canonical_text,
    "casing": canonical_text,
    "lower": canonical_text,
    "sorted": canonical_sorted,
    "ordering": canonical_sorted,
}


def canonicalize_value(value: Any, kind: str, *, aliases: dict[str, Any] | None = None) -> Any:
    """Canonicalize ``value`` according to ``kind``.

    Supported kinds: ``date``, ``amount``, ``id``, ``text``/``casing``/``lower``,
    ``sorted``/``ordering``, and ``alias``/``enum`` (which use ``aliases``).
    Unknown kinds return the value unchanged.
    """
    if kind in ("alias", "enum"):
        return canonical_alias(value, aliases or {})
    func = _CANONICALIZERS.get(kind)
    return func(value) if func else value


def fuzzy_ratio(a: Any, b: Any) -> float:
    """Similarity in [0, 1] between two values' canonical text forms."""
    return SequenceMatcher(None, canonical_text(a), canonical_text(b)).ratio()


def fuzzy_equal(a: Any, b: Any, threshold: float = 0.85) -> bool:
    """True when ``a`` and ``b`` are at least ``threshold`` similar as text."""
    return fuzzy_ratio(a, b) >= threshold


@dataclass
class ArgSpec:
    """How to compare one argument: by kind, with optional aliases/fuzziness."""

    kind: str = "exact"  # exact | date | amount | id | text | sorted | alias | fuzzy
    aliases: dict[str, Any] = field(default_factory=dict)
    fuzzy_threshold: float = 0.85


def values_match(expected: Any, actual: Any, spec: ArgSpec) -> bool:
    """Compare two values under a single ``ArgSpec``."""
    if spec.kind == "exact":
        return bool(expected == actual)
    if spec.kind == "fuzzy":
        return fuzzy_equal(expected, actual, spec.fuzzy_threshold)
    exp = canonicalize_value(expected, spec.kind, aliases=spec.aliases)
    act = canonicalize_value(actual, spec.kind, aliases=spec.aliases)
    return bool(exp == act)


def args_match(
    expected: dict[str, Any],
    actual: dict[str, Any],
    specs: dict[str, ArgSpec] | None = None,
    *,
    default: ArgSpec | None = None,
) -> bool:
    """Subset-match ``expected`` args against ``actual`` with per-field specs.

    Every key in ``expected`` must be present in ``actual`` and match under its
    spec (looked up in ``specs``, else ``default``, else exact equality). Extra
    keys in ``actual`` are ignored, matching the existing subset semantics.
    """
    specs = specs or {}
    default = default or ArgSpec()
    for key, exp in expected.items():
        if key not in actual:
            return False
        if not values_match(exp, actual[key], specs.get(key, default)):
            return False
    return True


@dataclass(frozen=True)
class ParsedCall:
    """A parsed ``name(pos, kw=value)`` call expression."""

    name: str
    args: list[Any]
    kwargs: dict[str, Any]


def parse_call(expr: str) -> ParsedCall:
    """Parse a function-call expression into name + literal args via ``ast``.

    Argument values must be Python literals (``ast.literal_eval``). Raises
    ``ValueError`` for anything that is not a single literal call expression --
    no names, attributes, or arbitrary code are evaluated.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid call expression: {expr!r}") from exc
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError(f"expected a simple name(...) call, got: {expr!r}")
    try:
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"call arguments must be literals: {expr!r}") from exc
    return ParsedCall(call.func.id, args, kwargs)

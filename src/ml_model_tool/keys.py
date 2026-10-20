"""The one table of spec keys: every key of the data spec (per source) and of the model spec (per model
type) is declared once, with its default, the values it takes and what it means. Made from it:

- the value checks (``check_keys``; a wrong value is a ``SpecError`` naming the key, the value given
  and what to write instead)
- the key names for the shape check of the command layer (unknown keys are rejected before running)
- the answers of the ``spec`` command (keys, defaults, values, meaning and a ready-to-write example)
- the key tables in docs/specs.md (``python -m ml_model_tool.keys docs/specs.md`` rewrites them; a test
  fails when they are out of date)

Checks that relate several keys (``volfrac`` within limits that depend on another key, the U-Net's
attention channels, ...) stay with the source or the model spec, and raise ``SpecError`` too.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agk.lang import tr


class _Required:
    def __repr__(self) -> str:
        return "required"


REQUIRED = _Required()


class SpecError(ValueError):
    """A wrong spec: ``where`` (the key, e.g. ``train.lr``), the ``value`` given and what is ``expected``."""

    def __init__(self, where: str, message: str, expected: str = ""):
        super().__init__(f"{where}: {message}" if where else message)
        self.where, self.expected = where, expected


def _show(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _bounds(lo, hi, lo_open=False, hi_open=False) -> tuple[str, str]:
    """(ja, en) of a range: ``1 以上`` / ``>= 1``, ``0 より大きい`` / ``> 0``, ``0 以上 1 未満`` / ``in [0, 1)``."""
    if lo is not None and hi is not None:
        a, b = "(" if lo_open else "[", ")" if hi_open else "]"
        return (f"{lo} {'より大きく' if lo_open else '以上'} {hi} {'未満' if hi_open else '以下'}", f"in {a}{lo}, {hi}{b}")
    if lo is not None:
        return (f"{lo} {'より大きい' if lo_open else '以上'}", f"{'>' if lo_open else '>='} {lo}")
    if hi is not None:
        return (f"{hi} {'未満' if hi_open else '以下'}", f"{'<' if hi_open else '<='} {hi}")
    return ("", "")


def _in(v, lo, hi, lo_open=False, hi_open=False) -> bool:
    if lo is not None and (v <= lo if lo_open else v < lo):
        return False
    if hi is not None and (v >= hi if hi_open else v > hi):
        return False
    return True


# ---------------------------------------------------------------------------
# what a value may be
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Int:
    lo: Optional[int] = None
    hi: Optional[int] = None

    def ok(self, v) -> bool:
        return _int(v) and _in(v, self.lo, self.hi)

    def text(self) -> tuple[str, str]:
        ja, en = _bounds(self.lo, self.hi)
        return (f"{ja}の整数" if ja else "整数", f"an integer {en}".strip())


@dataclass(frozen=True)
class Num:
    lo: Optional[float] = None
    hi: Optional[float] = None
    lo_open: bool = False
    hi_open: bool = False

    def ok(self, v) -> bool:
        return _num(v) and _in(v, self.lo, self.hi, self.lo_open, self.hi_open)

    def text(self) -> tuple[str, str]:
        ja, en = _bounds(self.lo, self.hi, self.lo_open, self.hi_open)
        return (f"{ja}の数" if ja else "数", f"a number {en}".strip())


@dataclass(frozen=True)
class Bool:
    def ok(self, v) -> bool:
        return isinstance(v, bool)

    def text(self) -> tuple[str, str]:
        return ("true か false", "true or false")


@dataclass(frozen=True)
class Choice:
    options: tuple

    def ok(self, v) -> bool:
        return v in self.options

    def text(self) -> tuple[str, str]:
        opts = " / ".join(_show(o) for o in self.options)
        return (f"{opts} のどれか", f"one of {opts}")


@dataclass(frozen=True)
class Pair:
    """``[min, max]`` with lo <= min <= max <= hi."""

    lo: float
    hi: float
    integer: bool = False

    def ok(self, v) -> bool:
        good = _int if self.integer else _num
        return (isinstance(v, list) and len(v) == 2 and all(good(e) for e in v)
                and self.lo <= v[0] <= v[1] <= self.hi)

    def text(self) -> tuple[str, str]:
        kind = ("整数", "integers") if self.integer else ("数", "numbers")
        return (f"[最小, 最大] の{kind[0]} 2 つ（{self.lo} ≦ 最小 ≦ 最大 ≦ {self.hi}）",
                f"[min, max], two {kind[1]} with {self.lo} <= min <= max <= {self.hi}")


@dataclass(frozen=True)
class Subset:
    """A list of distinct options (``empty`` allows [])."""

    options: tuple
    empty: bool = False

    def ok(self, v) -> bool:
        return (isinstance(v, list) and (self.empty or len(v) > 0) and len(set(map(_show, v))) == len(v)
                and all(e in self.options for e in v))

    def text(self) -> tuple[str, str]:
        opts = ", ".join(_show(o) for o in self.options)
        return (f"[{opts}] から選んだリスト" + ("（[] も可）" if self.empty else "（1 つ以上）"),
                f"a list of {opts}" + (" ([] allowed)" if self.empty else " (at least one)"))


@dataclass(frozen=True)
class Ints:
    """A list of integers in [lo, hi], ``min_len``-``max_len`` long (``unique``: no repeats)."""

    lo: int
    hi: int
    min_len: int = 0
    max_len: int = 64
    unique: bool = False

    def ok(self, v) -> bool:
        return (isinstance(v, list) and self.min_len <= len(v) <= self.max_len
                and all(_int(e) and self.lo <= e <= self.hi for e in v)
                and (not self.unique or len(set(v)) == len(v)))

    def text(self) -> tuple[str, str]:
        n = (f"{self.min_len}〜{self.max_len} 個", f"{self.min_len}-{self.max_len}")
        return (f"{self.lo}〜{self.hi} の整数のリスト（{n[0]}）", f"a list of {n[1]} integers in [{self.lo}, {self.hi}]")


@dataclass(frozen=True)
class Text:
    def ok(self, v) -> bool:
        return isinstance(v, str)

    def text(self) -> tuple[str, str]:
        return ("文字列", "a string")


@dataclass(frozen=True)
class Section:
    """A JSON object of further keys (``model`` / ``train`` / ``sample`` of the model spec)."""

    def ok(self, v) -> bool:
        return isinstance(v, dict)

    def text(self) -> tuple[str, str]:
        return ("鍵と値の組（{…}）", "an object ({...})")


@dataclass(frozen=True)
class Names:
    """A list of names (``inputs``: channel names of the dataset, checked against it later)."""

    def ok(self, v) -> bool:
        return isinstance(v, list) and all(isinstance(e, str) for e in v)

    def text(self) -> tuple[str, str]:
        return ("名前のリスト", "a list of names")


# ---------------------------------------------------------------------------
# a key
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Key:
    name: str
    default: Any
    rule: Any
    help: tuple[str, str]
    # how the default reads in the docs when it is not the JSON of ``default`` (e.g. "all channels")
    default_text: tuple[str, str] = ("", "")

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def values(self) -> str:
        return tr(*self.rule.text())

    def default_shown(self) -> str:
        if self.default_text[0]:
            return tr(*self.default_text)
        return tr("（必須）", "(required)") if self.required else _show(self.default)

    def expected(self) -> str:
        return self.values() + ("" if self.required else tr(f"（既定 {self.default_shown()}）",
                                                           f" (default {self.default_shown()})"))

    def check(self, value: Any, where: str) -> None:
        if not self.rule.ok(value):
            raise SpecError(where, tr(f"{_show(value)} は使えない。{self.expected()}",
                                      f"{_show(value)} is not allowed; {self.expected()}"), self.expected())

    def row(self) -> dict:
        """One key as the ``spec`` command shows it."""
        out = {"key": self.name, "values": self.values(), "meaning": tr(*self.help)}
        if self.required:
            out["required"] = True
        else:
            out["default"] = tr(*self.default_text) if self.default_text[0] else self.default
        return out


def defaults(keys: list[Key]) -> dict:
    return {k.name: k.default for k in keys if not k.required}


def names(keys: list[Key]) -> list[str]:
    return [k.name for k in keys]


def check_keys(doc: dict, keys: list[Key], prefix: str = "") -> dict:
    """Defaults filled in and every value checked. Unknown keys are rejected too (the command layer
    rejects them first; this keeps the library strict when used directly)."""
    by = {k.name: k for k in keys}
    unknown = [k for k in doc if k not in by]
    if unknown:
        where = prefix.rstrip(".") or tr("仕様", "spec")
        raise SpecError(where, tr(f"知らないキー {', '.join(unknown)}（使えるキー: {', '.join(by)}）",
                                  f"unknown keys {', '.join(unknown)} (keys: {', '.join(by)})"), ", ".join(by))
    out = {}
    for k in keys:
        if k.name in doc:
            k.check(doc[k.name], prefix + k.name)
            out[k.name] = doc[k.name]
        elif k.required:
            raise SpecError(prefix + k.name, tr(f"必須のキーが無い。{k.values()}",
                                                f"required key missing; {k.values()}"), k.values())
        else:
            out[k.name] = json.loads(json.dumps(k.default))          # a copy
    return out


def example(keys: list[Key]) -> dict:
    """The keys with their defaults (required keys as ``"<required>"``): a spec to copy and edit."""
    return {k.name: ("<required>" if k.required else k.default) for k in keys}


# ---------------------------------------------------------------------------
# the tables in docs/specs.md
# ---------------------------------------------------------------------------
def markdown(keys: list[Key]) -> str:
    rows = ["| キー | 既定 | 値 | 意味 |", "|---|---|---|---|"]
    for k in keys:
        default = k.default_text[0] or ("（必須）" if k.required else f"`{_show(k.default)}`")
        rows.append(f"| `{k.name}` | {default} | {k.rule.text()[0]} | {k.help[0]} |")
    return "\n".join(rows)


def tables() -> dict[str, list[Key]]:
    """Every key table, by the name used in the docs markers ``<!-- keys:NAME -->``."""
    from . import spec
    from .sources import SOURCES

    out = {f"data/{name}": src.KEYS for name, src in SOURCES.items()}
    for kind in spec.MODEL_TYPES:
        out[f"model/{kind}"] = spec.model_keys(kind)
        out[f"sample/{kind}"] = spec.sample_keys(kind) if kind in spec.GENERATIVE else []
    out["train"] = spec.train_keys("fno")
    out["spec"] = spec.TOP_KEYS
    return out


_MARK = re.compile(r"(<!-- keys:(?P<name>[\w/-]+) -->\n)(?P<body>.*?)(<!-- /keys -->)", re.S)


def render_docs(text: str) -> str:
    """``text`` with every ``<!-- keys:NAME -->`` ... ``<!-- /keys -->`` block made from the key tables."""
    from agk.lang import using

    found = tables()

    def sub(m: re.Match) -> str:
        name = m.group("name")
        if name not in found:
            raise KeyError(f"no key table {name!r} (known: {', '.join(found)})")
        with using("ja"):
            return m.group(1) + markdown(found[name]) + "\n" + m.group(4)

    return _MARK.sub(sub, text)


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m ml_model_tool.keys docs/specs.md``: rewrite the key tables of the file in place."""
    for path in (argv if argv is not None else sys.argv[1:]):
        p = Path(path)
        old = p.read_text(encoding="utf-8")
        new = render_docs(old)
        if new != old:
            p.write_text(new, encoding="utf-8")
            print(f"updated {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

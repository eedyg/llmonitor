"""Minimal Prometheus text-format parser (0.0.4) + histogram quantiles.

No third-party dependency: the engines export plain text and we only need
names, labels, counters, gauges and cumulative histograms.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Sample:
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Family:
    name: str
    type: str = "untyped"
    help: str = ""
    samples: list[Sample] = field(default_factory=list)


def _split_labels(blob: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    i, n = 0, len(blob)
    while i < n:
        while i < n and blob[i] in ", ":
            i += 1
        start = i
        while i < n and blob[i] != "=":
            i += 1
        key = blob[start:i].strip()
        if i >= n:
            break
        i += 1  # '='
        if i < n and blob[i] == '"':
            i += 1
            buf: list[str] = []
            while i < n:
                ch = blob[i]
                if ch == "\\" and i + 1 < n:
                    buf.append(blob[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    i += 1
                    break
                buf.append(ch)
                i += 1
            labels[key] = "".join(buf)
        else:
            start = i
            while i < n and blob[i] != ",":
                i += 1
            labels[key] = blob[start:i].strip()
    return labels


def parse(text: str) -> dict[str, Family]:
    families: dict[str, Family] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(" ", 3)
            if len(parts) >= 3 and parts[1] == "TYPE":
                fam = families.setdefault(parts[2], Family(parts[2]))
                fam.type = parts[3].strip()
            elif len(parts) >= 4 and parts[1] == "HELP":
                fam = families.setdefault(parts[2], Family(parts[2]))
                fam.help = parts[3].strip()
            continue
        name = line
        labels: dict[str, str] = {}
        brace = line.find("{")
        if brace >= 0:
            end = line.rfind("}")
            if end > brace:
                name = line[:brace]
                labels = _split_labels(line[brace + 1 : end])
                rest = line[end + 1 :].strip()
            else:  # malformed; skip
                continue
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            rest = parts[1]
        if not rest:
            continue
        token = rest.split()[0]
        try:
            value = float(token)
        except ValueError:
            continue
        familias = families.get(name)
        if familias is None:
            familias = families[name] = Family(name)
        familias.samples.append(Sample(value, labels))
    return families


def _match(labels: dict[str, str], want: dict[str, str]) -> bool:
    return all(labels.get(k) == v for k, v in want.items())


class Metrics:
    """Query helper over a parsed scrape."""

    def __init__(self, text: str) -> None:
        self.families = parse(text)
        self.ts = None

    def __bool__(self) -> bool:
        return bool(self.families)

    def has(self, name: str) -> bool:
        return name in self.families

    def samples(self, name: str, **labels: str) -> list[Sample]:
        fam = self.families.get(name)
        if not fam:
            return []
        return [s for s in fam.samples if _match(s.labels, labels)]

    def value(self, name: str, **labels: str) -> float | None:
        got = self.samples(name, **labels)
        if not got:
            return None
        return got[0].value

    def total(self, name: str, **labels: str) -> float | None:
        got = self.samples(name, **labels)
        if not got:
            return None
        return float(sum(s.value for s in got))

    def label_values(self, name: str, label: str) -> list[str]:
        fam = self.families.get(name)
        if not fam:
            return []
        return sorted({s.labels.get(label, "") for s in fam.samples})

    # -- histograms ---------------------------------------------------------
    def histogram(self, name: str, **labels: str) -> tuple[list[tuple[float, float]], float, float] | None:
        """Return (buckets[(le, cumulative)], sum, count)."""
        buckets: list[tuple[float, float]] = []
        for sample in self.samples(name + "_bucket", **labels):
            le = sample.labels.get("le")
            if le is None:
                continue
            try:
                bound = math.inf if le == "+Inf" else float(le)
            except ValueError:
                continue
            buckets.append((bound, sample.value))
        if not buckets:
            return None
        buckets.sort(key=lambda item: item[0])
        total_sum = self.value(name + "_sum", **labels) or 0.0
        count = self.value(name + "_count", **labels)
        if count is None:
            count = buckets[-1][1]
        return buckets, float(total_sum), float(count)

    def quantile(self, name: str, q: float, **labels: str) -> float | None:
        hist = self.histogram(name, **labels)
        if not hist:
            return None
        buckets, _sum, count = hist
        if count <= 0:
            return None
        rank = q * count
        prev_bound = 0.0
        prev_count = 0.0
        for bound, cumulative in buckets:
            if cumulative >= rank:
                if math.isinf(bound):
                    return prev_bound
                span = cumulative - prev_count
                if span <= 0:
                    return bound
                return prev_bound + (bound - prev_bound) * ((rank - prev_count) / span)
            prev_bound, prev_count = bound, cumulative
        return buckets[-1][0] if not math.isinf(buckets[-1][0]) else None

    def mean(self, name: str, **labels: str) -> float | None:
        hist = self.histogram(name, **labels)
        if not hist:
            return None
        _b, total_sum, count = hist
        return total_sum / count if count > 0 else None


def flatten(families: dict[str, Family]) -> dict[str, float]:
    """name{labels} -> value, useful for debugging / exports."""
    out: dict[str, float] = {}
    for name, fam in families.items():
        for sample in fam.samples:
            key = name
            if sample.labels:
                body = ",".join(f'{k}="{v}"' for k, v in sorted(sample.labels.items()))
                key = f"{name}{{{body}}}"
            out[key] = sample.value
    return out

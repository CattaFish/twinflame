"""Anchoring stage (plan Section B) — R8-invariant seed matches.

R8 renames identifiers but cannot rewrite the *content* of a string constant or
the *target* of a framework/library call. Two classes that share a rare string
(a unique log tag, URL, or error message) or call the same distinctive set of
`android/*`, `androidx/*`, `java/*`, `kotlin/*` APIs are almost certainly the
same class even after renaming. We turn those invariants into high-confidence
"anchor" pairs that the matcher locks in *before* structural scoring runs.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .model import Class

# Library/framework descriptor prefixes that survive R8 renaming verbatim.
FRAMEWORK_PREFIXES = (
    "Landroid/",
    "Landroidx/",
    "Ljava/",
    "Ljavax/",
    "Lkotlin/",
    "Lkotlinx/",
)

# Strings shorter than this, or purely numeric, carry too little signal.
MIN_STRING_LEN = 4
FRAMEWORK_WEIGHT = 0.25
HIERARCHY_WEIGHT = 1.0
_OBJECT_DESC = "Ljava/lang/Object;"
DEFAULT_MIN_CONFIDENCE = 0.5

# ★ 核心防爆阈值：如果一个 token 在单侧出现的类超过 20 个，按 IDF 权重极小且必然不是唯一锚点，
# 提前过滤跳过，杜绝 80,000 x 80,000 产生千万级笛卡尔积元组撑爆 35GB 内存。
MAX_ANCHOR_TOKEN_DF = 20


def useful_strings(c: Class) -> set[str]:
    return {s for s in c.strings if len(s) >= MIN_STRING_LEN and not s.isdigit()}


def hierarchy_tokens(c: Class) -> set[str]:
    toks: set[str] = set()
    sc = c.superclass
    if sc and sc != _OBJECT_DESC and sc.startswith(FRAMEWORK_PREFIXES):
        toks.add("S:" + sc)
    for iface in c.interfaces:
        if iface.startswith(FRAMEWORK_PREFIXES):
            toks.add("I:" + iface)
    return toks


def framework_calls(c: Class) -> Counter:
    counts: Counter = Counter()
    for m in c.methods:
        for ref in m.calls:
            if ref.startswith(FRAMEWORK_PREFIXES):
                counts[ref] += 1
    return counts


def _jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 0.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def seed_anchors(
    lhs: list[Class],
    rhs: list[Class],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[tuple[int, int, float]]:
    if not lhs or not rhs:
        return []

    def tokens(c: Class) -> dict[str, float]:
        toks: dict[str, float] = {s: 1.0 for s in useful_strings(c)}
        for h in hierarchy_tokens(c):
            toks["\x00" + h] = HIERARCHY_WEIGHT
        return toks

    lhs_tokens = [tokens(c) for c in lhs]
    rhs_tokens = [tokens(c) for c in rhs]

    n = len(lhs) + len(rhs)
    df: Counter = Counter()
    for ts in lhs_tokens:
        df.update(ts.keys())
    for ts in rhs_tokens:
        df.update(ts.keys())
    lhs_df: Counter = Counter()
    for ts in lhs_tokens:
        lhs_df.update(ts.keys())

    rhs_by_tok: dict[str, list[int]] = defaultdict(list)
    for j, ts in enumerate(rhs_tokens):
        for s in ts:
            rhs_by_tok[s].append(j)

    norm = math.log(n + 1) or 1.0

    evidence: dict[tuple[int, int], float] = defaultdict(float)
    for li, ts in enumerate(lhs_tokens):
        for s, weight in ts.items():
            ris = rhs_by_tok.get(s)
            if not ris:
                continue

            # ★ 核心剪枝：两侧出现频率过高的通用 Token（如公共接口、常规提示语）不作为锚点，
            # 避免巨量 (li, ri) 笛卡尔积耗尽 RAM
            if len(ris) > MAX_ANCHOR_TOKEN_DF or lhs_df[s] > MAX_ANCHOR_TOKEN_DF:
                continue

            tok_idf = math.log((n + 1) / df[s])
            split = lhs_df[s] * len(ris)
            w = weight * (tok_idf / norm) / split

            # 权重过低（即便累计也无法达到 0.5 门槛）的微弱项不写入，节省内存
            if w < 0.01:
                continue

            for ri in ris:
                evidence[(li, ri)] += w

    if not evidence:
        return []

    fw_lhs: dict[int, Counter] = {}
    fw_rhs: dict[int, Counter] = {}
    for (li, ri) in list(evidence):
        a = fw_lhs.setdefault(li, framework_calls(lhs[li]))
        b = fw_rhs.setdefault(ri, framework_calls(rhs[ri]))
        if a or b:
            evidence[(li, ri)] += FRAMEWORK_WEIGHT * _jaccard(a, b)

    ranked = sorted(
        ((conf, li, ri) for (li, ri), conf in evidence.items()),
        key=lambda t: (-t[0], t[1], t[2]),
    )

    used_l: set[int] = set()
    used_r: set[int] = set()
    anchors: list[tuple[int, int, float]] = []
    for raw, li, ri in ranked:
        conf = min(1.0, raw)
        if conf < min_confidence:
            break
        if li in used_l or ri in used_r:
            continue
        used_l.add(li)
        used_r.add(ri)
        anchors.append((li, ri, conf))
    return anchors

"""期待値計算と買い目選定。

期待値 = 的中確率 × オッズ。1.0を超えれば理論上プラスだが、
- モデル誤差（特に高配当側）
- 締切直前のオッズ変動（パリミュチュエルなので最終オッズは下がりがち）
があるので、しきい値は1.0より高め（初期値1.2）にしてバックテストで決める。
"""
from __future__ import annotations

import numpy as np

from . import config
from .model import combo_names, order_probs


def blend_with_market(p_model: np.ndarray, odds: np.ndarray, w: float) -> np.ndarray:
    """モデル確率とオッズから逆算した市場確率を対数線形で混ぜる（Benter方式）。"""
    valid = np.isfinite(odds) & (odds > 0)
    if w >= 1 or not valid.any():
        return p_model
    p_mkt = np.where(valid, 1 / np.where(valid, odds, 1), 0)
    p_mkt = p_mkt / p_mkt.sum()
    out = np.where(valid, np.exp(w * np.log(np.clip(p_model, 1e-9, 1)) + (1 - w) * np.log(np.clip(p_mkt, 1e-9, 1))), 0)
    return out / out.sum() * p_model[valid].sum() + np.where(valid, 0, p_model)


def kelly_stake(p: float, o: float, bankroll: float) -> int:
    f = (p * o - 1) / (o - 1) if o > 1 else 0
    stake = bankroll * min(max(f, 0) * config.KELLY_FRACTION, config.MAX_BET_SHARE)
    return int(max(100, round(stake / 100) * 100)) if f > 0 else 0


def analyze_race(win_p, odds: dict[str, dict[str, float]], lam2: float, lam3: float, *,
                 bankroll: float = 100_000, ev_th: float | None = None, blend: float | None = None,
                 min_prob: float | None = None, max_odds: float | None = None, max_points: int | None = None,
                 probs: dict | None = None):
    """1レース分：全組み合わせの確率・期待値と、推奨買い目を返す。"""
    ev_th = config.EV_THRESHOLD if ev_th is None else ev_th
    blend = config.MARKET_BLEND if blend is None else blend
    min_prob = config.MIN_PROB if min_prob is None else min_prob
    max_odds = config.MAX_ODDS if max_odds is None else max_odds
    max_points = config.MAX_POINTS_PER_RACE if max_points is None else max_points

    probs = probs or order_probs(np.asarray(win_p), lam2, lam3)  # バックテストでは事前計算を渡して高速化
    table, picks = [], []
    for kind, label in (("3t", "3連単"), ("2t", "2連単")):
        names = combo_names(kind)
        o = np.array([odds.get(kind, {}).get(c) or np.nan for c in names], float)
        p = blend_with_market(probs[kind], o, blend)
        ev = p * o
        for c, pi, oi, ei in zip(names, p, o, ev):
            row = {"type": label, "combo": c, "prob": float(pi),
                   "odds": None if np.isnan(oi) else float(oi),
                   "ev": None if np.isnan(ei) else float(ei)}
            table.append(row)
            if row["ev"] and ei >= ev_th and pi >= min_prob and oi <= max_odds:
                picks.append(row)
    picks.sort(key=lambda r: r["ev"], reverse=True)
    picks = picks[:max_points]
    for r in picks:
        r["tag"], r["stake"] = "本線", config.FLAT_STAKE
    roman = pick_roman(table, {r["combo"] for r in picks if r["type"] == "3連単"})
    return {"win_prob": [float(x) for x in np.asarray(win_p)], "table": table, "picks": picks, "roman": roman}


def pick_roman(table, exclude=()):
    """ロマン枠：100倍以上の3連単のうち、期待値が最も高い1点（基準に届かなければ出さない）。"""
    cands = [r for r in table if r["type"] == "3連単" and r["odds"] and r["ev"]
             and config.ROMAN_MIN_ODDS <= r["odds"] <= config.ROMAN_MAX_ODDS
             and r["prob"] >= config.ROMAN_MIN_PROB and r["ev"] >= config.ROMAN_EV
             and r["combo"] not in exclude]
    if not cands:
        return None
    best = dict(max(cands, key=lambda r: r["ev"]))
    best["tag"], best["stake"] = "ロマン", config.FLAT_STAKE
    return best

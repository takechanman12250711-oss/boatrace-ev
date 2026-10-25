"""バックテスト：過去期間で「本当に回収率が100%を超えるか」を確かめる。

- テスト期間より前のデータだけで学習（未来のデータは使わない）
- オッズは取得済みのもの（締切後の確定オッズ）を使う。実際は締切前に買うので、結果はやや楽観的
- しきい値の調整期間と評価期間を分ける（同じ期間で調整・評価すると数字が良く見えすぎる）
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import features
from .ev import analyze_race
from .model import Model

TYPE_KEY = {"3連単": "3t", "2連単": "2t"}


def _odds_by_race(con, race_ids) -> dict:
    out: dict = {}
    q = "SELECT race_id, bet_type, combo, odds FROM odds WHERE race_id IN (%s)"
    ids = list(race_ids)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in con.execute(q % ",".join("?" * len(chunk)), chunk):
            kind = TYPE_KEY.get(r["bet_type"])
            if kind:
                out.setdefault(r["race_id"], {}).setdefault(kind, {})[r["combo"]] = r["odds"]
    return out


def _payouts(con) -> dict:
    return {(r["race_id"], r["bet_type"], r["combo"]): r["payout"]
            for r in con.execute("SELECT race_id, bet_type, combo, payout FROM payouts")}


_PROB_CACHE: dict = {}


def simulate(races, odds, payouts, lam2, lam3, **params):
    from .model import order_probs
    rows = []
    for rid, p in races:
        if rid not in odds:
            continue
        key = (rid, lam2, lam3)
        if key not in _PROB_CACHE:
            _PROB_CACHE[key] = order_probs(np.asarray(p), lam2, lam3)
        res = analyze_race(p, odds[rid], lam2, lam3, bankroll=100_000, probs=_PROB_CACHE[key], **params)
        for b in res["picks"]:
            pay = payouts.get((rid, b["type"], b["combo"]), 0)
            stake = 100  # 評価は均等買い（100円）で行い、資金配分の影響と分ける
            rows.append({"race_id": rid, "type": b["type"], "combo": b["combo"], "ev": b["ev"],
                         "prob": b["prob"], "odds": b["odds"], "stake": stake,
                         "ret": pay * stake / 100, "hit": pay > 0})
    return pd.DataFrame(rows)


def summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"bets": 0}
    return {"races": int(bets["race_id"].nunique()), "bets": len(bets),
            "hit_rate": round(bets["hit"].mean(), 4),
            "roi": round(bets["ret"].sum() / bets["stake"].sum(), 4),
            "profit": int(bets["ret"].sum() - bets["stake"].sum())}


def run(con, test_from: str, tune_share: float = 0.5) -> dict:
    df = features.full_races(features.build(features.load(con)))
    train, test = df[df["date"] < test_from], df[df["date"] >= test_from]
    if train["race_id"].nunique() < 200 or test.empty:
        raise SystemExit("データ不足：学習200レース以上と、テスト期間のデータが必要です")
    model = Model().fit(train)
    test = test.assign(p=model.win_probs(test))
    races = [(rid, g.sort_values("boat")["p"].values) for rid, g in test.groupby("race_id")]
    odds, pays = _odds_by_race(con, test["race_id"].unique()), _payouts(con)

    # 前半で しきい値・市場混合比 を選び、後半で評価
    dates = sorted({r[0][:8] for r in races})
    cut = dates[int(len(dates) * tune_share)] if len(dates) > 1 else dates[0]
    tune = [r for r in races if r[0][:8] < cut]
    hold = [r for r in races if r[0][:8] >= cut]
    grid = []
    for ev_th, blend, max_odds in itertools.product([1.0, 1.1, 1.2, 1.3, 1.5], [0.3, 0.5, 0.7, 1.0], [50, 100, 200]):
        s = summarize(simulate(tune, odds, pays, model.lam2, model.lam3,
                               ev_th=ev_th, blend=blend, max_odds=max_odds))
        if s.get("bets", 0) >= 30:
            grid.append({"ev_th": ev_th, "blend": blend, "max_odds": max_odds} | s)
    if not grid:
        return {"model": model.metrics, "note": "オッズ付きのレースが少なく、しきい値を調整できません"}
    best = max(grid, key=lambda g: g["roi"])
    params = {k: best[k] for k in ("ev_th", "blend", "max_odds")}
    hold_bets = simulate(hold, odds, pays, model.lam2, model.lam3, **params)

    by_ev = {}
    if not hold_bets.empty:
        hold_bets["ev_bin"] = pd.cut(hold_bets["ev"], [1, 1.2, 1.5, 2, 3, np.inf])
        by_ev = {str(k): summarize(v) for k, v in hold_bets.groupby("ev_bin", observed=True)}
    return {"model": model.metrics, "best_params": params, "tune": best,
            "holdout": summarize(hold_bets), "holdout_by_ev": by_ev,
            "periods": {"test_from": test_from, "holdout_from": cut}}

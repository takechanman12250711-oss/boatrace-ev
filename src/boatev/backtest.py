"""バックテスト：過去期間で「本当に回収率が100%を超えるか」を確かめる。

- テスト期間より前のデータだけで学習（未来のデータは使わない）
- オッズは取得済みのもの（締切後の確定オッズ）を使う。実際は締切前に買うので、結果はやや楽観的
- しきい値の調整期間と評価期間を分ける（同じ期間で調整・評価すると数字が良く見えすぎる）
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config, features
from .ev import analyze_race
from .model import Model

MIN_TUNE_BETS = 300
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


def simulate(races, odds, payouts, lam2, lam3, roman=False, **params):
    from .model import order_probs
    rows = []
    for rid, p in races:
        if rid not in odds:
            continue
        key = (rid, lam2, lam3)
        if key not in _PROB_CACHE:
            _PROB_CACHE[key] = order_probs(np.asarray(p), lam2, lam3)
        res = analyze_race(p, odds[rid], lam2, lam3, bankroll=100_000, probs=_PROB_CACHE[key], **params)
        for b in res["picks"] + ([res["roman"]] if roman and res["roman"] else []):
            pay = payouts.get((rid, b["type"], b["combo"]), 0)
            stake = 100  # 評価は均等買い（100円）で行い、資金配分の影響と分ける
            rows.append({"race_id": rid, "tag": b.get("tag", "本線"), "type": b["type"], "combo": b["combo"], "ev": b["ev"],
                         "prob": b["prob"], "odds": b["odds"], "stake": stake,
                         "ret": pay * stake / 100, "hit": pay > 0})
    return pd.DataFrame(rows)


def roi_ci(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """レース単位のブートストラップで回収率の95%区間。高配当頼みだと幅が広くなる。"""
    if bets.empty:
        return None
    g = bets.groupby("race_id")[["stake", "ret"]].sum().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s = g[idx]
    rois = s[..., 1].sum(1) / s[..., 0].sum(1)
    return [round(float(np.percentile(rois, 2.5)), 3), round(float(np.percentile(rois, 97.5)), 3)]


def summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"bets": 0}
    hits = bets[bets["hit"]]
    return {"races": int(bets["race_id"].nunique()), "bets": len(bets),
            "hits": int(len(hits)), "avg_hit_odds": round(float(hits["ret"].mean() / 100), 1) if len(hits) else None,
            "roi_ci95": roi_ci(bets),
            "roi_without_top3": round(float((bets["ret"].sum() - hits["ret"].nlargest(3).sum()) / bets["stake"].sum()), 4),
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
        grid.append({"ev_th": ev_th, "blend": blend, "max_odds": max_odds} | s)
    # 少ない買い目でたまたま高く出た条件を選ばないよう、
    # 買い目300点以上の中から「回収率の95%区間の下限」が最も高い条件を選ぶ
    ok = [g for g in grid if g.get("bets", 0) >= MIN_TUNE_BETS and g.get("roi_ci95")]
    if not ok:
        return {"model": model.metrics, "grid": grid, "note": "オッズ付きのレースが少なく、しきい値を調整できません"}
    best = max(ok, key=lambda g: (g["roi_ci95"][0], g["roi"]))
    params = {k: best[k] for k in ("ev_th", "blend", "max_odds")}
    hold_bets = simulate(hold, odds, pays, model.lam2, model.lam3, **params)

    by_ev = {}
    if not hold_bets.empty:
        hold_bets["ev_bin"] = pd.cut(hold_bets["ev"], [1, 1.2, 1.5, 2, 3, np.inf])
        by_ev = {str(k): summarize(v) for k, v in hold_bets.groupby("ev_bin", observed=True)}
    # ロマン枠は条件を調整していないので、テスト期間全体で成績を見る
    roman_bets = simulate(races, odds, pays, model.lam2, model.lam3, roman=True, **params)
    roman_bets = roman_bets[roman_bets["tag"] == "ロマン"] if not roman_bets.empty else roman_bets
    roman_rep = summarize(roman_bets) | {"rule": f"3連単{config.ROMAN_MIN_ODDS:.0f}〜{config.ROMAN_MAX_ODDS:.0f}倍・確率{config.ROMAN_MIN_PROB:.1%}以上・期待値{config.ROMAN_EV}以上で最も期待値の高い1点"}
    return {"model": model.metrics, "best_params": params, "tune": best, "roman": roman_rep,
            "selection": f"買い目{MIN_TUNE_BETS}点以上の条件のうち、回収率95%区間の下限が最大のもの",
            "grid": sorted(grid, key=lambda g: -(g.get("roi") or 0)),
            "holdout": summarize(hold_bets), "holdout_by_ev": by_ev,
            "periods": {"test_from": test_from, "holdout_from": cut}}

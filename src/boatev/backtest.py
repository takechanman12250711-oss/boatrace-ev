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
        for b in res["picks"] + (list(res["romans"].values()) if roman else []):
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


def export_picks(con, test_from: str, holdout_from: str, params: dict) -> dict:
    """バックテストと同じ条件で、テスト期間の買い目をレースごとに書き出す。

    - 学習は test_from より前のデータだけ（run と同じ。random_state 固定なので同じモデルになる）
    - 条件（ev_th / blend / max_odds）は最終バックテストで選ばれたものを固定で使う
    - チャッピーの予想と突き合わせて「展開で絞る」効果を確かめるための材料。予想や設定は変えない
    """
    df = features.full_races(features.build(features.load(con)))
    train, test = df[df["date"] < test_from], df[df["date"] >= test_from]
    if train["race_id"].nunique() < 200 or test.empty:
        raise SystemExit("データ不足：学習200レース以上と、テスト期間のデータが必要です")
    model = Model().fit(train)
    test = test.assign(p=model.win_probs(test))
    races = [(rid, g.sort_values("boat")["p"].values) for rid, g in test.groupby("race_id")]
    odds, pays = _odds_by_race(con, test["race_id"].unique()), _payouts(con)
    tri = {rid: combo for (rid, kind, combo), _ in pays.items() if kind == "3連単"}
    bets = simulate(races, odds, pays, model.lam2, model.lam3, **params)
    by_race: dict = {}
    if not bets.empty:
        for r in bets.to_dict("records"):
            by_race.setdefault(r["race_id"], []).append({
                "type": r["type"], "combo": r["combo"], "ev": round(float(r["ev"]), 4),
                "prob": round(float(r["prob"]), 5), "odds": r["odds"],
                "hit": bool(r["hit"]), "ret": int(r["ret"])})
    out = []
    for rid, p in races:
        if rid not in odds:
            continue
        out.append({"race_id": rid, "date": rid[:8], "holdout": rid[:8] >= holdout_from,
                    "win_prob": [round(float(x), 4) for x in p],
                    "trifecta": tri.get(rid), "picks": by_race.get(rid, [])})
    hold = bets[bets["race_id"].str[:8] >= holdout_from] if not bets.empty else bets
    return {"version": "backtest-picks-v1", "test_from": test_from, "holdout_from": holdout_from,
            "params": params, "stake": 100, "model": model.metrics,
            "summary": {"all": summarize(bets), "holdout": summarize(hold)},
            "races": out}


BUCKETS = (0, 10, 30, 100, 1e9)


def _bucket(o):
    return next(i for i in range(4) if BUCKETS[i] <= o < BUCKETS[i + 1])


def drift_samples():
    """試運転の記録から「締切前オッズ ÷ 確定オッズ」の実例を、オッズ帯ごとに集める（勝った組み合わせのみ）。"""
    import glob, json, os
    by_final, by_pre = [[] for _ in range(4)], [[] for _ in range(4)]
    for f in sorted(glob.glob(str(config.SITE_DATA_DIR / "odds_pre" / "*.json"))):
        date = os.path.basename(f)[:8]
        day = config.SITE_DATA_DIR / f"{date}.json"
        if not day.exists():
            continue
        pre, d = json.loads(open(f).read()), json.loads(day.read_text())
        for rid, x in pre.items():
            z = d["races"].get(rid, {}).get("result")
            if not z or not z.get("3連単"):
                continue
            pay = next((q["payout"] for q in z["payouts"] if q["type"] == "3連単"), None)
            o = x["3t"].get(z["3連単"])
            if o and pay:
                fin = pay / 100
                by_final[_bucket(fin)].append(o / fin)
                by_pre[_bucket(o)].append(o / fin)
    return by_final, by_pre


def drift_sim(races, odds, pays, model, params, ths=(1.5, 2.0, 2.5, 3.0)):
    """過去3か月の確定オッズに、試運転で測った「締切前のずれ」をかけて買い目を選び直し、確定オッズで払い戻す。
    scale=1.0 は約23分前のずれ、0.5 はその半分（締切数分前の想定）。補正ありは、締切前オッズをオッズ帯ごとの中央値で割り引く。"""
    by_final, by_pre = drift_samples()
    n = sum(len(x) for x in by_final)
    if n < 200:
        return {"note": f"ずれの実例が少ない（{n}件）"}
    out = {"samples": n}
    for scale in (1.0, 0.5):
        meds = [float(np.median([1 + (r - 1) * scale for r in b])) if b else 1.0 for b in by_pre]
        for corr in (False, True):
            rng = np.random.default_rng(0)
            pre_odds = {}
            for rid, od in odds.items():
                t3 = {}
                for c, fo in (od.get("3t") or {}).items():
                    if not fo:
                        continue
                    pool = by_final[_bucket(fo)] or [1.0]
                    po = fo * (1 + (pool[rng.integers(len(pool))] - 1) * scale)
                    t3[c] = po / meds[_bucket(po)] if corr else po
                pre_odds[rid] = {"3t": t3}
            for th in ths:
                bets = simulate(races, pre_odds, pays, model.lam2, model.lam3, **(params | {"ev_th": th}))
                key = f"ずれ{'23分前' if scale == 1.0 else '半分'}_{'補正あり' if corr else '補正なし'}_期待値{th}"
                out[key] = summarize(bets)
    out["median_by_pre_bucket"] = {f"{BUCKETS[i]}-{BUCKETS[i+1]:.0f}倍": round(float(np.median(b)), 3) if b else None
                                   for i, b in enumerate(by_pre)}
    return out


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
    # ロマン枠は条件を調整していないので、テスト期間全体で枠ごとに成績を見る
    allb = simulate(races, odds, pays, model.lam2, model.lam3, roman=True, **params)
    dl = {r["race_id"]: r["deadline"] or "99:99" for r in con.execute(
        "SELECT race_id, deadline FROM races WHERE date >= ?", (test_from,))}
    n_days = len({rid[:8] for rid, _ in races if rid in odds})
    roman_rep, roman_daily = {}, {}
    for name, (lo, hi) in config.ROMAN_TIERS.items():
        rb = allb[allb["tag"] == name] if not allb.empty else allb
        roman_rep[name] = summarize(rb) | {"rule": f"3連単{lo}〜{hi}倍・毎レース"}
        roman_daily[name] = {}
        if rb.empty:
            continue
        rb = rb.assign(date=rb["race_id"].str[:8], dl=rb["race_id"].map(dl).fillna("99:99"))
        # 1日1点版：期待値がしきい値以上の候補のうち、その日最初に締切を迎えるレースの1点だけ
        for th in (1.3, 1.5, 2.0, 2.5, 3.0, 4.0):
            day = rb[rb["ev"] >= th].sort_values(["date", "dl", "race_id"]).groupby("date").head(config.ROMAN_PER_DAY)
            roman_daily[name][str(th)] = summarize(day) | {"days_with_pick": int(day["date"].nunique()), "days": n_days}
    # 的中重視枠：確率しきい値ごとに、その日締切の早い順で HONMEI_PER_DAY 点まで
    hb = allb[allb["tag"] == config.HONMEI_NAME] if not allb.empty else allb
    honmei = {}
    if not hb.empty:
        hb = hb.assign(date=hb["race_id"].str[:8], dl=hb["race_id"].map(dl).fillna("99:99"))
        for th in (0.10, 0.12, 0.15, 0.20, 0.25):
            for per in (3, 5):
                day = hb[hb["prob"] >= th].sort_values(["date", "dl", "race_id"]).groupby("date").head(per)
                s_ = summarize(day)
                honmei[f"確率{th}以上_1日{per}点"] = s_ | {
                    "days": n_days, "hits_per_day": round(s_.get("hits", 0) / max(n_days, 1), 2),
                    "days_with_hit": int(day[day["hit"]]["date"].nunique()) if not day.empty else 0}
    return {"model": model.metrics, "best_params": params, "tune": best, "roman": roman_rep, "honmei": honmei,
            "roman_daily": roman_daily,
            "drift_sim": drift_sim(races, odds, pays, model, params),
            "selection": f"買い目{MIN_TUNE_BETS}点以上の条件のうち、回収率95%区間の下限が最大のもの",
            "grid": sorted(grid, key=lambda g: -(g.get("roi") or 0)),
            "holdout": summarize(hold_bets), "holdout_by_ev": by_ev,
            "periods": {"test_from": test_from, "holdout_from": cut}}

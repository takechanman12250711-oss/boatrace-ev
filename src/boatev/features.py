"""特徴量。過去成績から作る値は必ず「そのレースより前」のデータだけで計算する（リーク防止）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NUM = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
POINTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}   # 着順→得点（近況の強さ）
ROLL = 30  # 選手の近況は直近30走

FEATURES = [
    "boat", "course_f", "class_num", "age", "weight",
    "nat_win", "nat_2", "loc_win", "loc_2", "motor_2", "boat_2",
    "ex_time", "ex_time_diff", "ex_time_rank",
    "nat_win_diff", "motor_2_rank",
    "r_pts", "r_st", "r_course_win", "r_course_n", "r_n",
    "venue_course_win", "wind_speed", "wave", "course_moved",
    # 日和型：選手×コースの決まり手傾向（公式の競走成績から自前で集計）
    "r_nige", "r_nogashi", "r_sashi", "r_makuri", "r_makurisashi", "r_sasare", "r_makurare",
    "r_course_st", "st_gap_inner", "in1_sasare", "in1_makurare", "out1_makuri",
    "venue_manshu",
]

# 全国平均に近い事前値（少数走の選手を平均へ寄せるため）
STYLE_PRIOR = {
    "nige":        {1: .55},
    "nogashi":     {2: .55},
    "sashi":       {2: .10, 3: .06, 4: .05, 5: .03, 6: .02},
    "makuri":      {2: .03, 3: .05, 4: .05, 5: .02, 6: .01},
    "makurisashi": {3: .04, 4: .04, 5: .03, 6: .02},
    "sasare":      {1: .15},
    "makurare":    {1: .15},
}
STYLE_K = 10


def load(con, date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    q = """SELECT e.*, r.date, r.jcd, r.rno, r.wind_speed, r.wave, r.distance, r.grade, r.kimarite,
                  (SELECT p.payout FROM payouts p WHERE p.race_id = e.race_id AND p.bet_type = '3連単' LIMIT 1) AS tri_payout
           FROM entries e JOIN races r USING(race_id)"""
    cond, args = [], []
    if date_from:
        cond.append("r.date >= ?"); args.append(date_from)
    if date_to:
        cond.append("r.date <= ?"); args.append(date_to)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    return pd.read_sql_query(q, con, params=args)


def _racer_history(df: pd.DataFrame) -> pd.DataFrame:
    """選手ごとの近況（直近ROLL走の平均得点・平均ST、コース別1着率）。"""
    df = df.sort_values(["date", "rno"]).copy()
    done = df["finish"].notna()
    df["_pts"] = df["finish"].map(POINTS).where(done)
    df["_win"] = (df["finish"] == 1).astype(float).where(done)
    g = df.groupby("racer_id", sort=False)
    df["r_pts"] = g["_pts"].transform(lambda s: s.shift().rolling(ROLL, min_periods=3).mean())
    df["r_st"] = g["st"].transform(lambda s: s.abs().shift().rolling(ROLL, min_periods=3).mean())
    df["r_n"] = g["_pts"].transform(lambda s: s.shift().notna().cumsum())

    gc = df.groupby(["racer_id", "course_f"], sort=False)["_win"]
    wins = gc.transform(lambda s: s.shift().fillna(0).cumsum())
    n = gc.transform(lambda s: s.shift().notna().cumsum())
    prior = df["course_f"].map({1: 0.5, 2: 0.15, 3: 0.12, 4: 0.1, 5: 0.06, 6: 0.04}).fillna(0.15)
    df["r_course_win"] = (wins + 5 * prior) / (n + 5)   # 少数走は全体平均に寄せる
    df["r_course_n"] = n
    return df


def _prior_only(df: pd.DataFrame, flag: pd.Series, keys: list[str]):
    """keysごとに、そのレースより前の (該当回数, 対象回数) を返す。flagがNaNの行は数えない。"""
    f, n = flag.fillna(0).astype(float), flag.notna().astype(float)
    cs = pd.DataFrame({"f": f, "n": n, **{k: df[k] for k in keys}}).groupby(keys, sort=False)[["f", "n"]].cumsum()
    return cs["f"] - f, cs["n"] - n


def _style_rates(df: pd.DataFrame) -> pd.DataFrame:
    """選手×コースの 逃げ・逃し・差し・まくり・まくり差し・差され・まくられ率、コース別平均ST。
    df は日付・レース番号順に並んでいること。"""
    done = df["finish"].notna() & df["kimarite"].notna()
    win = df["finish"] == 1
    # レースの勝者のコース
    win_course = df["course_f"].where(win).groupby(df["race_id"]).transform("max")
    kim = df["kimarite"]
    c = df["course_f"]
    flags = {
        "nige":        (win & (c == 1)),
        "nogashi":     (win_course == 1),                       # 2コースから見て1コースが逃げたか
        "sashi":       (win & (kim == "差し")),
        "makuri":      (win & (kim == "まくり")),
        "makurisashi": (win & (kim == "まくり差し")),
        "sasare":      ((c == 1) & (kim == "差し") & ~win),
        "makurare":    ((c == 1) & kim.isin(["まくり", "まくり差し"]) & ~win),
    }
    for name, flag in flags.items():
        prior_map = STYLE_PRIOR[name]
        applies = c.isin(list(prior_map))
        num, cnt = _prior_only(df, flag.astype(float).where(done & applies), ["racer_id", "course_f"])
        prior = c.map(prior_map)
        df[f"r_{name}"] = ((num + STYLE_K * prior) / (cnt + STYLE_K)).where(applies)

    # コース別の平均ST（直近の積み上げ。フライングは絶対値）
    st_num, st_cnt = _prior_only(df, df["st"].abs(), ["racer_id", "course_f"])
    df["r_course_st"] = ((st_num + 5 * 0.16) / (st_cnt + 5))

    # 同じレースの隣の艇との関係（進入コース順）
    by = df[["race_id", "course_f", "r_course_st", "r_makuri", "r_sasare", "r_makurare"]] \
        .drop_duplicates(["race_id", "course_f"]).set_index(["race_id", "course_f"])

    def neighbor(col, offset):
        key = pd.MultiIndex.from_arrays([df["race_id"], df["course_f"] + offset])
        return by[col].reindex(key).values

    df["st_gap_inner"] = df["r_course_st"].values - neighbor("r_course_st", -1)   # 内の艇よりSTが遅いと+
    df["out1_makuri"] = neighbor("r_makuri", +1)                                   # 外の艇のまくり傾向
    in1 = pd.MultiIndex.from_arrays([df["race_id"], np.ones(len(df), int)])
    df["in1_sasare"] = by["r_sasare"].reindex(in1).values                          # 1コース艇の差され率
    df["in1_makurare"] = by["r_makurare"].reindex(in1).values                      # 1コース艇のまくられ率
    return df


def _venue_manshu(df: pd.DataFrame) -> pd.Series:
    """場の万舟率（3連単1万円以上）。前日までの累積。"""
    races = df.drop_duplicates("race_id")[["race_id", "jcd", "date", "tri_payout"]]
    races = races[races["tri_payout"].notna()]
    daily = (races.assign(m=(races["tri_payout"] >= 10000).astype(int))
             .groupby(["jcd", "date"])["m"].agg(["sum", "count"]).reset_index().sort_values("date"))
    g = daily.groupby("jcd")
    daily["cm"], daily["cn"] = g["sum"].cumsum(), g["count"].cumsum()
    # 当日を含めないよう、各日付より前の最新値を使う
    out = pd.Series(np.nan, index=df.index)
    for jcd, d in daily.groupby("jcd"):
        idx = df.index[df["jcd"] == jcd]
        pos = np.searchsorted(d["date"].values, df.loc[idx, "date"].values, side="left") - 1
        cm = np.where(pos >= 0, d["cm"].values[np.clip(pos, 0, None)], 0)
        cn = np.where(pos >= 0, d["cn"].values[np.clip(pos, 0, None)], 0)
        out.loc[idx] = (cm + 100 * 0.18) / (cn + 100)
    return out.fillna(0.18)


def _venue_course(df: pd.DataFrame) -> pd.Series:
    """場×コースの1着率（前日までの累積）。"""
    done = df[df["finish"].notna()]
    daily = (done.assign(w=(done["finish"] == 1).astype(int))
             .groupby(["jcd", "course_f", "date"])["w"].agg(["sum", "count"]).reset_index()
             .sort_values("date"))
    gg = daily.groupby(["jcd", "course_f"])
    daily["cw"] = gg["sum"].cumsum() - daily["sum"]
    daily["cn"] = gg["count"].cumsum() - daily["count"]
    m = df[["jcd", "course_f", "date"]].merge(daily[["jcd", "course_f", "date", "cw", "cn"]],
                                               on=["jcd", "course_f", "date"], how="left")
    # その日の集計が無い行（当日予測など）は最新の累積を使う
    last = daily.assign(cw=daily["cw"] + daily["sum"], cn=daily["cn"] + daily["count"]) \
        .groupby(["jcd", "course_f"])[["cw", "cn"]].last()
    miss = m["cw"].isna()
    if miss.any():
        fill = m.loc[miss, ["jcd", "course_f"]].merge(last, left_on=["jcd", "course_f"],
                                                      right_index=True, how="left")
        m.loc[miss, ["cw", "cn"]] = fill[["cw", "cn"]].values
    prior = df["course_f"].map({1: 0.5, 2: 0.15, 3: 0.12, 4: 0.1, 5: 0.06, 6: 0.04}).fillna(0.15).values
    return pd.Series((m["cw"].fillna(0).values + 50 * prior) / (m["cn"].fillna(0).values + 50), index=df.index)


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 進入コース：直前のスタート展示 → 無ければ実際の進入(過去データ) → 無ければ枠なり
    df["course_f"] = df["ex_course"].fillna(df["course"]).fillna(df["boat"]).astype(int)
    df["course_moved"] = (df["course_f"] != df["boat"]).astype(int)
    df["class_num"] = df["class"].map(CLASS_NUM)
    if "kimarite" not in df:
        df["kimarite"] = None
    if "tri_payout" not in df:
        df["tri_payout"] = np.nan
    df = _racer_history(df)
    df = _style_rates(df)
    df["venue_course_win"] = _venue_course(df)
    df["venue_manshu"] = _venue_manshu(df)

    g = df.groupby("race_id")
    df["ex_time_diff"] = df["ex_time"] - g["ex_time"].transform("mean")
    df["ex_time_rank"] = g["ex_time"].rank(method="min")
    df["nat_win_diff"] = df["nat_win"] - g["nat_win"].transform("mean")
    df["motor_2_rank"] = g["motor_2"].rank(ascending=False, method="min")
    for c in FEATURES:
        if c not in df:
            df[c] = np.nan
    return df.sort_values(["race_id", "boat"]).reset_index(drop=True)


def full_races(df: pd.DataFrame) -> pd.DataFrame:
    """6艇そろい、1〜3着が確定しているレースだけ（学習・検証用）。"""
    ok = df.groupby("race_id").apply(
        lambda x: len(x) == 6 and {1, 2, 3} <= set(x["finish"].dropna().astype(int)),
        include_groups=False,
    )
    return df[df["race_id"].isin(ok[ok].index)]

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
]


def load(con, date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    q = """SELECT e.*, r.date, r.jcd, r.rno, r.wind_speed, r.wave, r.distance, r.grade
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
    df = _racer_history(df)
    df["venue_course_win"] = _venue_course(df)

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

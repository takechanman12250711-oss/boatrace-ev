"""テスト用の架空データ生成（実データなしでパイプライン全体を通す）。"""
import itertools
import numpy as np
from boatev import db

rng = np.random.default_rng(0)
COURSE_BONUS = np.array([1.6, 0.4, 0.3, 0.1, -0.3, -0.6])


def make(con, days=120, venues=("01", "12", "24")):
    racers = {rid: {"skill": rng.normal(0, 0.6), "cls": None} for rid in range(4000, 4300)}
    for rid, r in racers.items():
        r["cls"] = "A1" if r["skill"] > .7 else "A2" if r["skill"] > .2 else "B1" if r["skill"] > -.6 else "B2"
    ids = list(racers)
    motors = {(v, m): rng.normal(0, 0.4) for v in venues for m in range(1, 61)}
    start = np.datetime64("2026-01-01")
    for d in range(days):
        date = str(start + d).replace("-", "")
        for v in venues:
            for rno in range(1, 13):
                rid = db.race_id(date, v, rno)
                db.upsert(con, "races", [{"race_id": rid, "date": date, "jcd": v, "rno": rno, "distance": 1800,
                                          "wind_speed": float(rng.integers(0, 7)), "wave": float(rng.integers(0, 8))}], ("race_id",))
                picks = rng.choice(ids, 6, replace=False)
                ms = rng.choice(60, 6, replace=False) + 1
                ex = rng.normal(6.75, 0.06, 6)
                score = np.array([racers[p]["skill"] + motors[(v, m)] for p, m in zip(picks, ms)]) + COURSE_BONUS - (ex - 6.75) * 5
                order = np.argsort(-(score + rng.gumbel(size=6)))
                fin = np.empty(6, int); fin[order] = np.arange(1, 7)
                ents = []
                for b in range(6):
                    sk = racers[picks[b]]["skill"]
                    ents.append({"race_id": rid, "boat": b + 1, "racer_id": int(picks[b]), "name": f"選手{picks[b]}",
                                 "age": 30, "weight": 52.0, "class": racers[picks[b]]["cls"],
                                 "nat_win": round(5.5 + sk * 1.5 + rng.normal(0, .3), 2),
                                 "nat_2": round(35 + sk * 12, 2), "loc_win": round(5.5 + sk * 1.5 + rng.normal(0, .8), 2),
                                 "loc_2": round(35 + sk * 12 + rng.normal(0, 8), 2), "motor_no": int(ms[b]),
                                 "motor_2": round(33 + motors[(v, ms[b])] * 15 + rng.normal(0, 4), 2), "boat_no": 1, "boat_2": 33.0,
                                 "ex_time": round(float(ex[b]), 2), "course": b + 1, "st": round(abs(rng.normal(.15, .04)), 2),
                                 "finish": int(fin[b])})
                db.upsert(con, "entries", ents, ("race_id", "boat"))
                wc = int(np.where(fin == 1)[0][0]) + 1
                kim = "逃げ" if wc == 1 else str(rng.choice(["差し", "まくり", "まくり差し"]))
                db.upsert(con, "races", [{"race_id": rid, "kimarite": kim}], ("race_id",))
                # 市場：真の強さ＋ノイズから作ったオッズ（控除率25%）
                mk = np.exp(score + rng.normal(0, .5, 6)); mk /= mk.sum()
                for kind, label, n in (("3t", "3連単", 3), ("2t", "2連単", 2)):
                    combos = list(itertools.permutations(range(6), n))
                    pr = []
                    for c in combos:
                        p, rest = 1.0, mk.copy()
                        for i in c:
                            p *= rest[i] / rest.sum(); rest[i] = 0
                        pr.append(p)
                    pr = np.array(pr)
                    o = np.round(0.75 / pr, 1)
                    db.upsert(con, "odds", ({"race_id": rid, "bet_type": label, "combo": "-".join(str(i + 1) for i in c),
                                             "odds": float(x)} for c, x in zip(combos, o)), ("race_id", "bet_type", "combo"))
                    win = tuple(int(np.where(fin == k)[0][0]) for k in range(1, n + 1))
                    db.upsert(con, "payouts", [{"race_id": rid, "bet_type": label, "combo": "-".join(str(i + 1) for i in win),
                                                "payout": int(o[combos.index(win)] * 100)}], ("race_id", "bet_type", "combo"))
        con.commit()

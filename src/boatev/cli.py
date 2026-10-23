"""コマンド一覧

  python -m boatev.cli history  --from 20250101 --to 20261022   過去のB/Kを一括取込（学習データ）
  python -m boatev.cli collect  --date 20261022                 公式サイトから出走表・直前・オッズ・結果（バックテスト用オッズの蓄積）
  python -m boatev.cli backfill-odds --from 20220101 --to 20261022  過去オッズだけ補完（途中再開可）
  python -m boatev.cli train                                    モデル学習
  python -m boatev.cli backtest --test-from 20260701             回収率の検証としきい値調整
  python -m boatev.cli predict  [--date 今日] [--within 20]      締切N分以内のレースを予想 → docs/data/
  python -m boatev.cli results  [--date 今日]                    結果を取り込み、予想JSONに的中を反映
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from . import config, db, features, history, scrape

JST = dt.timezone(dt.timedelta(hours=9))


def now():
    return dt.datetime.now(JST)


def daterange(a: str, b: str):
    d, end = dt.datetime.strptime(a, "%Y%m%d"), dt.datetime.strptime(b, "%Y%m%d")
    while d <= end:
        yield d.strftime("%Y%m%d")
        d += dt.timedelta(days=1)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 取込
def cmd_history(args):
    with db.session() as con:
        for date in daterange(args.date_from, args.date_to):
            try:
                races, entries, pays = history.fetch_day(date)
            except Exception as e:  # 1日失敗しても続ける
                log(date, "失敗:", e)
                continue
            db.upsert(con, "races", races, ("race_id",))
            db.upsert(con, "entries", entries, ("race_id", "boat"))
            db.upsert(con, "payouts", pays, ("race_id", "bet_type", "combo"))
            con.commit()
            log(date, f"races={len({r['race_id'] for r in races})} entries={len(entries)}")
        miss = config.DATA_DIR / "parse_misses.json"
        if history.MISSES:
            miss.write_text(json.dumps(history.MISSES, ensure_ascii=False, indent=1))
        elif miss.exists():
            miss.unlink()


def store_odds(con, rid, date, jcd, rno):
    ts = now().isoformat(timespec="seconds")
    got = {}
    for kind, label, fn in (("3t", "3連単", scrape.odds_3t), ("2t", "2連単", scrape.odds_2t)):
        o = fn(date, jcd, rno)
        got[kind] = o
        db.upsert(con, "odds", ({"race_id": rid, "bet_type": label, "combo": c, "odds": v, "fetched_at": ts}
                                for c, v in o.items()), ("race_id", "bet_type", "combo"))
    return got


def collect_race(con, date, jcd, rno, *, before=True, odds=True, result=False):
    rid = db.race_id(date, jcd, rno)
    race, entries = scrape.racelist(date, jcd, rno)
    if not entries:
        return None
    db.upsert(con, "races", [race], ("race_id",))
    db.upsert(con, "entries", entries, ("race_id", "boat"))
    got = {}
    if before:
        w, ex = scrape.beforeinfo(date, jcd, rno)
        db.upsert(con, "races", [w], ("race_id",))
        db.upsert(con, "entries", ex, ("race_id", "boat"))
    if odds:
        got = store_odds(con, rid, date, jcd, rno)
    if result:
        res, pays = scrape.result(date, jcd, rno)
        db.upsert(con, "entries", res, ("race_id", "boat"))
        db.upsert(con, "payouts", pays, ("race_id", "bet_type", "combo"))
    con.commit()
    return race | {"_odds": got}


def cmd_collect(args):
    with db.session() as con:
        for date in daterange(args.date, args.to or args.date):
            for jcd in scrape.venues_of_day(date):
                for rno in range(1, 13):
                    collect_race(con, date, jcd, rno, result=True)
                log(date, config.VENUES[jcd], "done")


def cmd_backfill_odds(args):
    """過去のオッズだけを補完。レース一覧は取込済みのB/Kから作るので、1レース2回のアクセスで済む。
    新しい日付から順に取り、--max-minutes で打ち切る（Actionsの制限時間対策）。途中再開可。"""
    import time

    t0 = time.monotonic()
    with db.session() as con:
        rows = con.execute(
            """SELECT r.race_id, r.date, r.jcd, r.rno FROM races r
               WHERE r.date BETWEEN ? AND ?
                 AND NOT EXISTS (SELECT 1 FROM odds o WHERE o.race_id = r.race_id AND o.bet_type='3連単')
               ORDER BY r.date DESC, r.jcd, r.rno""", (args.date_from, args.date_to)).fetchall()
        log(f"未取得 {len(rows)} レース")
        for n, r in enumerate(rows, 1):
            if args.max_minutes and time.monotonic() - t0 > args.max_minutes * 60:
                log(f"時間切れで中断（{n - 1} レース取得）。再実行すると続きから取ります")
                break
            store_odds(con, r["race_id"], r["date"], r["jcd"], r["rno"])
            if n % 50 == 0:
                con.commit()
                log(r["date"], f"{n}/{len(rows)}")


# ---------------------------------------------------------------- 学習・検証
def cmd_train(args):
    from .model import Model

    with db.session() as con:
        df = features.full_races(features.build(features.load(con, args.date_from)))
    log(f"学習データ: {df['race_id'].nunique()} レース")
    m = Model().fit(df)
    m.save()
    print(json.dumps(m.metrics, ensure_ascii=False, indent=2))


def cmd_backtest(args):
    from . import backtest

    with db.session() as con:
        rep = backtest.run(con, args.test_from)
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    (config.SITE_DATA_DIR).mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "backtest.json").write_text(json.dumps(rep, ensure_ascii=False, default=str))


# ---------------------------------------------------------------- 予想
def _site_file(date):
    return config.SITE_DATA_DIR / f"{date}.json"


def _load_site(date):
    p = _site_file(date)
    return json.loads(p.read_text()) if p.exists() else {"date": date, "races": {}}


def _save_site(data):
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now().isoformat(timespec="seconds")
    _site_file(data["date"]).write_text(json.dumps(data, ensure_ascii=False))
    idx_path = config.SITE_DATA_DIR / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"dates": []}
    idx["dates"] = sorted(set(idx["dates"]) | {data["date"]}, reverse=True)[:60]
    idx_path.write_text(json.dumps(idx))


def predict_races(con, model, targets, date):
    """targets: [(jcd, rno, odds)] → レースごとの分析結果"""
    from .ev import analyze_race

    start = (dt.datetime.strptime(date, "%Y%m%d") - dt.timedelta(days=400)).strftime("%Y%m%d")
    df = features.build(features.load(con, start, date))
    out = {}
    for jcd, rno, odds in targets:
        rid = db.race_id(date, jcd, rno)
        g = df[df["race_id"] == rid].sort_values("boat")
        if len(g) != 6:
            continue
        p = model.win_probs(g)
        res = analyze_race(p, odds, model.lam2, model.lam3)
        top = sorted((r for r in res["table"] if r["type"] == "3連単"), key=lambda r: -r["prob"])[:10]
        out[rid] = {
            "race_id": rid, "jcd": jcd, "venue": config.VENUES[jcd], "rno": rno,
            "entries": [{"boat": int(r["boat"]), "name": r["name"], "class": r["class"],
                         "course": int(r["course_f"]),
                         "ex_time": None if r["ex_time"] != r["ex_time"] else float(r["ex_time"]),
                         "win_prob": round(float(pi), 4)} for (_, r), pi in zip(g.iterrows(), p)],
            "picks": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()} for b in res["picks"]],
            "top3t": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()} for b in top],
            "predicted_at": now().isoformat(timespec="seconds"),
        }
    return out


def cmd_predict(args):
    from .model import Model

    date = args.date or now().strftime("%Y%m%d")
    model = Model.load()
    site = _load_site(date)
    with db.session() as con:
        targets = []
        for jcd in scrape.venues_of_day(date):
            for rno in range(1, 13):
                rid = db.race_id(date, jcd, rno)
                race = con.execute("SELECT deadline FROM races WHERE race_id=?", (rid,)).fetchone()
                if race is None or race["deadline"] is None:
                    race_info, entries = scrape.racelist(date, jcd, rno)
                    if not entries:
                        continue
                    db.upsert(con, "races", [race_info], ("race_id",))
                    db.upsert(con, "entries", entries, ("race_id", "boat"))
                    deadline = race_info.get("deadline")
                else:
                    deadline = race["deadline"]
                if args.within and deadline:
                    h, m = map(int, deadline.split(":"))
                    left = (now().replace(hour=h, minute=m, second=0, microsecond=0) - now()).total_seconds() / 60
                    if not (0 < left <= args.within):
                        continue
                r = collect_race(con, date, jcd, rno, before=True, odds=True)
                if r:
                    targets.append((jcd, rno, r["_odds"]))
                    site["races"].setdefault(rid, {})["deadline"] = deadline
        log(f"予想対象 {len(targets)} レース")
        for rid, v in predict_races(con, model, targets, date).items():
            site["races"][rid] = site["races"].get(rid, {}) | v
    _save_site(site)


def cmd_results(args):
    date = args.date or now().strftime("%Y%m%d")
    site = _load_site(date)
    with db.session() as con:
        for rid, r in site["races"].items():
            if "result" in r or "jcd" not in r:
                continue
            ents, pays = scrape.result(date, r["jcd"], r["rno"])
            if not pays:
                continue
            db.upsert(con, "entries", ents, ("race_id", "boat"))
            db.upsert(con, "payouts", pays, ("race_id", "bet_type", "combo"))
            paymap = {(p["bet_type"], p["combo"]): p["payout"] for p in pays}
            stake = sum(b["stake"] for b in r.get("picks", []))
            ret = sum(paymap.get((b["type"], b["combo"]), 0) * b["stake"] / 100 for b in r.get("picks", []))
            r["result"] = {"3連単": next((p["combo"] for p in pays if p["bet_type"] == "3連単"), None),
                           "payouts": [{"type": p["bet_type"], "combo": p["combo"], "payout": p["payout"]} for p in pays],
                           "stake": stake, "return": int(ret)}
            con.commit()
    _save_site(site)


def cmd_status(args):
    """DBの中身の件数と抜けをdocs/data/status.jsonに書く（Actionsのログを見なくても状況がわかるように）。"""
    with db.session() as con:
        q = lambda sql: [dict(r) for r in con.execute(sql)]
        st = {
            "generated_at": now().isoformat(timespec="seconds"),
            "totals": q("""SELECT (SELECT COUNT(*) FROM races) races, (SELECT COUNT(*) FROM entries) entries,
                                  (SELECT COUNT(*) FROM payouts) payouts,
                                  (SELECT COUNT(DISTINCT race_id) FROM odds) races_with_odds,
                                  (SELECT MIN(date) FROM races) first_date, (SELECT MAX(date) FROM races) last_date""")[0],
            # 各列がどれだけ埋まっているか（0%の列は読み取り失敗の疑い）
            "fill_rate": q("""SELECT ROUND(AVG(racer_id IS NOT NULL),3) racer_id, ROUND(AVG(class IS NOT NULL),3) class,
                                     ROUND(AVG(nat_win IS NOT NULL),3) nat_win, ROUND(AVG(motor_2 IS NOT NULL),3) motor_2,
                                     ROUND(AVG(ex_time IS NOT NULL),3) ex_time, ROUND(AVG(course IS NOT NULL),3) course,
                                     ROUND(AVG(st IS NOT NULL),3) st, ROUND(AVG(finish IS NOT NULL),3) finish
                              FROM entries""")[0],
            "recent_days": q("""SELECT r.date, COUNT(DISTINCT r.jcd) venues, COUNT(DISTINCT r.race_id) races,
                                       COUNT(e.boat) entries, SUM(e.finish IS NOT NULL) finished,
                                       (SELECT COUNT(*) FROM payouts p JOIN races r2 USING(race_id) WHERE r2.date=r.date AND p.bet_type='3連単') tri_payouts
                                FROM races r LEFT JOIN entries e USING(race_id)
                                GROUP BY r.date ORDER BY r.date DESC LIMIT 14"""),
            "sample": q("""SELECT e.race_id, e.boat, e.name, e.class, e.nat_win, e.motor_2, e.ex_time, e.course, e.st, e.finish
                           FROM entries e ORDER BY e.race_id DESC, e.boat LIMIT 6"""),
        }
    # 着順・級別が欠けているレース（場ごと）
    with db.session() as con:
        st["gaps"] = [dict(r) for r in con.execute("""
            SELECT r.date, r.jcd, COUNT(*) entries, SUM(e.class IS NULL) no_class, SUM(e.finish IS NULL) no_finish,
                   SUM(e.ex_time IS NULL) no_ex
            FROM entries e JOIN races r USING(race_id) GROUP BY r.date, r.jcd
            HAVING no_class > 0 OR no_finish > 12 OR no_ex > 0 ORDER BY r.date DESC LIMIT 40""")]
    miss = config.DATA_DIR / "parse_misses.json"
    if miss.exists():
        st["parse_misses"] = json.loads(miss.read_text())[:80]
    if config.MODEL_PATH.exists():
        from .model import Model
        st["model"] = Model.load().metrics
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "status.json").write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(st["totals"], ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="boatev")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("history"); p.add_argument("--from", dest="date_from", required=True); p.add_argument("--to", dest="date_to", required=True); p.set_defaults(f=cmd_history)
    p = sp.add_parser("collect"); p.add_argument("--date", required=True); p.add_argument("--to"); p.set_defaults(f=cmd_collect)
    p = sp.add_parser("backfill-odds"); p.add_argument("--from", dest="date_from", required=True); p.add_argument("--to", dest="date_to", required=True); p.add_argument("--max-minutes", type=int, default=0); p.set_defaults(f=cmd_backfill_odds)
    p = sp.add_parser("train"); p.add_argument("--from", dest="date_from"); p.set_defaults(f=cmd_train)
    p = sp.add_parser("backtest"); p.add_argument("--test-from", required=True); p.set_defaults(f=cmd_backtest)
    p = sp.add_parser("predict"); p.add_argument("--date"); p.add_argument("--within", type=int, default=0); p.set_defaults(f=cmd_predict)
    p = sp.add_parser("results"); p.add_argument("--date"); p.set_defaults(f=cmd_results)
    p = sp.add_parser("status"); p.set_defaults(f=cmd_status)
    p = sp.add_parser("debug-page"); p.add_argument("url"); p.set_defaults(f=lambda a: print(__import__("boatev.http", fromlist=["get"]).get(a.url)[:5000]))
    args = ap.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()

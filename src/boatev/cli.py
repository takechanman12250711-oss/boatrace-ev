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
def _history_log(date, sec, **kw):
    """取込1日ごとの所要時間・通信状況を記録（status.jsonで確認する）。"""
    from . import http
    p = config.DATA_DIR / "history_log.json"
    try:
        rows = json.loads(p.read_text()) if p.exists() else []
    except ValueError:
        rows = []
    rows.append({"date": date, "sec": round(sec, 1), "at": now().isoformat(timespec="seconds"),
                 **kw, "http": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in http.STATS.items()}})
    p.write_text(json.dumps(rows[-200:], ensure_ascii=False))


def cmd_history(args):
    """取込済み（決まり手まで入っている日）は飛ばすので、途中で止まっても再実行で続きから進む。"""
    import time

    t0 = time.monotonic()
    with db.session() as con:
        done = {r[0] for r in con.execute(
            "SELECT date FROM races GROUP BY date HAVING SUM(kimarite IS NOT NULL) > 0")}
        for date in daterange(args.date_from, args.date_to):
            if date in done and not args.force:
                continue
            if args.max_minutes and time.monotonic() - t0 > args.max_minutes * 60:
                log(f"時間切れで中断（{date} から再開できます）")
                break
            t_day = time.monotonic()
            try:
                races, entries, pays = history.fetch_day(date)
            except Exception as e:  # 1日失敗しても続ける
                log(date, "失敗:", e)
                _history_log(date, time.monotonic() - t_day, error=str(e)[:200])
                continue
            db.upsert(con, "races", races, ("race_id",))
            db.upsert(con, "entries", entries, ("race_id", "boat"))
            db.upsert(con, "payouts", pays, ("race_id", "bet_type", "combo"))
            con.commit()
            log(date, f"races={len({r['race_id'] for r in races})} entries={len(entries)}")
            _history_log(date, time.monotonic() - t_day, races=len({r['race_id'] for r in races}))
        miss = config.DATA_DIR / "parse_misses.json"
        if history.MISSES:
            miss.write_text(json.dumps(history.MISSES, ensure_ascii=False, indent=1))
        elif miss.exists():
            miss.unlink()


def store_odds(con, rid, date, jcd, rno, kinds=("3t", "2t")):
    ts = now().isoformat(timespec="seconds")
    got = {}
    for kind, label, fn in (("3t", "3連単", scrape.odds_3t), ("2t", "2連単", scrape.odds_2t)):
        if kind not in kinds:
            continue
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
        res, pays, kim = scrape.result(date, jcd, rno)
        db.upsert(con, "races", [{"race_id": rid, "kimarite": kim}], ("race_id",))
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
            store_odds(con, r["race_id"], r["date"], r["jcd"], r["rno"], kinds=tuple(args.kinds.split(",")))
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


def _save_report(name, text):
    """docs/data と data/reports（Actionsのキャッシュに残る）の両方に保存。
    長時間ジョブでpushに失敗しても、次の status 実行で docs/data に戻せる。"""
    for d in (config.SITE_DATA_DIR, config.DATA_DIR / "reports"):
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)


def cmd_compare_features(args):
    """日和型の指標を入れる前後で、検証期間の精度を比べる。"""
    from .model import Model
    from .features import FEATURES

    new = FEATURES[FEATURES.index("r_nige"):]
    with db.session() as con:
        df = features.full_races(features.build(features.load(con, args.date_from)))
    out = {}
    for name, feats in (("基本のみ", [f for f in FEATURES if f not in new]), ("日和型を追加", FEATURES)):
        m = Model(feats).fit(df, refit=False)
        out[name] = {k: m.metrics[k] for k in ("win_top1_acc", "win_logloss", "tri_logloss", "tri_top1_acc")}
        log(name, out[name])
    out["races"] = int(df["race_id"].nunique())
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _save_report("feature_compare.json", json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_backtest(args):
    from . import backtest

    with db.session() as con:
        rep = backtest.run(con, args.test_from)
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    (config.SITE_DATA_DIR).mkdir(parents=True, exist_ok=True)
    _save_report("backtest.json", json.dumps(rep, ensure_ascii=False, default=str))


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


def _r4(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} if d else None


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
            "picks": [_r4(b) for b in res["picks"]],
            "roman": _r4(res["roman"]),
            "top3t": [_r4(b) for b in top],
            "predicted_at": now().isoformat(timespec="seconds"),
        }
    return out


def _ensure_program(con, date):
    """当日の番組表（締切時刻・選手・モーター）をDBに入れる。
    公式の番組表ファイルを使えば、公式サイトへのアクセスなしで全レース分そろう。"""
    have = con.execute("SELECT COUNT(*) FROM races WHERE date=? AND deadline IS NOT NULL", (date,)).fetchone()[0]
    if have:
        return
    text = history._download_text("b", date)
    if text:
        races, entries = history.parse_b(text, date)
        db.upsert(con, "races", races, ("race_id",))
        db.upsert(con, "entries", entries, ("race_id", "boat"))
        con.commit()
        log(f"番組表ファイルから {len(races)} レース取込")
        return
    log("番組表ファイルがまだ無いため、公式サイトの出走表から取得")
    for jcd in scrape.venues_of_day(date):
        for rno in range(1, 13):
            race_info, entries = scrape.racelist(date, jcd, rno)
            if entries:
                db.upsert(con, "races", [race_info], ("race_id",))
                db.upsert(con, "entries", entries, ("race_id", "boat"))
    con.commit()


def _minutes_left(deadline):
    h, m = map(int, deadline.split(":"))
    return (now().replace(hour=h, minute=m, second=0, microsecond=0) - now()).total_seconds() / 60


def cmd_predict(args):
    """締切 2〜within 分前のレースを、まだ予想していなければ予想する。
    1レースあたり公式サイトへのアクセスは2回（直前情報・3連単オッズ）だけ。"""
    from .model import Model

    date = args.date or now().strftime("%Y%m%d")
    model = Model.load()
    site = _load_site(date)
    pre_path = config.SITE_DATA_DIR / "odds_pre" / f"{date}.json"
    pre = json.loads(pre_path.read_text()) if pre_path.exists() else {}
    with db.session() as con:
        _ensure_program(con, date)
        rows = con.execute("SELECT race_id, jcd, rno, deadline FROM races WHERE date=? AND deadline IS NOT NULL "
                           "ORDER BY deadline", (date,)).fetchall()
        targets = []
        for r in rows:
            if site["races"].get(r["race_id"], {}).get("predicted_at"):
                continue
            if not (2 < _minutes_left(r["deadline"]) <= args.within):
                continue
            w, ex = scrape.beforeinfo(date, r["jcd"], r["rno"])
            db.upsert(con, "races", [w], ("race_id",))
            db.upsert(con, "entries", ex, ("race_id", "boat"))
            odds = store_odds(con, r["race_id"], date, r["jcd"], r["rno"], kinds=("3t",))
            con.commit()
            if not odds.get("3t"):
                continue
            pre[r["race_id"]] = {"at": now().isoformat(timespec="seconds"), "3t": odds["3t"]}
            targets.append((r["jcd"], r["rno"], odds))
            site["races"].setdefault(r["race_id"], {})["deadline"] = r["deadline"]
        log(f"予想対象 {len(targets)} レース")
        roman_used = sum(1 for x in site["races"].values() if x.get("roman"))
        for rid, v in sorted(predict_races(con, model, targets, date).items(),
                             key=lambda kv: (site["races"][kv[0]].get("deadline") or "", kv[0])):
            # ロマンは「期待値 ROMAN_DAILY_EV 以上」を最初に満たしたレースだけ（1日 ROMAN_PER_DAY 点まで）
            rm = v.get("roman")
            if rm and rm["ev"] >= config.ROMAN_DAILY_EV and roman_used < config.ROMAN_PER_DAY:
                roman_used += 1
            else:
                v["roman"] = None
            site["races"][rid] = site["races"].get(rid, {}) | v
    if targets:
        # 締切前オッズを残しておく（後で「実戦に近い」バックテストに使える）
        pre_path.parent.mkdir(parents=True, exist_ok=True)
        pre_path.write_text(json.dumps(pre, separators=(",", ":")))
    _save_site(site)


def cmd_results(args):
    date = args.date or now().strftime("%Y%m%d")
    site = _load_site(date)
    today = date == now().strftime("%Y%m%d")
    with db.session() as con:
        for rid, r in site["races"].items():
            if "result" in r or "jcd" not in r:
                continue
            if today and r.get("deadline") and _minutes_left(r["deadline"]) > -5:
                continue  # まだ結果が出ていない
            ents, pays, kim = scrape.result(date, r["jcd"], r["rno"])
            if not pays:
                continue
            db.upsert(con, "races", [{"race_id": rid, "kimarite": kim}], ("race_id",))
            db.upsert(con, "entries", ents, ("race_id", "boat"))
            db.upsert(con, "payouts", pays, ("race_id", "bet_type", "combo"))
            paymap = {(p["bet_type"], p["combo"]): p["payout"] for p in pays}
            ret = lambda bets: int(sum(paymap.get((b["type"], b["combo"]), 0) * b["stake"] / 100 for b in bets))
            picks, roman = r.get("picks", []), [r["roman"]] if r.get("roman") else []
            r["result"] = {"3連単": next((p["combo"] for p in pays if p["bet_type"] == "3連単"), None),
                           "payouts": [{"type": p["bet_type"], "combo": p["combo"], "payout": p["payout"]} for p in pays],
                           "stake": sum(b["stake"] for b in picks), "return": ret(picks),
                           "roman_stake": sum(b["stake"] for b in roman), "roman_return": ret(roman)}
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
            "days_loaded": q("""SELECT COUNT(DISTINCT date) days, SUM(k) days_with_kimarite FROM
                                 (SELECT date, MAX(kimarite IS NOT NULL) k FROM races GROUP BY date)""")[0],
            "kimarite": q("""SELECT kimarite, COUNT(*) n FROM races GROUP BY kimarite ORDER BY n DESC"""),
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
    hl = config.DATA_DIR / "history_log.json"
    if hl.exists():
        rows = json.loads(hl.read_text())
        st["history_log_tail"] = rows[-15:]
        secs = [r["sec"] for r in rows[-100:]]
        st["history_sec_per_day_avg"] = round(sum(secs) / len(secs), 1) if secs else None
    miss = config.DATA_DIR / "parse_misses.json"
    if miss.exists():
        st["parse_misses"] = json.loads(miss.read_text())[:80]
    if config.MODEL_PATH.exists():
        from .model import Model
        st["model"] = Model.load().metrics
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rep_dir = config.DATA_DIR / "reports"
    if rep_dir.exists():
        for f in rep_dir.glob("*.json"):
            dst = config.SITE_DATA_DIR / f.name
            if not dst.exists() or dst.stat().st_mtime < f.stat().st_mtime:
                dst.write_text(f.read_text())
    (config.SITE_DATA_DIR / "status.json").write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(st["totals"], ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="boatev")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("history"); p.add_argument("--from", dest="date_from", required=True); p.add_argument("--to", dest="date_to", required=True); p.add_argument("--max-minutes", type=int, default=0); p.add_argument("--force", action="store_true"); p.set_defaults(f=cmd_history)
    p = sp.add_parser("collect"); p.add_argument("--date", required=True); p.add_argument("--to"); p.set_defaults(f=cmd_collect)
    p = sp.add_parser("backfill-odds"); p.add_argument("--from", dest="date_from", required=True); p.add_argument("--to", dest="date_to", required=True); p.add_argument("--max-minutes", type=int, default=0); p.add_argument("--kinds", default="3t", help="3t / 2t / 3t,2t"); p.set_defaults(f=cmd_backfill_odds)
    p = sp.add_parser("train"); p.add_argument("--from", dest="date_from"); p.set_defaults(f=cmd_train)
    p = sp.add_parser("compare-features"); p.add_argument("--from", dest="date_from"); p.set_defaults(f=cmd_compare_features)
    p = sp.add_parser("backtest"); p.add_argument("--test-from", required=True); p.set_defaults(f=cmd_backtest)
    p = sp.add_parser("predict"); p.add_argument("--date"); p.add_argument("--within", type=int, default=0); p.set_defaults(f=cmd_predict)
    p = sp.add_parser("results"); p.add_argument("--date"); p.set_defaults(f=cmd_results)
    p = sp.add_parser("status"); p.set_defaults(f=cmd_status)
    p = sp.add_parser("debug-page"); p.add_argument("url"); p.set_defaults(f=lambda a: print(__import__("boatev.http", fromlist=["get"]).get(a.url)[:5000]))
    args = ap.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()

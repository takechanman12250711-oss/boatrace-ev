"""公式サイト（boatrace.jp）のページを読み取る。

HTML構造は変わることがあるので、取れない項目は None のまま進めて落ちないようにしてある。
構造が変わったら `python -m boatev.cli debug-page <url>` で中身を見て直す。
"""
from __future__ import annotations

import itertools
import re
import unicodedata

from bs4 import BeautifulSoup

from . import config, http
from .db import race_id

BOATS = range(1, 7)


# ---------------------------------------------------------------- helpers
def norm(s) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


def num(s, kind=float):
    s = norm(str(s) if s is not None else "")
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if not m:
        return None
    try:
        return kind(float(m.group())) if kind is int else kind(m.group())
    except ValueError:
        return None


def soup_of(page: str, jcd: str, rno: int, date: str) -> BeautifulSoup | None:
    url = f"{config.BOATRACE_BASE}/{page}?rno={rno}&jcd={jcd}&hd={date}"
    html = http.get(url, cache=http.is_past(date))
    return BeautifulSoup(html, "lxml") if html else None


def st_value(text: str):
    """'.12' 'F.03' 'L' を数値に。フライングは負値、出遅れは None。"""
    t = norm(text).replace(" ", "")
    m = re.search(r"(F)?(\d?\.\d+)", t)
    if not m:
        return None
    v = float(m.group(2))
    return -v if m.group(1) else v


# ---------------------------------------------------------------- 開催場
def venues_of_day(date: str) -> list[str]:
    html = http.get(f"{config.BOATRACE_BASE}/index?hd={date}", cache=http.is_past(date))
    if not html:
        return []
    return sorted(set(re.findall(r"jcd=(\d{2})", html)) & set(config.VENUES))


# ---------------------------------------------------------------- 出走表
def racelist(date: str, jcd: str, rno: int):
    s = soup_of("racelist", jcd, rno, date)
    if s is None:
        return None, []
    rid = race_id(date, jcd, rno)
    race = {"race_id": rid, "date": date, "jcd": jcd, "rno": rno}
    t = s.select_one(".heading2_titleName")
    race["title"] = norm(t.get_text()) if t else None
    h3 = s.select_one("h3.title16_titleDetail__add2020") or s.select_one(".heading2_title h3")
    if h3:
        txt = norm(h3.get_text(" "))
        race["distance"] = num(re.search(r"\d{4}m", txt).group(), int) if re.search(r"\d{4}m", txt) else None
        race["grade"] = txt.split()[0] if txt else None
    for tr in s.select("table tr"):
        if "締切予定時刻" in tr.get_text():
            times = re.findall(r"\d{1,2}:\d{2}", tr.get_text())
            if len(times) >= rno:
                race["deadline"] = times[rno - 1]
            break

    entries = []
    for tb in s.select("tbody.is-fs12")[:6]:
        tds = tb.select("tr")[0].find_all("td", recursive=False)
        if not tds:
            continue
        boat = num(tds[0].get_text(), int)
        e = {"race_id": rid, "boat": boat}
        small = [norm(d.get_text(" ")) for d in tb.select("div.is-fs11")]
        if small:
            m = re.search(r"(\d{4})\s*/\s*([AB][12])", small[0])
            if m:
                e["racer_id"], e["class"] = int(m.group(1)), m.group(2)
        if len(small) > 1:
            parts = small[1].split()
            e["branch"] = parts[0].split("/")[0] if parts else None
            e["age"] = num(re.search(r"\d+歳", small[1]).group(), int) if re.search(r"\d+歳", small[1]) else None
            e["weight"] = num(re.search(r"[\d.]+kg", small[1]).group()) if re.search(r"[\d.]+kg", small[1]) else None
        nm = tb.select_one("div.is-fs18")
        e["name"] = norm(nm.get_text()).replace(" ", "") if nm else None
        cells = [norm(td.get_text(" ")).split() for td in tb.select("tr")[0].select("td.is-lineH2")]
        if len(cells) >= 5:
            flst, nat, loc, mot, bt = cells[:5]
            e["f_count"] = num(next((x for x in flst if x.startswith("F")), None), int)
            e["l_count"] = num(next((x for x in flst if x.startswith("L")), None), int)
            e["avg_st"] = num(flst[-1]) if flst else None
            e["nat_win"], e["nat_2"], e["nat_3"] = (num(x) for x in (nat + [None] * 3)[:3])
            e["loc_win"], e["loc_2"], e["loc_3"] = (num(x) for x in (loc + [None] * 3)[:3])
            e["motor_no"], e["motor_2"], e["motor_3"] = num((mot + [None])[0], int), *(num(x) for x in (mot + [None] * 3)[1:3])
            e["boat_no"], e["boat_2"], e["boat_3"] = num((bt + [None])[0], int), *(num(x) for x in (bt + [None] * 3)[1:3])
        entries.append(e)
    return race, entries


# ---------------------------------------------------------------- 直前情報
def beforeinfo(date: str, jcd: str, rno: int):
    s = soup_of("beforeinfo", jcd, rno, date)
    if s is None:
        return {}, []
    rid = race_id(date, jcd, rno)
    entries = {}
    for tb in s.select("table.is-w748 tbody")[:6]:
        rows = tb.select("tr")
        cells = [norm(td.get_text(" ")) for td in rows[0].find_all("td")]
        if not cells:
            continue
        boat = num(cells[0], int)
        if boat not in BOATS:
            continue
        e = {"race_id": rid, "boat": boat}
        rest = cells[3:]
        e["ex_time"] = next((num(c) for c in rest if re.fullmatch(r"\d\.\d{2}", c)), None)
        e["tilt"] = next((num(c) for c in rest if re.fullmatch(r"-?\d\.\d", c)), None)
        e["parts"] = " ".join(c for c in rest if re.search(r"[ぁ-んァ-ン一-龥]", c)) or None
        if len(rows) > 1:
            adj = [norm(td.get_text()) for td in rows[1].find_all("td")]
            e["adj_weight"] = next((num(c) for c in adj if re.fullmatch(r"\d+\.\d", c)), None)
        entries[boat] = e

    # スタート展示：上から順に進入コース
    for course, row in enumerate(s.select("div.table1 table.is-w238 tbody tr"), start=1):
        b = row.select_one("[class*=boatImage1Number]")
        t = row.select_one("[class*=boatImage1Time]")
        boat = num(b.get_text(), int) if b else None
        if boat in entries:
            entries[boat]["ex_course"] = course
            entries[boat]["ex_st"] = st_value(t.get_text()) if t else None

    race = {"race_id": rid}
    for unit in s.select(".weather1_bodyUnit"):
        title = norm(unit.select_one(".weather1_bodyUnitLabelTitle").get_text()) if unit.select_one(".weather1_bodyUnitLabelTitle") else ""
        data = unit.select_one(".weather1_bodyUnitLabelData")
        val = num(data.get_text()) if data else None
        if "気温" in title:
            race["temp"] = val
        elif "風速" in title:
            race["wind_speed"] = val
        elif "水温" in title:
            race["water_temp"] = val
        elif "波高" in title:
            race["wave"] = val
        elif title and data is None:
            race["weather"] = title
        img = unit.select_one("[class*=is-wind]")
        if img:
            m = re.search(r"is-wind(\d+)", " ".join(img.get("class", [])))
            if m:
                race["wind_dir"] = m.group(1)
    return race, list(entries.values())


# ---------------------------------------------------------------- オッズ
def _grid_combos(first_list_len: int, depth: int):
    """公式オッズ表は「1着ごとの列」が横に並び、行が2着・3着の昇順。"""
    per_first = {
        f: [c for c in itertools.permutations(BOATS, depth) if c[0] == f] for f in BOATS
    }
    rows = len(per_first[1])
    return [per_first[f][r] for r in range(rows) for f in BOATS]


def odds_3t(date: str, jcd: str, rno: int) -> dict[str, float | None]:
    """チャッピーの実績ある読み方と同じ：表の中の2着・3着の艇番を直接読む（並び順を仮定しない）。
    各行は1着1〜6の順に [2着(rowspan=4の行だけ)] [3着] [オッズ] が並ぶ。"""
    s = soup_of("odds3t", jcd, rno, date)
    if s is None:
        return {}
    body = s.select_one("tbody.is-p3-0")
    out, second = {}, [0] * 6
    if body is not None:
        for tr in body.find_all("tr", recursive=False):
            cells = tr.find_all("td", recursive=False)
            i = 0
            for first in BOATS:
                if i < len(cells) and str(cells[i].get("rowspan", "")) == "4":
                    second[first - 1] = num(cells[i].get_text(), int) or 0
                    i += 1
                if i + 1 >= len(cells):
                    break
                third, odd = num(cells[i].get_text(), int), num(cells[i + 1].get_text())
                i += 2
                combo = (first, second[first - 1], third)
                if all(c in BOATS for c in combo) and len(set(combo)) == 3:
                    out["-".join(map(str, combo))] = odd
    if len(out) == 120:
        return out
    # 予備：oddsPointを並び順で割り当て
    vals = [norm(td.get_text()) for td in s.select("td.oddsPoint")][:120]
    if len(vals) < 120:
        return out
    return {"-".join(map(str, c)): num(v) for c, v in zip(_grid_combos(6, 3), vals)}


def odds_2t(date: str, jcd: str, rno: int) -> dict[str, float | None]:
    s = soup_of("odds2tf", jcd, rno, date)
    if s is None:
        return {}
    vals = [norm(td.get_text()) for td in s.select("td.oddsPoint")][:30]
    if len(vals) < 30:
        return {}
    return {"-".join(map(str, c)): num(v) for c, v in zip(_grid_combos(6, 2), vals)}


# ---------------------------------------------------------------- 結果
BET_TYPES = ("3連単", "3連複", "2連単", "2連複", "拡連複", "単勝", "複勝")


def result(date: str, jcd: str, rno: int):
    s = soup_of("raceresult", jcd, rno, date)
    if s is None:
        return [], []
    rid = race_id(date, jcd, rno)
    entries = []
    tables = s.select("table.is-w495")
    if tables:
        for tr in tables[0].select("tbody tr"):
            c = [norm(td.get_text(" ")) for td in tr.find_all("td")]
            if len(c) < 2:
                continue
            boat = num(c[1], int)
            if boat not in BOATS:
                continue
            fin = num(c[0], int)
            rt = re.match(r"(\d)'(\d+)\"(\d)", c[-1].replace(" ", ""))
            entries.append({
                "race_id": rid, "boat": boat, "finish": fin,
                "race_time": int(rt[1]) * 60 + int(rt[2]) + int(rt[3]) / 10 if rt else None,
            })
    starts = s.select("table.is-w495 [class*=boatImage1Number]")
    for course, b in enumerate(starts[:6], start=1):
        boat = num(b.get_text(), int)
        t = b.find_parent("tr").select_one("[class*=boatImage1Time]") if b.find_parent("tr") else None
        for e in entries:
            if e["boat"] == boat:
                e["course"], e["st"] = course, st_value(t.get_text()) if t else None

    kim = None
    for th in s.find_all(["th", "td"]):
        if norm(th.get_text()) == "決まり手":
            cell = th.find_parent("table").select_one("tbody td") if th.find_parent("table") else None
            m = re.search(r"(逃げ|まくり差し|差し|まくり|抜き|恵まれ)", norm(cell.get_text())) if cell else None
            kim = m.group(1) if m else None
            break
    payouts, current = [], None
    for tr in s.select("table tr"):
        text = norm(tr.get_text(" "))
        for bt in BET_TYPES:
            if text.startswith(bt):
                current = bt
        if current not in ("3連単", "2連単"):
            continue
        nums = [norm(x.get_text()) for x in tr.select("[class*=numberSet1_number]")]
        pay = re.search(r"¥\s*([\d,]+)", text)
        if len(nums) >= 2 and pay:
            pop = re.findall(r"(\d+)\s*$", text)
            payouts.append({
                "race_id": rid, "bet_type": current, "combo": "-".join(nums),
                "payout": int(pay.group(1).replace(",", "")),
                "popularity": int(pop[0]) if pop else None,
            })
    return entries, payouts, kim

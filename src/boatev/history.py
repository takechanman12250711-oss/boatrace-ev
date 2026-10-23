"""過去データ：公式の番組表(B)・競走成績(K)のLZHを取り込む。

学習データの大部分はここから作る（1日分で全場・全レース）。
  B: 選手登番・級別・全国/当地勝率・モーター/ボート2連率
  K: 展示タイム・進入コース・ST・着順・3連単/2連単の払戻
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from . import config, http
from .db import race_id


def _download_text(kind: str, date: str) -> str | None:
    yymmdd, yyyymm = date[2:], date[:6]
    url = f"{config.MBRACE_BASE}/{kind.upper()}/{yyyymm}/{kind.lower()}{yymmdd}.lzh"
    blob = http.get(url, binary=True, cache=True)
    if not blob:
        return None
    return _extract_lzh(blob).decode("cp932", errors="replace")


def _extract_lzh(blob: bytes) -> bytes:
    try:
        import lhafile  # pip install lhafile

        f = lhafile.LhaFile(io.BytesIO(blob))
        return b"".join(f.read(info.filename) for info in f.infolist())
    except ImportError:
        pass
    if not shutil.which("lha"):
        raise RuntimeError("LZH展開には `pip install lhafile` か lha コマンドが必要です")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.lzh"
        p.write_bytes(blob)
        subprocess.run(["lha", "xqw=" + d, str(p)], check=True)
        return b"".join(x.read_bytes() for x in Path(d).iterdir() if x.suffix.upper() == ".TXT")


MISSES: list[dict] = []   # 読めなかった行（調査用）


def _miss(kind, date, line):
    if len(MISSES) < 300:
        MISSES.append({"kind": kind, "date": date, "line": line})


def _lines(text: str):
    for raw in text.splitlines():
        yield unicodedata.normalize("NFKC", raw).rstrip()


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 番組表
B_ENTRY = re.compile(
    r"^([1-6])\s(\d{4})(\D+?)(\d{2})(\D+?)(\d{2})([AB][12])\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)"
)


def parse_b(text: str, date: str):
    races, entries, jcd, rno = [], [], None, None
    for line in _lines(text):
        m = re.match(r"^(\d{2})BBGN", line)
        if m:
            jcd = m.group(1)
            continue
        m = re.match(r"^\s*(\d{1,2})R\s+(.+?)\s+H(\d{4})m(.*)$", line)
        if m and jcd:
            rno = int(m.group(1))
            dl = re.search(r"(\d{1,2}:\d{2})", m.group(4))
            races.append({
                "race_id": race_id(date, jcd, rno), "date": date, "jcd": jcd, "rno": rno,
                "grade": m.group(2).split()[0], "distance": int(m.group(3)),
                "deadline": dl.group(1) if dl else None,
            })
            continue
        m = B_ENTRY.match(line)
        if not m and jcd and rno and re.match(r"^[1-6]\s\d{4}", line):
            _miss("B", date, line)
        if m and jcd and rno:
            g = m.groups()
            entries.append({
                "race_id": race_id(date, jcd, rno), "boat": int(g[0]), "racer_id": int(g[1]),
                "name": g[2].replace(" ", ""), "age": int(g[3]), "branch": g[4].strip(),
                "weight": float(g[5]), "class": g[6],
                "nat_win": _f(g[7]), "nat_2": _f(g[8]), "loc_win": _f(g[9]), "loc_2": _f(g[10]),
                "motor_no": int(g[11]), "motor_2": _f(g[12]), "boat_no": int(g[13]), "boat_2": _f(g[14]),
            })
    return races, entries


# ---------------------------------------------------------------- 競走成績
K_ENTRY = re.compile(r"^\s*(\S{1,2})\s+([1-6])\s+(\d{4})\s+(\D+?)\s+(\d+)\s+(\d+)\s*(.*)$")
K_PAY = re.compile(r"^\s*(3連単|2連単)\s+(\d-\d(?:-\d)?)\s+(\d+)(?:\s+人気\s+(\d+))?")


def _race_time(tok: str):
    m = re.fullmatch(r"(\d)\.(\d{2})\.(\d)", tok or "")
    return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 10 if m else None


def parse_k(text: str, date: str):
    races, entries, payouts, jcd, rno = [], [], [], None, None
    for line in _lines(text):
        m = re.match(r"^(\d{2})KBGN", line)
        if m:
            jcd, rno = m.group(1), None
            continue
        if re.match(r"^\s*\d{1,2}R\s", line) and not re.match(r"^\s*(\d{1,2})R\s+(.+?)\s+H(\d{4})m\s+(.*)$", line):
            _miss("K-header", date, line)
        m = re.match(r"^\s*(\d{1,2})R\s+(.+?)\s+H(\d{4})m\s+(.*)$", line)
        if m and jcd:
            rno = int(m.group(1))
            rest = m.group(4)
            wd = re.search(r"風\s+(\S+)\s+(\d+)m", rest)
            wv = re.search(r"波\s+(\d+)cm", rest)
            races.append({
                "race_id": race_id(date, jcd, rno), "date": date, "jcd": jcd, "rno": rno,
                "distance": int(m.group(3)), "weather": rest.split()[0] if rest.split() else None,
                "wind_dir": wd.group(1) if wd else None, "wind_speed": _f(wd.group(2)) if wd else None,
                "wave": _f(wv.group(1)) if wv else None,
            })
            continue
        if not (jcd and rno):
            continue
        m = K_ENTRY.match(line)
        if not m and re.match(r"^\s*\S{1,2}\s+[1-6]\s+\d{4}\s", line):
            _miss("K", date, line)
        if m and re.fullmatch(r"\d{2}|[A-Z]\d?|\S", m.group(1)):
            fin, boat, rid_, _, motor, boat_no, rest = m.groups()
            tok = rest.split()
            st = None
            if len(tok) > 2:
                st = _f(tok[2].lstrip("F"))
                if st is not None and tok[2].startswith("F"):
                    st = -st
            entries.append({
                "race_id": race_id(date, jcd, rno), "boat": int(boat), "racer_id": int(rid_),
                "finish": int(fin) if fin.isdigit() else None,
                "ex_time": _f(tok[0]) if tok else None,
                "course": int(tok[1]) if len(tok) > 1 and tok[1].isdigit() else None,
                "st": st, "race_time": _race_time(tok[3]) if len(tok) > 3 else None,
            })
            continue
        m = K_PAY.match(line)
        if m:
            payouts.append({
                "race_id": race_id(date, jcd, rno), "bet_type": m.group(1), "combo": m.group(2),
                "payout": int(m.group(3)), "popularity": int(m.group(4)) if m.group(4) else None,
            })
    return races, entries, payouts


def fetch_day(date: str):
    """1日分のB/Kを取得してパース。開催なし・未公開は空。"""
    b = _download_text("b", date)
    k = _download_text("k", date)
    races_b, entries_b = parse_b(b, date) if b else ([], [])
    races_k, entries_k, payouts = parse_k(k, date) if k else ([], [], [])
    return races_b + races_k, entries_b + entries_k, payouts

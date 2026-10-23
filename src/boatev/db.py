"""SQLite。レース・出走・オッズ・払戻を1つのDBにまとめる。"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterable, Mapping

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,        -- YYYYMMDD_jcd_rno
    date TEXT, jcd TEXT, rno INTEGER,
    title TEXT, grade TEXT, distance INTEGER, deadline TEXT,
    weather TEXT, wind_dir TEXT, wind_speed REAL, wave REAL,
    temp REAL, water_temp REAL,
    kimarite TEXT                   -- 決まり手（1着艇の）
);
CREATE TABLE IF NOT EXISTS entries (
    race_id TEXT, boat INTEGER,
    racer_id INTEGER, name TEXT, age INTEGER, branch TEXT, weight REAL, class TEXT,
    f_count INTEGER, l_count INTEGER, avg_st REAL,
    nat_win REAL, nat_2 REAL, nat_3 REAL,
    loc_win REAL, loc_2 REAL, loc_3 REAL,
    motor_no INTEGER, motor_2 REAL, motor_3 REAL,
    boat_no INTEGER, boat_2 REAL, boat_3 REAL,
    -- 直前情報
    ex_time REAL, tilt REAL, propeller TEXT, parts TEXT, adj_weight REAL,
    ex_course INTEGER, ex_st REAL,
    -- 結果
    course INTEGER, st REAL, finish INTEGER, race_time REAL,
    PRIMARY KEY (race_id, boat)
);
CREATE TABLE IF NOT EXISTS odds (
    race_id TEXT, bet_type TEXT, combo TEXT, odds REAL, fetched_at TEXT,
    PRIMARY KEY (race_id, bet_type, combo)
);
CREATE TABLE IF NOT EXISTS payouts (
    race_id TEXT, bet_type TEXT, combo TEXT, payout INTEGER, popularity INTEGER,
    PRIMARY KEY (race_id, bet_type, combo)
);
CREATE INDEX IF NOT EXISTS idx_entries_racer ON entries(racer_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
"""


def connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    _migrate(con)
    return con


MIGRATIONS = {"races": {"kimarite": "TEXT"}}


def _migrate(con):
    """既存DB（Actionsのキャッシュ）に後から足した列を追加する。"""
    for table, cols in MIGRATIONS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for col, typ in cols.items():
            if col not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


@contextmanager
def session():
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def upsert(con: sqlite3.Connection, table: str, rows: Iterable[Mapping], keys: tuple[str, ...]) -> int:
    """主キーが同じ行は、渡された列だけ上書き（他の列は保持）。"""
    n = 0
    for row in rows:
        row = {k: v for k, v in row.items() if v is not None}
        if not row:
            continue
        cols = list(row)
        placeholders = ",".join("?" for _ in cols)
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in keys)
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        sql += f" ON CONFLICT({','.join(keys)}) DO " + (f"UPDATE SET {updates}" if updates else "NOTHING")
        con.execute(sql, [row[c] for c in cols])
        n += 1
    return n


def race_id(date: str, jcd: str, rno: int) -> str:
    return f"{date}_{jcd}_{int(rno):02d}"

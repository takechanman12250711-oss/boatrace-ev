"""全体設定。環境変数で上書きできる値はここに集約する。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("BOATEV_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = Path(os.environ.get("BOATEV_DB", DATA_DIR / "boatev.sqlite"))
MODEL_PATH = DATA_DIR / "model.pkl"
SITE_DATA_DIR = ROOT / "docs" / "data"

# 公式サイト
BOATRACE_BASE = "https://www.boatrace.jp/owpc/pc/race"
# 過去データ（番組表=B / 競走成績=K）のLZH
MBRACE_BASE = "https://www1.mbrace.or.jp/od2"

# アクセスはゆっくり（公式サイトへの負荷を避ける）
REQUEST_INTERVAL_SEC = float(os.environ.get("BOATEV_INTERVAL", "1.5"))
USER_AGENT = "boatrace-ev/0.1 (personal research)"

VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川",
    "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国",
    "11": "びわこ", "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀",
    "16": "児島", "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

# 買い目選定のしきい値（バックテストで調整する）
# 2026/10 最終バックテストで選ばれた条件（期待値1.5以上・オッズ200倍以下・モデルと市場を半々）
EV_THRESHOLD = float(os.environ.get("BOATEV_EV", "1.5"))    # 期待値がこれ以上
MIN_PROB = float(os.environ.get("BOATEV_MIN_PROB", "0.01"))  # 確率1%未満は除外
MAX_ODDS = float(os.environ.get("BOATEV_MAX_ODDS", "200"))   # 高すぎるオッズはモデル誤差が大きい
MARKET_BLEND = float(os.environ.get("BOATEV_BLEND", "0.5"))  # モデルと市場確率の混合比（0=市場のみ,1=モデルのみ）
KELLY_FRACTION = 0.1   # ケリー基準の何割を賭けるか
MAX_BET_SHARE = 0.02   # 1点あたり資金の上限比率
MAX_POINTS_PER_RACE = 6

# 予想の記録は1点100円で統一（バックテストと同じ条件で比べるため）
FLAT_STAKE = 100

# ロマン枠：本線とは別に、高配当の3連単を1レース1点だけ出す。成績は本線と分けて記録する
ROMAN_MIN_ODDS = 100     # 100倍以上
ROMAN_MAX_ODDS = 2000
ROMAN_MIN_PROB = 0.002   # 確率0.2%未満（500回に1回未満）は出さない
ROMAN_EV = 1.3           # ロマン候補になる期待値の下限（バックテスト用）
ROMAN_DAILY_EV = 2.0     # 予想で出すのはこの期待値以上だけ
ROMAN_PER_DAY = 1        # 1日に出すロマンの上限

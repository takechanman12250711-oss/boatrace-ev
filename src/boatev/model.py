"""着順確率モデル。

1. 勾配ブースティングで各艇の「1着になる確率」を出す
2. レース内で合計1に正規化
3. Plackett-Luce（2着・3着は補正指数 λ2, λ3 付き）で 3連単120通り／2連単30通りの確率に展開
   ※ 1着確率をそのまま2・3着に流用すると強い艇を過大評価するので、λを検証データで推定する
"""
from __future__ import annotations

import itertools
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from .features import FEATURES

PERM3 = list(itertools.permutations(range(6), 3))
PERM2 = list(itertools.permutations(range(6), 2))


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, None)
    return p / p.sum()


def order_probs(p: np.ndarray, lam2: float, lam3: float) -> dict[str, np.ndarray]:
    """p: 6艇の1着確率 → {'3t': 120通り, '2t': 30通り}"""
    p = _norm(np.asarray(p, float))
    q2, q3 = p ** lam2, p ** lam3
    t3 = np.empty(len(PERM3))
    for n, (i, j, k) in enumerate(PERM3):
        t3[n] = p[i] * q2[j] / (q2.sum() - q2[i]) * q3[k] / (q3.sum() - q3[i] - q3[j])
    t2 = np.array([p[i] * q2[j] / (q2.sum() - q2[i]) for i, j in PERM2])
    return {"3t": t3, "2t": t2}


def combo_names(kind: str) -> list[str]:
    perms = PERM3 if kind == "3t" else PERM2
    return ["-".join(str(x + 1) for x in c) for c in perms]


class Model:
    def __init__(self, features: list[str] | None = None):
        self.features = list(features or FEATURES)
        self.clf = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=0,
        )
        self.lam2 = self.lam3 = 1.0
        self.metrics: dict = {}

    # -- 1着確率（レース内正規化）
    def win_probs(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.clf.predict_proba(df[self._feats()])[:, 1]
        s = pd.Series(raw, index=df.index)
        return (s / s.groupby(df["race_id"]).transform("sum")).values

    def fit(self, df: pd.DataFrame, valid_share: float = 0.2, refit: bool = True):
        dates = np.sort(df["date"].unique())
        cut = dates[int(len(dates) * (1 - valid_share))]
        tr, va = df[df["date"] < cut], df[df["date"] >= cut]
        self.clf.fit(tr[self._feats()], (tr["finish"] == 1).astype(int))

        va = va.assign(p=self.win_probs(va))
        races = [(g["p"].values, g["finish"].values) for _, g in va.groupby("race_id")]
        self.lam2, self.lam3 = self._fit_lambdas(races)
        self.metrics = self._evaluate(races) | {"train_races": tr["race_id"].nunique(),
                                                "valid_races": len(races), "valid_from": cut}
        # 最終モデルは全期間で学習し直す
        if refit:
            self.clf.fit(df[self._feats()], (df["finish"] == 1).astype(int))
        return self

    def _feats(self):
        return getattr(self, "features", FEATURES)  # 旧バージョンのpickle互換

    @staticmethod
    def _ll(races, lam2, lam3):
        ll = 0.0
        for p, fin in races:
            order = [int(np.where(fin == r)[0][0]) for r in (1, 2, 3)]
            i, j, k = order
            q2, q3 = p ** lam2, p ** lam3
            ll += np.log(q2[j] / (q2.sum() - q2[i])) + np.log(q3[k] / (q3.sum() - q3[i] - q3[j]))
        return ll

    def _fit_lambdas(self, races):
        grid = np.arange(0.3, 1.21, 0.05)
        best = max(((a, b) for a in grid for b in grid if b <= a + 1e-9),
                   key=lambda ab: self._ll(races, *ab))
        return float(best[0]), float(best[1])

    def _evaluate(self, races):
        hit1 = np.mean([np.argmax(p) == np.where(f == 1)[0][0] for p, f in races])
        ll_win = -np.mean([np.log(p[np.where(f == 1)[0][0]]) for p, f in races])
        ll3, hit3 = [], []
        for p, f in races:
            t3 = order_probs(p, self.lam2, self.lam3)["3t"]
            truth = tuple(int(np.where(f == r)[0][0]) for r in (1, 2, 3))
            idx = PERM3.index(truth)
            ll3.append(-np.log(t3[idx]))
            hit3.append(np.argmax(t3) == idx)
        return {"win_top1_acc": round(float(hit1), 4), "win_logloss": round(float(ll_win), 4),
                "tri_logloss": round(float(np.mean(ll3)), 4), "tri_top1_acc": round(float(np.mean(hit3)), 4),
                "tri_logloss_uniform": round(float(np.log(120)), 4),
                "lam2": self.lam2, "lam3": self.lam3}

    def save(self, path=config.MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=config.MODEL_PATH) -> "Model":
        with open(path, "rb") as f:
            return pickle.load(f)

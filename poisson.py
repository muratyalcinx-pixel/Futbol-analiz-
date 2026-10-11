"""
poisson.py
Zaman ağırlıklı Poisson modeli: takım hücum/savunma güçlerinden beklenen golleri
hesaplar, skor matrisi üzerinden 1-X-2, Alt/Üst ve KG Var olasılıklarını üretir.

Kullanım:
    from data_loader import load_matches
    from poisson import PoissonModel

    df = load_matches(["T1"], n_seasons=5)
    model = PoissonModel().fit(df)
    print(model.predict("Galatasaray", "Fenerbahce"))
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson


class PoissonModel:
    def __init__(self, half_life_days: float = 365, shrinkage: float = 5, max_goals: int = 10):
        """
        half_life_days : Maç ağırlığının yarıya düştüğü gün sayısı (yeni maçlar daha önemli).
        shrinkage      : Az maç oynayan takımları lig ortalamasına çeken sahte maç sayısı.
        max_goals      : Skor matrisinde bir takım için hesaplanacak maksimum gol.
        """
        self.xi = np.log(2) / half_life_days
        self.k = shrinkage
        self.max_goals = max_goals
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, as_of=None) -> "PoissonModel":
        """df sütunları: date, home, away, home_goals, away_goals"""
        as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
        d = df[df["date"] <= as_of].copy()
        if d.empty:
            raise ValueError("Eğitim için veri yok.")

        days_ago = (as_of - d["date"]).dt.days.clip(lower=0)
        d["w"] = np.exp(-self.xi * days_ago)
        d["w_hg"] = d["w"] * d["home_goals"]
        d["w_ag"] = d["w"] * d["away_goals"]

        # Lig ortalamaları (ağırlıklı)
        self.lg_home = d["w_hg"].sum() / d["w"].sum()
        self.lg_away = d["w_ag"].sum() / d["w"].sum()

        k = self.k
        h = d.groupby("home")[["w", "w_hg", "w_ag"]].sum()
        a = d.groupby("away")[["w", "w_hg", "w_ag"]].sum()

        # Ev sahibi: hücum = attığı gol, savunma = yediği gol (lig ortalamasına oranla)
        self.home_att = ((h["w_hg"] + k * self.lg_home) / (h["w"] + k)) / self.lg_home
        self.home_def = ((h["w_ag"] + k * self.lg_away) / (h["w"] + k)) / self.lg_away
        # Deplasman
        self.away_att = ((a["w_ag"] + k * self.lg_away) / (a["w"] + k)) / self.lg_away
        self.away_def = ((a["w_hg"] + k * self.lg_home) / (a["w"] + k)) / self.lg_home

        self.teams = sorted(set(h.index) | set(a.index))
        self.fitted = True
        return self

    # -------------------------------------------------------------- predict
    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Bilinmeyen takım (ör. yeni çıkan) için güç 1.0 (lig ortalaması) kabul edilir."""
        if not self.fitted:
            raise RuntimeError("Önce fit() çağırın.")
        lam_h = self.home_att.get(home, 1.0) * self.away_def.get(away, 1.0) * self.lg_home
        lam_a = self.away_att.get(away, 1.0) * self.home_def.get(home, 1.0) * self.lg_away
        return float(lam_h), float(lam_a)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """M[i, j] = ev sahibi i gol, deplasman j gol atma olasılığı."""
        lam_h, lam_a = self.expected_goals(home, away)
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
        return m / m.sum()

    def predict(self, home: str, away: str, ou_lines=(1.5, 2.5, 3.5)) -> dict:
        m = self.score_matrix(home, away)
        lam_h, lam_a = self.expected_goals(home, away)

        i, j = np.indices(m.shape)
        total = i + j

        out = {
            "home": home,
            "away": away,
            "xg_home": round(lam_h, 2),
            "xg_away": round(lam_a, 2),
            "p1": float(np.tril(m, -1).sum()),   # ev sahibi kazanır
            "px": float(np.trace(m)),            # beraberlik
            "p2": float(np.triu(m, 1).sum()),    # deplasman kazanır
            "btts_yes": float(m[1:, 1:].sum()),  # karşılıklı gol var
        }
        out["btts_no"] = 1 - out["btts_yes"]

        for line in ou_lines:
            over = float(m[total > line].sum())
            out[f"over_{line}"] = over
            out[f"under_{line}"] = 1 - over

        best = np.unravel_index(m.argmax(), m.shape)
        out["most_likely_score"] = f"{best[0]}-{best[1]}"
        return out


# -------------------------------------------------------------- backtest
def _market_probs(r):
    """Bahis oranlarından marj arındırılmış 1-X-2 olasılıkları (yoksa NaN)."""
    o = np.array([r.get("odds_1"), r.get("odds_x"), r.get("odds_2")], dtype=float)
    if np.all(np.isfinite(o)) and np.all(o > 1):
        raw = 1 / o
        return raw / raw.sum(), raw.sum() - 1
    return np.full(3, np.nan), np.nan


def _market_over25(r):
    o = np.array([r.get("odds_over25"), r.get("odds_under25")], dtype=float)
    if np.all(np.isfinite(o)) and np.all(o > 1):
        raw = 1 / o
        return raw[0] / raw.sum()
    return np.nan


def backtest_detailed(df: pd.DataFrame, start_date, **model_kwargs) -> pd.DataFrame:
    """
    Zaman sıralı doğrulama: her maç gününde, o günden ÖNCEKİ verilerle model eğitilir.
    Her maç için model olasılıkları, (varsa) bahis oranlarından çıkan olasılıklar
    ve gerçek sonuç tek satırda döndürülür.
    """
    df = df.sort_values("date")
    days = df.loc[df["date"] >= pd.Timestamp(start_date), "date"].unique()

    rows = []
    for day in days:
        train = df[df["date"] < day]
        if len(train) < 100:
            continue
        model = PoissonModel(**model_kwargs).fit(train)
        for _, r in df[df["date"] == day].iterrows():
            p = model.predict(r["home"], r["away"])
            m, overround = _market_probs(r)
            rows.append({
                "date": r["date"], "home": r["home"], "away": r["away"],
                "score": f"{int(r['home_goals'])}-{int(r['away_goals'])}",
                "result": r["result"],
                "p1": p["p1"], "px": p["px"], "p2": p["p2"],
                "m1": m[0], "mx": m[1], "m2": m[2], "overround": overround,
                "p_over25": p["over_2.5"], "m_over25": _market_over25(r),
                "over25": int(r["home_goals"] + r["away_goals"] > 2.5),
            })
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    """backtest_detailed çıktısından özet metrikler (Brier, log-loss, isabet)."""
    if bt.empty:
        return {"matches": 0}

    Y = np.column_stack([bt["result"] == "H", bt["result"] == "D", bt["result"] == "A"]).astype(float)
    P = bt[["p1", "px", "p2"]].to_numpy()
    M = bt[["m1", "mx", "m2"]].to_numpy()

    brier = ((P - Y) ** 2).sum(axis=1)
    logloss = -np.log(np.clip((P * Y).sum(axis=1), 1e-9, None))
    acc = P.argmax(axis=1) == Y.argmax(axis=1)

    out = {
        "matches": len(bt),
        "brier": float(brier.mean()),
        "log_loss": float(logloss.mean()),
        "accuracy": float(acc.mean()),
        "odds_matches": 0,
        "ou_matches": 0,
    }

    # Bahis oranı karşılaştırması: yalnızca oranı olan maçlar (adil kıyas)
    mask = ~np.isnan(M).any(axis=1)
    if mask.any():
        Pm, Mm, Ym = P[mask], M[mask], Y[mask]
        out.update({
            "odds_matches": int(mask.sum()),
            "model_brier_odds": float(((Pm - Ym) ** 2).sum(axis=1).mean()),
            "market_brier": float(((Mm - Ym) ** 2).sum(axis=1).mean()),
            "model_ll_odds": float(-np.log(np.clip((Pm * Ym).sum(axis=1), 1e-9, None)).mean()),
            "market_ll": float(-np.log(np.clip((Mm * Ym).sum(axis=1), 1e-9, None)).mean()),
            "model_acc_odds": float((Pm.argmax(axis=1) == Ym.argmax(axis=1)).mean()),
            "market_acc": float((Mm.argmax(axis=1) == Ym.argmax(axis=1)).mean()),
            "beats_market_share": float(((Pm * Ym).sum(axis=1) > (Mm * Ym).sum(axis=1)).mean()),
            "avg_overround": float(bt.loc[mask, "overround"].mean()),
        })

    # Alt/Üst 2.5 (ikili Brier)
    mask2 = bt["m_over25"].notna().to_numpy()
    if mask2.any():
        y2 = bt.loc[mask2, "over25"].to_numpy(dtype=float)
        out.update({
            "ou_matches": int(mask2.sum()),
            "model_ou_brier": float(((bt.loc[mask2, "p_over25"].to_numpy() - y2) ** 2).mean()),
            "market_ou_brier": float(((bt.loc[mask2, "m_over25"].to_numpy() - y2) ** 2).mean()),
        })
    return out


def backtest(df: pd.DataFrame, start_date, **model_kwargs) -> dict:
    """Kısa yol: özet sözlük döndürür."""
    return summarize(backtest_detailed(df, start_date, **model_kwargs))


if __name__ == "__main__":
    from data_loader import load_matches

    data = load_matches(["T1"], n_seasons=5)
    model = PoissonModel().fit(data)

    home, away = model.teams[0], model.teams[1]
    print(model.predict(home, away))

    bt = backtest_detailed(data, data["date"].max() - pd.Timedelta(days=365))
    print(summarize(bt))

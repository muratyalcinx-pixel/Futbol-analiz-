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
def backtest(df: pd.DataFrame, start_date: str, **model_kwargs) -> dict:
    """
    Zaman sıralı doğrulama: her maç gününde, o günden ÖNCEKİ verilerle model eğitilir.
    Brier skoru ve log-loss döndürür (düşük = iyi). Rastgele bölme kullanılmaz.
    """
    df = df.sort_values("date")
    test_days = df.loc[df["date"] >= pd.Timestamp(start_date), "date"].unique()

    briers, losses, correct = [], [], []
    for day in test_days:
        train = df[df["date"] < day]
        if len(train) < 100:
            continue
        model = PoissonModel(**model_kwargs).fit(train)
        for _, r in df[df["date"] == day].iterrows():
            p = model.predict(r["home"], r["away"])
            probs = np.array([p["p1"], p["px"], p["p2"]])
            y = np.array([r["result"] == "H", r["result"] == "D", r["result"] == "A"], dtype=float)
            briers.append(((probs - y) ** 2).sum())
            losses.append(-np.log(max(probs[y.argmax()], 1e-9)))
            correct.append(probs.argmax() == y.argmax())

    return {
        "matches": len(briers),
        "brier": float(np.mean(briers)),
        "log_loss": float(np.mean(losses)),
        "accuracy": float(np.mean(correct)),
    }


if __name__ == "__main__":
    from data_loader import load_matches

    data = load_matches(["T1"], n_seasons=5)
    model = PoissonModel().fit(data)

    home, away = model.teams[0], model.teams[1]
    print(model.predict(home, away))

    print(backtest(data, start_date=data["date"].max() - pd.Timedelta(days=180)))

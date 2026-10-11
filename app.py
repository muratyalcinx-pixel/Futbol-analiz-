# -*- coding: utf-8 -*-
"""
Futbol Olasılık Analizi (Hugging Face Spaces + Gradio)
"""

from datetime import date
from io import StringIO
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from scipy.stats import poisson

# =====================================================================
# 1) VERİ
# =====================================================================
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
RAW_DIR = Path("data/raw")

LEAGUES = {
    "T1": "Türkiye Süper Lig",
    "E0": "İngiltere Premier Lig",
    "SP1": "İspanya La Liga",
    "D1": "Almanya Bundesliga",
    "I1": "İtalya Serie A",
    "F1": "Fransa Ligue 1",
}

RENAME = {
    "Div": "league", "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
}


def season_codes(n_seasons=5):
    today = date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    codes = [f"{(start_year - i) % 100:02d}{(start_year - i + 1) % 100:02d}" for i in range(n_seasons)]
    return codes[::-1]


def _download_csv(season, league):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{league}_{season}.csv"
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        try:
            resp = requests.get(BASE_URL.format(season=season, league=league), timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[UYARI] {league} {season} indirilemedi: {e}")
            return None
        text = resp.content.decode("latin-1")
        cache_path.write_text(text, encoding="utf-8")
    try:
        df = pd.read_csv(StringIO(text))
    except Exception as e:
        print(f"[UYARI] {league} {season} okunamadı: {e}")
        return None
    df = df.dropna(how="all")
    df["season"] = season
    return df


def load_matches(leagues, n_seasons=5):
    frames = [_download_csv(s, lg) for lg in leagues for s in season_codes(n_seasons)]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError("Hiç veri indirilemedi. İnternet bağlantısını kontrol edin.")
    data = pd.concat(frames, ignore_index=True)
    keep = [c for c in RENAME if c in data.columns] + ["season"]
    data = data[keep].rename(columns=RENAME)
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])
    data[["home_goals", "away_goals"]] = data[["home_goals", "away_goals"]].astype(int)
    return data.sort_values("date").reset_index(drop=True)


# =====================================================================
# 2) MODEL (zaman ağırlıklı Poisson)
# =====================================================================
class PoissonModel:
    def __init__(self, half_life_days=365, shrinkage=5, max_goals=10):
        self.xi = np.log(2) / half_life_days
        self.k = shrinkage
        self.max_goals = max_goals
        self.fitted = False

    def fit(self, df, as_of=None):
        as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
        d = df[df["date"] <= as_of].copy()
        if d.empty:
            raise ValueError("Eğitim için veri yok.")
        days_ago = (as_of - d["date"]).dt.days.clip(lower=0)
        d["w"] = np.exp(-self.xi * days_ago)
        d["w_hg"] = d["w"] * d["home_goals"]
        d["w_ag"] = d["w"] * d["away_goals"]

        self.lg_home = d["w_hg"].sum() / d["w"].sum()
        self.lg_away = d["w_ag"].sum() / d["w"].sum()
        k = self.k
        h = d.groupby("home")[["w", "w_hg", "w_ag"]].sum()
        a = d.groupby("away")[["w", "w_hg", "w_ag"]].sum()

        self.home_att = ((h["w_hg"] + k * self.lg_home) / (h["w"] + k)) / self.lg_home
        self.home_def = ((h["w_ag"] + k * self.lg_away) / (h["w"] + k)) / self.lg_away
        self.away_att = ((a["w_ag"] + k * self.lg_away) / (a["w"] + k)) / self.lg_away
        self.away_def = ((a["w_hg"] + k * self.lg_home) / (a["w"] + k)) / self.lg_home
        self.teams = sorted(set(h.index) | set(a.index))
        self.fitted = True
        return self

    def expected_goals(self, home, away):
        lam_h = self.home_att.get(home, 1.0) * self.away_def.get(away, 1.0) * self.lg_home
        lam_a = self.away_att.get(away, 1.0) * self.home_def.get(home, 1.0) * self.lg_away
        return float(lam_h), float(lam_a)

    def score_matrix(self, home, away):
        lam_h, lam_a = self.expected_goals(home, away)
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
        return m / m.sum()

    def predict(self, home, away, ou_lines=(1.5, 2.5, 3.5)):
        m = self.score_matrix(home, away)
        lam_h, lam_a = self.expected_goals(home, away)
        i, j = np.indices(m.shape)
        total = i + j
        out = {
            "xg_home": round(lam_h, 2), "xg_away": round(lam_a, 2),
            "p1": float(np.tril(m, -1).sum()),
            "px": float(np.trace(m)),
            "p2": float(np.triu(m, 1).sum()),
            "btts_yes": float(m[1:, 1:].sum()),
        }
        out["btts_no"] = 1 - out["btts_yes"]
        for line in ou_lines:
            over = float(m[total > line].sum())
            out[f"over_{line}"] = over
            out[f"under_{line}"] = 1 - over
        best = np.unravel_index(m.argmax(), m.shape)
        out["most_likely_score"] = f"{best[0]}-{best[1]}"
        return out


def backtest(df, start_date, **model_kwargs):
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
    if not briers:
        return {"matches": 0}
    return {"matches": len(briers), "brier": float(np.mean(briers)),
            "log_loss": float(np.mean(losses)), "accuracy": float(np.mean(correct))}


# =====================================================================
# 3) ARAYÜZ (Gradio)
# =====================================================================
_CACHE = {}


def get_model(league, n_seasons, half_life, shrinkage):
    key = (league, int(n_seasons), int(half_life), float(shrinkage))
    if key not in _CACHE:
        try:
            data = load_matches([league], int(n_seasons))
            model = PoissonModel(half_life_days=float(half_life), shrinkage=float(shrinkage)).fit(data)
        except Exception as e:
            raise gr.Error(f"Veri/model hatası: {e}")
        _CACHE[key] = (model, data)
    return _CACHE[key]


def last_matches(data, team, n=5):
    d = data[(data["home"] == team) | (data["away"] == team)].tail(n).copy()
    d["Skor"] = d["home_goals"].astype(str) + " - " + d["away_goals"].astype(str)
    d["Tarih"] = d["date"].dt.strftime("%d.%m.%Y")
    return d[["Tarih", "home", "Skor", "away"]].rename(columns={"home": "Ev", "away": "Deplasman"})


def fig_1x2(p, home, away):
    vals = [p["p1"], p["px"], p["p2"]]
    fig = go.Figure(go.Bar(
        x=[f"1 ({home})", "X", f"2 ({away})"], y=vals,
        text=[f"{v:.1%}" for v in vals], textposition="outside",
        marker_color=["#2E86AB", "#A0A0A0", "#E4572E"],
    ))
    fig.update_layout(title="Maç sonucu (1-X-2)", yaxis_tickformat=".0%", yaxis_range=[0, 1],
                      height=380, margin=dict(t=50, b=10))
    return fig


def fig_ou(p):
    lines = [1.5, 2.5, 3.5]
    xs = [str(l) for l in lines]
    fig = go.Figure()
    fig.add_bar(name="Alt", x=xs, y=[p[f"under_{l}"] for l in lines],
                text=[f"{p[f'under_{l}']:.1%}" for l in lines], textposition="inside", marker_color="#7A9E7E")
    fig.add_bar(name="Üst", x=xs, y=[p[f"over_{l}"] for l in lines],
                text=[f"{p[f'over_{l}']:.1%}" for l in lines], textposition="inside", marker_color="#F2A541")
    fig.update_layout(barmode="stack", title="Alt / Üst", yaxis_tickformat=".0%",
                      height=380, margin=dict(t=50, b=10))
    return fig


def fig_matrix(model, home, away, n=6):
    mat = model.score_matrix(home, away)[:n, :n]
    fig = px.imshow(mat, x=list(range(n)), y=list(range(n)), text_auto=".1%",
                    color_continuous_scale="Blues", aspect="equal",
                    labels=dict(x=f"{away} golleri", y=f"{home} golleri", color="Olasılık"))
    fig.update_layout(title="Skor olasılıkları", height=500, margin=dict(t=50))
    return fig


def train(league, n_seasons, half_life, shrinkage):
    model, data = get_model(league, n_seasons, half_life, shrinkage)
    teams = model.teams
    status = (f"✅ **{len(data):,} maç** yüklendi · son maç: {data['date'].max():%d.%m.%Y} · "
              f"{len(teams)} takım. Şimdi takımları seçip **Tahmin et**'e basın.")
    return (status,
            gr.update(choices=teams, value=teams[0]),
            gr.update(choices=teams, value=teams[1] if len(teams) > 1 else teams[0]))


def predict(league, n_seasons, half_life, shrinkage, home, away):
    if not home or not away:
        raise gr.Error("Önce 'Modeli eğit' butonuna basıp takımları seçin.")
    if home == away:
        raise gr.Error("Lütfen farklı iki takım seçin.")
    model, data = get_model(league, n_seasons, half_life, shrinkage)
    p = model.predict(home, away)

    md = f"""## {home} - {away}

| Sonuç | Olasılık | Adil oran |
|---|---|---|
| **1** ({home}) | **{p['p1']:.1%}** | {1 / p['p1']:.2f} |
| **X** (Beraberlik) | **{p['px']:.1%}** | {1 / p['px']:.2f} |
| **2** ({away}) | **{p['p2']:.1%}** | {1 / p['p2']:.2f} |

**Beklenen gol:** {p['xg_home']} - {p['xg_away']} &nbsp;·&nbsp; **En olası skor:** {p['most_likely_score']}
&nbsp;·&nbsp; **KG Var:** {p['btts_yes']:.1%} &nbsp;·&nbsp; **KG Yok:** {p['btts_no']:.1%}

| Çizgi | Alt | Üst |
|---|---|---|
| 1.5 | {p['under_1.5']:.1%} | {p['over_1.5']:.1%} |
| 2.5 | {p['under_2.5']:.1%} | {p['over_2.5']:.1%} |
| 3.5 | {p['under_3.5']:.1%} | {p['over_3.5']:.1%} |

*Eğitim amaçlıdır; sonuçlar kesinlik taşımaz ve bahis tavsiyesi değildir.*"""

    return (md, fig_1x2(p, home, away), fig_ou(p), fig_matrix(model, home, away),
            last_matches(data, home), last_matches(data, away))


def run_backtest(league, n_seasons, half_life, shrinkage, days):
    model, data = get_model(league, n_seasons, half_life, shrinkage)
    start = data["date"].max() - pd.Timedelta(days=int(days))
    res = backtest(data, start, half_life_days=float(half_life), shrinkage=float(shrinkage))
    if res["matches"] == 0:
        raise gr.Error("Bu dönemde test edilecek maç bulunamadı.")
    return f"""### Backtest sonucu (son {int(days)} gün)

| Metrik | Değer |
|---|---|
| Maç sayısı | {res['matches']} |
| Brier skoru | {res['brier']:.3f} |
| Log-loss | {res['log_loss']:.3f} |
| İsabet oranı | {res['accuracy']:.1%} |

Referans: her sonuca 1/3 vermenin Brier skoru ≈ 0.667, log-loss ≈ 1.099. Bunların altı model değer katıyor demektir."""


with gr.Blocks(title="Futbol Olasılık Analizi", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚽ Futbol Olasılık Analizi\nZaman ağırlıklı Poisson modeli · 1-X-2, Alt/Üst, KG Var")

    with gr.Row():
        league = gr.Dropdown(choices=[(name, code) for code, name in LEAGUES.items()],
                             value="T1", label="Lig")
        n_seasons = gr.Slider(1, 5, value=5, step=1, label="Kaç sezon geriye?")
    with gr.Row():
        half_life = gr.Slider(90, 1095, value=365, step=30, label="Ağırlık yarı ömrü (gün)")
        shrinkage = gr.Slider(0, 20, value=5, step=1, label="Küçük örnek düzeltmesi")

    train_btn = gr.Button("1) Veriyi yükle ve modeli eğit", variant="primary")
    status = gr.Markdown()
    settings = [league, n_seasons, half_life, shrinkage]

    with gr.Tabs():
        with gr.Tab("Tahmin"):
            with gr.Row():
                home = gr.Dropdown(choices=[], label="Ev sahibi")
                away = gr.Dropdown(choices=[], label="Deplasman")
            predict_btn = gr.Button("2) Tahmin et", variant="primary")
            summary = gr.Markdown()
            with gr.Row():
                plot_1x2 = gr.Plot()
                plot_ou = gr.Plot()
            plot_matrix = gr.Plot()
        with gr.Tab("Form"):
            with gr.Row():
                form_home = gr.Dataframe(label="Ev sahibi - son 5 maç", interactive=False)
                form_away = gr.Dataframe(label="Deplasman - son 5 maç", interactive=False)
        with gr.Tab("Model Doğruluğu"):
            days = gr.Slider(60, 365, value=180, step=30, label="Test dönemi (gün)")
            bt_btn = gr.Button("Backtest çalıştır")
            bt_out = gr.Markdown()

    train_btn.click(train, settings, [status, home, away])
    predict_btn.click(predict, settings + [home, away],
                      [summary, plot_1x2, plot_ou, plot_matrix, form_home, form_away])
    bt_btn.click(run_backtest, settings + [days], bt_out)

demo.launch()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import LEAGUES, load_matches
from poisson import PoissonModel, backtest

st.set_page_config(page_title="Futbol Olasılık Analizi", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner="Maç verileri yükleniyor...", ttl=6 * 3600)
def get_data(league: str, n_seasons: int) -> pd.DataFrame:
    return load_matches([league], n_seasons)


@st.cache_resource(show_spinner="Model eğitiliyor...")
def get_model(league: str, n_seasons: int, half_life: int, shrinkage: float) -> PoissonModel:
    return PoissonModel(half_life_days=half_life, shrinkage=shrinkage).fit(get_data(league, n_seasons))


st.sidebar.header("Ayarlar")
league = st.sidebar.selectbox("Lig", list(LEAGUES), format_func=lambda c: LEAGUES[c])
n_seasons = st.sidebar.slider("Kaç sezon geriye?", 1, 5, 5)
half_life = st.sidebar.slider("Ağırlık yarı ömrü (gün)", 90, 1095, 365, step=30)
shrinkage = st.sidebar.slider("Küçük örnek düzeltmesi", 0, 20, 5)

try:
    data = get_data(league, n_seasons)
    model = get_model(league, n_seasons, half_life, shrinkage)
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

st.sidebar.caption(f"{len(data):,} maç · son maç: {data['date'].max():%d.%m.%Y}")

st.title("⚽ Futbol Olasılık Analizi")
st.caption(f"{LEAGUES[league]} · Zaman ağırlıklı Poisson modeli")

teams = model.teams
c1, c2 = st.columns(2)
home = c1.selectbox("Ev sahibi", teams, index=0)
away = c2.selectbox("Deplasman", teams, index=min(1, len(teams) - 1))

if home == away:
    st.warning("Lütfen farklı iki takım seçin.")
    st.stop()

p = model.predict(home, away)

tab1, tab2, tab3, tab4 = st.tabs(["Olasılıklar", "Skor Matrisi", "Form", "Model Doğruluğu"])

with tab1:
    m1, m2, m3 = st.columns(3)
    for col, label, key in [(m1, f"{home} (1)", "p1"), (m2, "Beraberlik (X)", "px"), (m3, f"{away} (2)", "p2")]:
        col.metric(label, f"{p[key]:.1%}")
        col.caption(f"Adil oran: {1 / p[key]:.2f}")

    x1, x2, x3 = st.columns(3)
    x1.metric("Beklenen gol (ev)", p["xg_home"])
    x2.metric("Beklenen gol (dep.)", p["xg_away"])
    x3.metric("En olası skor", p["most_likely_score"])

    left, right = st.columns(2)

    fig = go.Figure(go.Bar(
        x=["1", "X", "2"], y=[p["p1"], p["px"], p["p2"]],
        text=[f"{v:.1%}" for v in (p["p1"], p["px"], p["p2"])], textposition="outside",
        marker_color=["#2E86AB", "#A0A0A0", "#E4572E"],
    ))
    fig.update_layout(title="Maç sonucu (1-X-2)", yaxis_tickformat=".0%", yaxis_range=[0, 1],
                      height=360, margin=dict(t=50, b=10))
    left.plotly_chart(fig, use_container_width=True)

    lines = [1.5, 2.5, 3.5]
    fig2 = go.Figure()
    fig2.add_bar(name="Alt", x=[str(l) for l in lines], y=[p[f"under_{l}"] for l in lines],
                 text=[f"{p[f'under_{l}']:.1%}" for l in lines], textposition="inside", marker_color="#7A9E7E")
    fig2.add_bar(name="Üst", x=[str(l) for l in lines], y=[p[f"over_{l}"] for l in lines],
                 text=[f"{p[f'over_{l}']:.1%}" for l in lines], textposition="inside", marker_color="#F2A541")
    fig2.update_layout(barmode="stack", title="Alt / Üst", yaxis_tickformat=".0%",
                       height=360, margin=dict(t=50, b=10))
    right.plotly_chart(fig2, use_container_width=True)

    b1, b2 = st.columns(2)
    b1.metric("KG Var", f"{p['btts_yes']:.1%}")
    b2.metric("KG Yok", f"{p['btts_no']:.1%}")

with tab2:
    n = 6
    mat = model.score_matrix(home, away)[:n, :n]
    fig3 = px.imshow(
        mat, x=list(range(n)), y=list(range(n)), text_auto=".1%",
        color_continuous_scale="Blues", aspect="equal",
        labels=dict(x=f"{away} golleri", y=f"{home} golleri", color="Olasılık"),
    )
    fig3.update_layout(height=520, margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Her hücre, o kesin skorun gerçekleşme olasılığıdır.")

with tab3:
    def last_matches(team: str, n: int = 5) -> pd.DataFrame:
        d = data[(data["home"] == team) | (data["away"] == team)].tail(n).copy()
        d["Skor"] = d["home_goals"].astype(str) + " - " + d["away_goals"].astype(str)
        d["Tarih"] = d["date"].dt.strftime("%d.%m.%Y")
        return d[["Tarih", "home", "Skor", "away"]].rename(columns={"home": "Ev", "away": "Deplasman"})

    f1, f2 = st.columns(2)
    f1.subheader(home)
    f1.dataframe(last_matches(home), hide_index=True, use_container_width=True)
    f2.subheader(away)
    f2.dataframe(last_matches(away), hide_index=True, use_container_width=True)

with tab4:
    st.write("Model, son 180 gündeki her maç gününde yalnızca o günden önceki verilerle yeniden eğitilir.")
    if st.button("Backtest çalıştır"):
        with st.spinner("Hesaplanıyor (biraz sürebilir)..."):
            start = data["date"].max() - pd.Timedelta(days=180)
            res = backtest(data, start_date=start, half_life_days=half_life, shrinkage=shrinkage)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Maç sayısı", res["matches"])
        r2.metric("Brier skoru", f"{res['brier']:.3f}")
        r3.metric("Log-loss", f"{res['log_loss']:.3f}")
        r4.metric("İsabet oranı", f"{res['accuracy']:.1%}")
        st.caption("Referans: her sonuca 1/3 vermenin Brier skoru ≈ 0.667, log-loss ≈ 1.099.")

st.divider()
st.caption("Bu uygulama eğitim amaçlıdır; sonuçlar kesinlik taşımaz ve bahis tavsiyesi değildir.")

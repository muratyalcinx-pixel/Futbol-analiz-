"""
app.py
Streamlit arayüzü: lig/takım seç, 1-X-2 ve Alt/Üst olasılıklarını gör,
modeli geçmiş maçlarda ve bahis oranlarıyla kıyasla.

Çalıştırma:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import LEAGUES, load_matches
from poisson import PoissonModel, backtest_detailed, summarize

st.set_page_config(page_title="Futbol Olasılık Analizi", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------- önbellek
@st.cache_data(show_spinner="Maç verileri yükleniyor...", ttl=6 * 3600)
def get_data(league: str, n_seasons: int) -> pd.DataFrame:
    return load_matches([league], n_seasons)


@st.cache_resource(show_spinner="Model eğitiliyor...")
def get_model(league: str, n_seasons: int, half_life: int, shrinkage: float) -> PoissonModel:
    return PoissonModel(half_life_days=half_life, shrinkage=shrinkage).fit(get_data(league, n_seasons))


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def run_backtest(league: str, n_seasons: int, half_life: int, shrinkage: float, days: int) -> pd.DataFrame:
    data = get_data(league, n_seasons)
    start = data["date"].max() - pd.Timedelta(days=days)
    return backtest_detailed(data, start, half_life_days=half_life, shrinkage=shrinkage)


# ----------------------------------------------------------------- kenar çubuğu
st.sidebar.header("Ayarlar")
league = st.sidebar.selectbox("Lig", list(LEAGUES), format_func=lambda c: LEAGUES[c])
n_seasons = st.sidebar.slider("Kaç sezon geriye?", 1, 5, 5)
half_life = st.sidebar.slider("Ağırlık yarı ömrü (gün)", 90, 1095, 365, step=30,
                              help="Düşük değer = son maçlara daha çok önem")
shrinkage = st.sidebar.slider("Küçük örnek düzeltmesi", 0, 20, 5,
                              help="Az maç oynayan takımları lig ortalamasına çeker")

try:
    data = get_data(league, n_seasons)
    model = get_model(league, n_seasons, half_life, shrinkage)
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

st.sidebar.caption(f"{len(data):,} maç · son maç: {data['date'].max():%d.%m.%Y}")

# ------------------------------------------------------------------- başlık
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

# ------------------------------------------------------------- olasılıklar
with tab1:
    m1, m2, m3 = st.columns(3)
    for col, label, key in [(m1, f"{home} (1)", "p1"), (m2, "Beraberlik (X)", "px"), (m3, f"{away} (2)", "p2")]:
        col.metric(label, f"{p[key]:.1%}")
        col.caption(f"Adil oran: {1 / p[key]:.2f}")

    mat_all = model.score_matrix(home, away)
    top = sorted(((mat_all[i, j], f"{i}-{j}") for i in range(6) for j in range(6)), reverse=True)[:3]

    x1, x2, x3 = st.columns(3)
    x1.metric("Beklenen gol (ev)", p["xg_home"])
    x2.metric("Beklenen gol (dep.)", p["xg_away"])
    x3.metric("En olası 3 skor", " · ".join(s for _, s in top))
    x3.caption("  ".join(f"{s}: {v:.1%}" for v, s in top))

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

# ------------------------------------------------------------ skor matrisi
with tab2:
    n = 6
    mat = mat_all[:n, :n]
    fig3 = px.imshow(
        mat, x=list(range(n)), y=list(range(n)), text_auto=".1%",
        color_continuous_scale="Blues", aspect="equal",
        labels=dict(x=f"{away} golleri", y=f"{home} golleri", color="Olasılık"),
    )
    fig3.update_layout(height=520, margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Her hücre, o kesin skorun gerçekleşme olasılığıdır.")

# -------------------------------------------------------------------- form
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

# ----------------------------------------------------------------- backtest
with tab4:
    st.write(
        "Model, test dönemindeki her maç gününde **yalnızca o günden önceki verilerle** "
        "yeniden eğitilir (geleceği görmez). Sonuçlar aynı maçlardaki bahis oranlarından "
        "çıkan olasılıklarla kıyaslanır."
    )
    days = st.slider("Test dönemi (gün)", 60, 730, 365, step=30,
                     help="Uzun dönem = daha güvenilir sonuç, ama daha yavaş hesaplanır")

    if st.button("Backtest çalıştır"):
        with st.spinner("Hesaplanıyor (biraz sürebilir)..."):
            bt = run_backtest(league, n_seasons, half_life, shrinkage, days)
        s = summarize(bt)

        if s["matches"] == 0:
            st.warning("Bu dönemde test edilecek maç bulunamadı. Test dönemini uzatmayı dene.")
        else:
            st.subheader("Genel sonuç")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Maç sayısı", s["matches"])
            g2.metric("Brier", f"{s['brier']:.3f}")
            g3.metric("Log-loss", f"{s['log_loss']:.3f}")
            g4.metric("İsabet", f"{s['accuracy']:.1%}")
            st.caption("Referans: her sonuca 1/3 vermenin Brier skoru ≈ 0.667, log-loss ≈ 1.099 (düşük = iyi).")

            st.subheader("Model vs bahis oranları")
            if s["odds_matches"] == 0:
                st.info("Bu lig/sezonlar için oran verisi bulunamadı, kıyaslama yapılamıyor.")
            else:
                rows = [
                    ["1-X-2 Brier (düşük iyi)", f"{s['model_brier_odds']:.3f}", f"{s['market_brier']:.3f}", "0.667"],
                    ["1-X-2 Log-loss (düşük iyi)", f"{s['model_ll_odds']:.3f}", f"{s['market_ll']:.3f}", "1.099"],
                    ["İsabet oranı", f"{s['model_acc_odds']:.1%}", f"{s['market_acc']:.1%}", "—"],
                ]
                if s["ou_matches"] > 0:
                    rows.append(["Alt/Üst 2.5 Brier (düşük iyi)", f"{s['model_ou_brier']:.3f}",
                                 f"{s['market_ou_brier']:.3f}", "0.250"])
                table = pd.DataFrame(rows, columns=["Metrik", "Model", "Bahis oranları", "Rastgele"])
                st.dataframe(table, hide_index=True, use_container_width=True)
                st.caption(
                    f"{s['odds_matches']} maçta oran vardı (Alt/Üst: {s['ou_matches']} maç). "
                    f"Oranlardaki ortalama bahisçi marjı ≈ {s['avg_overround']:.1%}, kıyaslamadan önce arındırıldı."
                )

                share = s["beats_market_share"]
                st.metric("Gerçek sonuca oranlardan daha yüksek olasılık veren maçların payı", f"{share:.1%}")

                diff = s["model_brier_odds"] - s["market_brier"]
                if diff > 0.005:
                    st.info("Bahis oranları bu dönemde modelden daha iyi tahmin etmiş. Bu beklenen bir durum: "
                            "oranlar sakatlık, kadro ve haber bilgisini de içerir, bu model içermez.")
                elif diff < -0.005:
                    st.success("Model bu dönemde oranlardan daha düşük (daha iyi) Brier skoru aldı. "
                               "Yine de test süresini uzatıp başka liglerde de doğrula; kısa dönemlerde şans payı büyüktür.")
                else:
                    st.info("Model ile bahis oranları bu dönemde pratikte aynı seviyede.")

            with st.expander("Maç maç sonuçlar"):
                show = bt.copy()
                show["date"] = show["date"].dt.strftime("%d.%m.%Y")
                st.dataframe(show.round(3), hide_index=True, use_container_width=True)
                st.download_button("CSV olarak indir", show.to_csv(index=False).encode("utf-8"),
                                   file_name="backtest.csv", mime="text/csv")

st.divider()
st.caption("Bu uygulama eğitim amaçlıdır; sonuçlar kesinlik taşımaz ve bahis tavsiyesi değildir.")

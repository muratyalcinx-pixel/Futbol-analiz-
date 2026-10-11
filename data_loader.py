"""
data_loader.py
football-data.co.uk üzerinden son N sezonun maç verilerini indirir,
yerel olarak önbelleğe alır ve tek bir DataFrame olarak döndürür.

Kullanım:
    from data_loader import load_matches
    df = load_matches(leagues=["T1", "E0"], n_seasons=5)
"""

from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
RAW_DIR = Path("data/raw")

# Lig kodları (football-data.co.uk)
LEAGUES = {
    "T1": "Türkiye Süper Lig",
    "E0": "İngiltere Premier Lig",
    "SP1": "İspanya La Liga",
    "D1": "Almanya Bundesliga",
    "I1": "İtalya Serie A",
    "F1": "Fransa Ligue 1",
}

# Her yerde aynı isimleri kullanmak için sütun eşleştirme
RENAME = {
    "Div": "league",
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",  # H / D / A
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "B365H": "odds_1",
    "B365D": "odds_x",
    "B365A": "odds_2",
    "B365>2.5": "odds_over25",
    "B365<2.5": "odds_under25",
}


def season_codes(n_seasons: int = 5) -> list[str]:
    """Son n sezonun kodlarını üretir (ör. '2526'). Devam eden sezon dahildir."""
    today = date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    codes = []
    for i in range(n_seasons):
        y = start_year - i
        codes.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return codes[::-1]  # eskiden yeniye


def _download_csv(season: str, league: str, refresh: bool = False) -> pd.DataFrame | None:
    """Tek sezon + lig CSV'sini indirir; varsa önbellekten okur."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{league}_{season}.csv"

    if cache_path.exists() and not refresh:
        text = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        url = BASE_URL.format(season=season, league=league)
        try:
            resp = requests.get(url, timeout=30)
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


def load_matches(
    leagues: list[str] | None = None,
    n_seasons: int = 5,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Seçili ligler için son n sezonun maçlarını döndürür.

    Parametreler
    ------------
    leagues   : Lig kodları listesi. Varsayılan ["T1"].
    n_seasons : Kaç sezon geriye gidileceği.
    refresh   : True ise önbelleği yok sayıp yeniden indirir.
    """
    leagues = leagues or ["T1"]
    frames = []

    for league in leagues:
        for season in season_codes(n_seasons):
            df = _download_csv(season, league, refresh=refresh)
            if df is not None:
                frames.append(df)

    if not frames:
        raise RuntimeError("Hiç veri indirilemedi. İnternet bağlantısını kontrol edin.")

    data = pd.concat(frames, ignore_index=True)

    # Yalnızca mevcut sütunları al ve yeniden adlandır
    keep = [c for c in RENAME if c in data.columns] + ["season"]
    data = data[keep].rename(columns=RENAME)

    # Tarih: bazı sezonlarda dd/mm/yy, bazılarında dd/mm/yyyy
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")

    # Oynanmamış / eksik maçları ayıkla
    data = data.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])
    data[["home_goals", "away_goals"]] = data[["home_goals", "away_goals"]].astype(int)

    # Alt/Üst 2.5 hedef değişkeni ve toplam gol
    data["total_goals"] = data["home_goals"] + data["away_goals"]
    data["over25"] = (data["total_goals"] > 2.5).astype(int)

    return data.sort_values("date").reset_index(drop=True)


FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

FIXTURE_RENAME = {
    "Div": "league",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "B365H": "odds_1",
    "B365D": "odds_x",
    "B365A": "odds_2",
    "B365>2.5": "odds_over25",
    "B365<2.5": "odds_under25",
}


def load_fixtures(league: str) -> pd.DataFrame:
    """
    football-data.co.uk'nin yaklaşan maç listesini (fixtures.csv) indirir ve
    seçili lige göre filtreler. Dosya canlı değil, periyodik güncellenir.
    Sütunlar: date, time, home, away (+ varsa oranlar).
    """
    resp = requests.get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.content.decode("latin-1")))
    df = df.dropna(how="all")

    if "Div" in df.columns:
        df = df[df["Div"] == league]

    keep = [c for c in FIXTURE_RENAME if c in df.columns]
    df = df[keep].rename(columns=FIXTURE_RENAME)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "home", "away"])

    sort_cols = ["date"] + (["time"] if "time" in df.columns else [])
    return df.sort_values(sort_cols).reset_index(drop=True)


if __name__ == "__main__":
    df = load_matches(leagues=["T1"], n_seasons=5)
    print(df.shape)
    print(df.tail())
    df.to_csv("data/processed_matches.csv", index=False)

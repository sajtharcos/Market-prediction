import argparse
import warnings
from datetime import date, timedelta
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ── Argumentumok ────────────────────────────────────────────────────────
def next_saturday():
    d = date.today()
    days = (5 - d.weekday()) % 7
    return (d + timedelta(days=days or 7)).isoformat()


p = argparse.ArgumentParser()
p.add_argument("--file", required=True, help="Excel/ODS fájl neve")
p.add_argument("--next_date", default=next_saturday(), help="Pl. 2026-04-13")
p.add_argument("--helyszin", default=None, help="Ha üres, akkor minden ismert helyszínre készít becslést")
p.add_argument("--weather", default="felhős")
p.add_argument("--temp", type=float, default=20.0)
p.add_argument("--rain", default="nem")
p.add_argument("--holiday", default="nem")
p.add_argument("--visitors", type=float, default=None)
args = p.parse_args()


# ── Excel/ODS beolvasás ────────────────────────────────────────────────
def load_table(filepath: str) -> pd.DataFrame:
    ext = filepath.lower().split(".")[-1]

    try:
        if ext == "ods":
            xls = pd.ExcelFile(filepath, engine="odf")
        else:
            xls = pd.ExcelFile(filepath)
    except Exception as e:
        print(f"❌ Nem sikerült megnyitni a fájlt: {filepath}")
        print(f"Hiba: {e}")
        sys.exit(1)

    if "Adatok" in xls.sheet_names:
        sheet_to_use = "Adatok"
    else:
        sheet_to_use = xls.sheet_names[0]

    try:
        if ext == "ods":
            df_ = pd.read_excel(filepath, sheet_name=sheet_to_use, engine="odf")
        else:
            df_ = pd.read_excel(filepath, sheet_name=sheet_to_use)
    except Exception as e:
        print(f"❌ Nem sikerült beolvasni a '{sheet_to_use}' munkalapot.")
        print(f"Hiba: {e}")
        sys.exit(1)

    return df_


df = load_table(args.file)


# ── Oszlopnevek egységesítése ───────────────────────────────────────────
def normalize_colname(col):
    c = str(col).strip().lower()
    c = c.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o")
    c = c.replace("ö", "o").replace("ő", "o").replace("ú", "u").replace("ü", "u").replace("ű", "u")
    c = c.replace("(", "").replace(")", "")
    c = c.replace("?", "").replace(".", "")
    c = c.replace("°", "")
    c = c.replace("/", " ")
    c = c.replace("-", " ")
    c = " ".join(c.split())
    return c


df.columns = [normalize_colname(c) for c in df.columns]

column_map = {
    "datum": "datum",
    "helyszin": "helyszin",
    "termek": "termek",
    "kategoria": "kategoria",
    "ar ft": "ar",
    "ar": "ar",
    "eladott menny": "mennyiseg",
    "mennyiseg": "mennyiseg",
    "ido perc": "idopercek",
    "ido": "idopercek",
    "idojaras": "idojaras",
    "hofok c": "hofok",
    "hofok": "hofok",
    "eso": "eso",
    "unnep": "unnep",
    "vasarlok szama kb": "vasarlok",
    "vasarlok szama": "vasarlok",
    "vasarlok": "vasarlok",
}

df = df.rename(columns={c: column_map[c] for c in df.columns if c in column_map})

required_cols = [
    "datum", "termek", "kategoria", "ar", "mennyiseg", "idopercek",
    "idojaras", "hofok", "eso", "unnep", "vasarlok"
]
missing = [c for c in required_cols if c not in df.columns]

if "helyszin" not in df.columns:
    df["helyszin"] = "ismeretlen"

if missing:
    print("❌ Hiányzó szükséges oszlop(ok):", ", ".join(missing))
    print("A felismert oszlopok:", list(df.columns))
    sys.exit(1)


# ── Tisztítás ────────────────────────────────────────────────────────────
for col in ["termek", "kategoria", "idojaras", "eso", "unnep", "helyszin"]:
    df[col] = df[col].astype(str).str.strip().str.lower()

args.weather = str(args.weather).strip().lower()
args.rain = str(args.rain).strip().lower()
args.holiday = str(args.holiday).strip().lower()
if args.helyszin is not None:
    args.helyszin = str(args.helyszin).strip().lower()

df = df.dropna(subset=["datum", "termek", "mennyiseg"])

try:
    df["datum"] = pd.to_datetime(df["datum"])
except Exception:
    df["datum"] = pd.to_datetime(df["datum"], format="%Y.%m.%d", errors="coerce")

df = df.dropna(subset=["datum"])

df["ar"] = pd.to_numeric(df["ar"], errors="coerce")
df["mennyiseg"] = pd.to_numeric(df["mennyiseg"], errors="coerce")
df["hofok"] = pd.to_numeric(df["hofok"], errors="coerce")
df["vasarlok"] = pd.to_numeric(df["vasarlok"], errors="coerce")
df["idopercek"] = pd.to_numeric(df["idopercek"], errors="coerce")

df["ar"] = df["ar"].fillna(df["ar"].median())
df["mennyiseg"] = df["mennyiseg"].fillna(0)
df["hofok"] = df["hofok"].fillna(18)
df["vasarlok"] = df["vasarlok"].fillna(df["vasarlok"].median() if df["vasarlok"].notna().any() else 70)
df["idopercek"] = df["idopercek"].fillna(240)

if len(df) < 5:
    print("⚠️ Kevés adat! Legalább 5–8 sor kell a minimális predikcióhoz.")
    sys.exit(1)

print(f"📂 Betöltve: {len(df)} sor, {df['termek'].nunique()} termék, {df['datum'].nunique()} piac-nap\n")


# ── Feature engineering ─────────────────────────────────────────────────
df["ho"] = df["datum"].dt.month
df["het"] = df["datum"].dt.isocalendar().week.astype(int)

feature_cols = [
    "termek", "kategoria", "helyszin", "ar", "hofok",
    "idojaras", "eso", "unnep", "vasarlok", "ho", "het"
]

categorical_cols = ["termek", "kategoria", "helyszin", "idojaras", "eso", "unnep"]
numeric_cols = ["ar", "hofok", "vasarlok", "ho", "het"]

X = df[feature_cols].copy()
y = df["mennyiseg"].copy()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])


# ── Modell ──────────────────────────────────────────────────────────────
if len(df) >= 10:
    cv = min(5, max(2, len(df) // 2))
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()
    print(f"📊 Keresztvalidáció (CV MAE): ±{cv_mae:.2f} egység átlagos hiba\n")

pipeline.fit(X, y)


# ── Predikció ───────────────────────────────────────────────────────────
try:
    next_dt = pd.to_datetime(args.next_date)
except Exception:
    print("❌ Hibás dátumformátum. Használj ilyet: 2026-04-13")
    sys.exit(1)

next_mo = next_dt.month
next_wk = int(next_dt.isocalendar().week)

known_locations = sorted(df["helyszin"].unique())

if args.helyszin:
    helyszinek_lista = [args.helyszin]
    if args.helyszin not in known_locations:
        print(f"⚠️ Ismeretlen helyszín: {args.helyszin}")
        print("A modell ettől még készít becslést, de a helyszínhatás csak korlátozottan lesz megbízható.")
else:
    helyszinek_lista = known_locations

if args.visitors is None:
    similar = df[df["idojaras"] == args.weather]
    if len(similar) > 0:
        pred_visitors = float(similar["vasarlok"].mean())
    else:
        pred_visitors = float(df["vasarlok"].mean())
else:
    pred_visitors = float(args.visitors)

results = []

# Ha ismert helyszínt adtál meg, a terméklistát abból vesszük,
# ha új helyszínt, akkor az összes ismert termékből építünk scenariót.
if args.helyszin and args.helyszin in known_locations:
    termekek = df[df["helyszin"] == args.helyszin]["termek"].unique()
else:
    termekek = df["termek"].unique()

for helyszin in helyszinek_lista:
    for termek in termekek:
        # Ha van ilyen termék + helyszín múltban, azt preferáljuk
        term_rows = df[(df["termek"] == termek) & (df["helyszin"] == helyszin)]

        # Ha az új helyszín ismeretlen vagy nincs ott még ilyen termék, fallback az összes ilyen termékre
        if len(term_rows) == 0:
            term_rows = df[df["termek"] == termek]

        if len(term_rows) == 0:
            continue

        avg_ar = float(term_rows["ar"].mean())
        kat = str(term_rows["kategoria"].mode().iloc[0]).strip().lower()

        x_new = pd.DataFrame([{
            "termek": termek,
            "kategoria": kat,
            "helyszin": helyszin,
            "ar": avg_ar,
            "hofok": args.temp,
            "idojaras": args.weather,
            "eso": args.rain,
            "unnep": args.holiday,
            "vasarlok": pred_visitors,
            "ho": next_mo,
            "het": next_wk
        }])

        pred = max(0.0, float(pipeline.predict(x_new)[0]))

        n_obs = len(term_rows)
        conf = "alacsony" if n_obs < 4 else "közepes" if n_obs < 8 else "magas"

        if helyszin not in known_locations:
            conf = "nagyon alacsony" if conf == "alacsony" else conf

        results.append({
            "Helyszín": helyszin,
            "Termék": termek,
            "Kategória": kat,
            "Jósolt menny.": round(pred, 1),
            "Átlag ár (Ft)": round(avg_ar),
            "Becsült bevétel (Ft)": round(pred * avg_ar),
            "Adatpontok": n_obs,
            "Megbízhatóság": conf,
        })

results_df = pd.DataFrame(results)

if results_df.empty:
    print("⚠️ Nem készült predikció.")
    print("A fájlban található értékek:")
    print("Időjárás:", sorted(df["idojaras"].unique()))
    print("Eső:", sorted(df["eso"].unique()))
    print("Ünnep:", sorted(df["unnep"].unique()))
    print("Helyszín:", known_locations)
    sys.exit(1)

results_df = results_df.sort_values(["Helyszín", "Jósolt menny."], ascending=[True, False])


# ── Kiírás ──────────────────────────────────────────────────────────────
print("═" * 70)
print(f"  PREDIKCIÓ — {args.next_date}")
print(f"  Időjárás: {args.weather}, {args.temp}°C, Eső: {args.rain}, Ünnep: {args.holiday}")
print(f"  Becsült látogatók: {pred_visitors:.0f} fő")
print("═" * 70)

for h in helyszinek_lista:
    sub = results_df[results_df["Helyszín"] == h]
    if sub.empty:
        continue

    print(f"\n  📍 {h}")
    print(sub.drop(columns="Helyszín").to_string(index=False))
    print(f"  → Becsült bevétel: {sub['Becsült bevétel (Ft)'].sum():,} Ft")

print(f"\n{'═' * 70}")
print(f"  Összes becsült bevétel (minden helyszín): {results_df['Becsült bevétel (Ft)'].sum():,} Ft")


# ── Feature importance ──────────────────────────────────────────────────
try:
    ohe = pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = list(cat_feature_names) + numeric_cols

    importances = pipeline.named_steps["model"].feature_importances_
    imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

    print("\n📌 Legfontosabb tényezők:")
    for feat, val in imp.head(10).items():
        print(f"   {feat:<30} {val * 100:.1f}%")
except Exception:
    pass


# ── Mentés ───────────────────────────────────────────────────────────────
out = "piac_predikcio_eredmeny.xlsx"
results_df.to_excel(out, index=False)
print(f"\n✅ Eredmény elmentve: {out}")

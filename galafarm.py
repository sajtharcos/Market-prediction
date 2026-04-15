import argparse
import warnings
from datetime import date
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ── Argumentumok ────────────────────────────────────────────────────────
def default_date():
    return date.today().isoformat()


p = argparse.ArgumentParser()
p.add_argument("--file", required=True, help="Excel/ODS fájl neve")
p.add_argument("--next_date", default=default_date(), help="Pl. 2026-04-13")
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

    sheet_candidates = ["Adatok", "adatok", "Sheet1", "Munka1"]
    sheet_to_use = None

    for s in sheet_candidates:
        if s in xls.sheet_names:
            sheet_to_use = s
            break

    if sheet_to_use is None:
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
    repl = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ö": "o", "ő": "o",
        "ú": "u", "ü": "u", "ű": "u", "°": ""
    }
    for k, v in repl.items():
        c = c.replace(k, v)

    for ch in ["(", ")", "?", ".", ",", ":", ";"]:
        c = c.replace(ch, "")

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
    "eladott mennyiseg": "mennyiseg",
    "mennyiseg": "mennyiseg",
    "ido perc": "idopercek",
    "ido percek": "idopercek",
    "idopercek": "idopercek",
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
    "datum", "termek", "kategoria", "ar", "mennyiseg",
    "idopercek", "idojaras", "hofok", "eso", "unnep", "vasarlok"
]

if "helyszin" not in df.columns:
    df["helyszin"] = "ismeretlen"

missing = [c for c in required_cols if c not in df.columns]
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

parsed = pd.to_datetime(df["datum"], errors="coerce")
if parsed.isna().any():
    parsed2 = pd.to_datetime(df["datum"], format="%Y.%m.%d", errors="coerce")
    parsed = parsed.fillna(parsed2)
df["datum"] = parsed
df = df.dropna(subset=["datum"])

for col in ["ar", "mennyiseg", "hofok", "vasarlok", "idopercek"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["ar"] = df["ar"].fillna(df["ar"].median() if df["ar"].notna().any() else 0)
df["mennyiseg"] = df["mennyiseg"].fillna(0)
df["hofok"] = df["hofok"].fillna(df["hofok"].median() if df["hofok"].notna().any() else 18)
df["vasarlok"] = df["vasarlok"].fillna(df["vasarlok"].median() if df["vasarlok"].notna().any() else 70)
df["idopercek"] = df["idopercek"].fillna(df["idopercek"].median() if df["idopercek"].notna().any() else 240)

if len(df) < 5:
    print("⚠️ Kevés adat! Legalább 5–8 sor kell a minimális predikcióhoz.")
    sys.exit(1)

print(f"📂 Betöltve: {len(df)} sor, {df['termek'].nunique()} termék, {df['datum'].nunique()} piac-nap\n")


# ── Feature engineering ─────────────────────────────────────────────────
df["ho"] = df["datum"].dt.month
df["het"] = df["datum"].dt.isocalendar().week.astype(int)
df["nap"] = df["datum"].dt.weekday
df["hetvege"] = df["nap"].isin([5, 6]).astype(int)
df["jo_ido"] = ((df["hofok"] >= 15) & (df["eso"] == "nem")).astype(int)


# ── ELADÁS MODELL ───────────────────────────────────────────────────────
feature_cols = [
    "termek", "kategoria", "helyszin", "ar", "hofok",
    "idojaras", "eso", "unnep", "vasarlok", "ho", "het", "nap", "hetvege", "jo_ido"
]

categorical_cols = ["termek", "kategoria", "helyszin", "idojaras", "eso", "unnep"]
numeric_cols = ["ar", "hofok", "vasarlok", "ho", "het", "nap", "hetvege", "jo_ido"]

X = df[feature_cols].copy()
y = df["mennyiseg"].copy()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

sales_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", sales_model)
])

if len(df) >= 10:
    cv = min(5, max(2, len(df) // 2))
    try:
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
        cv_mae = -cv_scores.mean()
        print(f"📊 Keresztvalidáció (CV MAE): ±{cv_mae:.2f} egység átlagos hiba\n")
    except Exception as e:
        print(f"⚠️ A keresztvalidáció most kimaradt: {e}\n")

pipeline.fit(X, y)


# ── LÁTOGATÓSZÁM MODELL ────────────────────────────────────────────────
visitor_features = [
    "helyszin", "idojaras", "eso", "unnep", "hofok",
    "ho", "het", "nap", "hetvege", "jo_ido"
]

X_vis = df[visitor_features].copy()
y_vis = df["vasarlok"].copy()

visitor_categorical_cols = ["helyszin", "idojaras", "eso", "unnep"]
visitor_numeric_cols = ["hofok", "ho", "het", "nap", "hetvege", "jo_ido"]

visitor_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), visitor_categorical_cols),
        ("num", "passthrough", visitor_numeric_cols),
    ]
)

visitor_model = Pipeline([
    ("prep", visitor_preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    ))
])

visitor_model.fit(X_vis, y_vis)


# ── Predikció ───────────────────────────────────────────────────────────
try:
    next_dt = pd.to_datetime(args.next_date)
except Exception:
    print("❌ Hibás dátumformátum. Használj ilyet: 2026-04-13")
    sys.exit(1)

next_mo = next_dt.month
next_wk = int(next_dt.isocalendar().week)
next_day = int(next_dt.weekday())
next_weekend = 1 if next_day in [5, 6] else 0
next_good_weather = 1 if (args.temp >= 15 and args.rain == "nem") else 0

known_locations = sorted(df["helyszin"].unique())

if args.helyszin:
    helyszinek_lista = [args.helyszin]
    if args.helyszin not in known_locations:
        print(f"⚠️ Ismeretlen helyszín: {args.helyszin}")
        print("A modell ettől még készít becslést, de a helyszínhatás korlátozottan lesz megbízható.")
else:
    helyszinek_lista = known_locations


# ── Látogatószám becslése ──────────────────────────────────────────────
if args.visitors is None:
    vis_location = args.helyszin if args.helyszin is not None else known_locations[0]

    x_vis_new = pd.DataFrame([{
        "helyszin": vis_location,
        "idojaras": args.weather,
        "eso": args.rain,
        "unnep": args.holiday,
        "hofok": args.temp,
        "ho": next_mo,
        "het": next_wk,
        "nap": next_day,
        "hetvege": next_weekend,
        "jo_ido": next_good_weather
    }])

    pred_visitors = max(1.0, float(visitor_model.predict(x_vis_new)[0]))
else:
    pred_visitors = float(args.visitors)


# ── Eredmények ──────────────────────────────────────────────────────────
results = []

if args.helyszin and args.helyszin in known_locations:
    termekek = df[df["helyszin"] == args.helyszin]["termek"].unique()
else:
    termekek = df["termek"].unique()

for helyszin in helyszinek_lista:
    for termek in termekek:
        term_rows = df[(df["termek"] == termek) & (df["helyszin"] == helyszin)]

        if len(term_rows) == 0:
            term_rows = df[df["termek"] == termek]

        if len(term_rows) == 0:
            continue

        avg_ar = float(term_rows["ar"].mean())

        kat_mode = term_rows["kategoria"].mode()
        kat = str(kat_mode.iloc[0]).strip().lower() if len(kat_mode) > 0 else "ismeretlen"

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
            "het": next_wk,
            "nap": next_day,
            "hetvege": next_weekend,
            "jo_ido": next_good_weather
        }])

        pred = max(0.0, float(pipeline.predict(x_new)[0]))

        n_obs = len(term_rows)
        conf = "alacsony" if n_obs < 4 else "közepes" if n_obs < 8 else "magas"

        if helyszin not in known_locations and conf == "alacsony":
            conf = "nagyon alacsony"

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
print(f"  PREDIKCIÓ — {next_dt.date()}")
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
        print(f"   {feat:<35} {val * 100:.1f}%")
except Exception:
    pass


# ── Mentés ───────────────────────────────────────────────────────────────
out = "piac_predikcio_eredmeny.xlsx"
results_df.to_excel(out, index=False)
print(f"\n✅ Eredmény elmentve: {out}")

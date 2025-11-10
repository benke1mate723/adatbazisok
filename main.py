# main.py
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd
import psycopg2

# ================== Beállítások ==================
BASE = Path(__file__).parent.resolve()
ADAGOK_STEMS = ["Adagok"]
HUTOPANELEK_STEMS = ["Hűtőpanelek", "Hutopanelek"]

PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDB   = os.getenv("PGDATABASE", "beadando")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASS = os.getenv("PGPASSWORD", "Traktor1t")

SCHEMA = "tryout4"
def T(name: str) -> str:
    return f"{SCHEMA}.{name}"

def log(msg: str) -> None:
    print(msg, flush=True)

# ================== I/O segédek ==================
def df_to_copy_buffer(df: pd.DataFrame, sep: str = "\t") -> io.StringIO:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, sep=sep, na_rep="")
    buf.seek(0)
    return buf

def _sniff_sep(text_sample: str) -> str:
    return max([",",";","\t","|"], key=lambda ch: text_sample.count(ch))

def _try_read_csv(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read(65536)
    sample = raw.decode("utf-8", errors="ignore")
    sep = _sniff_sep(sample)
    for enc in ("utf-8-sig", "utf-8", "cp1250", "latin1", "iso-8859-2"):
        try:
            return pd.read_csv(
                path, sep=sep, encoding=enc, engine="python",
                dtype=str, skip_blank_lines=True, keep_default_na=False
            )
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return pd.read_csv(
        path, sep=sep, encoding="utf-8", engine="python",
        dtype=str, skip_blank_lines=True, keep_default_na=False
    )

def _read_table_any(base: Path, stems: List[str]) -> tuple[pd.DataFrame, str]:
    for stem in stems:
        xlsx = base / f"{stem}.xlsx"
        csv  = base / f"{stem}.csv"
        if xlsx.exists():
            return pd.read_excel(xlsx, dtype=str), xlsx.name
        if csv.exists():
            return _try_read_csv(csv), csv.name
    raise FileNotFoundError(f"Nincs ilyen fájl a mappában ({base}): {stems}.*")

# ================== Normalizálók ==================
def _norm_time(t: str | None) -> str:
    t = ("" if t is None else str(t)).strip().replace(".", ":")
    m = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", t)
    if not m:
        return "00:00:00"
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def _norm_date(d: str | None) -> str | None:
    d = ("" if d is None else str(d)).strip()
    if not d:
        return None
    d = re.sub(r"[^\d./\-]", "", d)
    d = re.sub(r"[./]", "-", d)
    d = re.sub(r"-{2,}", "-", d).strip("-")
    m1 = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", d)   # YYYY-M-D
    m2 = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", d)   # D-M-YYYY
    if m1:
        y, m, dd = map(int, m1.groups())
        if 1 <= m <= 12 and 1 <= dd <= 31:
            return f"{y:04d}-{m:02d}-{dd:02d}"
    if m2:
        dd, m, y = map(int, m2.groups())
        if 1 <= m <= 12 and 1 <= dd <= 31:
            return f"{y:04d}-{m:02d}-{dd:02d}"
    return None

def _join_dt(date_s: str | None, time_s: str | None) -> str | None:
    d = _norm_date(date_s)
    if d is None:
        return None
    t = _norm_time(time_s)
    return f"{d} {t}"

def _hms_to_minutes(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    m = re.match(r"^(\d{1,3}):(\d{2})(?::(\d{2}))?$", s)
    if not m:
        return _norm_decimal(s)
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return hh * 60 + mm + ss / 60.0

def _norm_decimal(x: str | None) -> float | None:
    if x is None:
        return None
    s = str(x).replace("\xa0", " ").replace(" ", "").strip()
    if s == "":
        return None
    s = re.sub(r"[^0-9,\.-]", "", s)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

# ================== Fő ==================
def main() -> None:
    log(f"Working dir: {BASE}")

    # 1) Beolvasás
    adagok, adag_file = _read_table_any(BASE, ADAGOK_STEMS)
    log(f"✔ Adagok: {adag_file} | {len(adagok)} sor")
    panelek, panel_file = _read_table_any(BASE, HUTOPANELEK_STEMS)
    log(f"✔ Hűtőpanelek: {panel_file} | {len(panelek)} sor")


    adagok = adagok.replace(r"^\s*$", pd.NA, regex=True)
    mask_all_blank_global = adagok.isna().all(axis=1)
    before_all = len(adagok)
    adagok = adagok.loc[~mask_all_blank_global].copy()
    after_all = len(adagok)
    log(f"Adag globális üressor-szűrés: {before_all} → {after_all} (eldobva: {before_all-after_all})")

    import unicodedata
    def _norm_header(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"[\s\.]+", "_", s)
        s = re.sub(r"[^a-z0-9_]", "", s)
        return s.strip("_")

    original_cols = list(adagok.columns)
    adagok.columns = [_norm_header(c) for c in adagok.columns]

    expected_order = ["adag_szam","kezdet_datum","kezdet_ido","vege_datum","vege_ido","adagkozi_ido","adagido"]

    def count_hits(cols):
        c = 0
        for name in expected_order:
            for col in cols:
                if name in col:
                    c += 1
                    break
        return c

    hits = count_hits(adagok.columns)
    if hits < 2 and len(adagok.columns) >= 7:
        rename_map = {adagok.columns[i]: expected_order[i] for i in range(7)}
        adagok.rename(columns=rename_map, inplace=True)
        log(f"i Fejlécek nem passzoltak → pozíciós átnevezés az első 7 oszlopra: {expected_order}")
    else:
        log("i Fejlécfelismerés elegendő találattal ment tovább.")

    log(f"i Adagok oszlopok (normalizálás előtt): {list(adagok.columns)}")

    CAND = {
        "adag_szam":      ["adag_szam","adagszam","adag","batch","sorszam"],
        "kezdet_datum":   ["kezdet_datum","kezdet_datuma","kezdet_date","kezdet","start_datum","start_date"],
        "kezdet_ido":     ["kezdet_ido","kezdet_ideje","kezdet_time","start_ido","start_time","ido_kezd"],
        "vege_datum":     ["vege_datum","vege_datuma","vege_date","veg","end_datum","end_date"],
        "vege_ido":       ["vege_ido","vege_ideje","vege_time","end_ido","end_time","ido_veg"],
        "adagkozi_ido":   ["adagkozi_ido","adagkozi","koztes_ido","kozti_ido","intervallum","atfordulasi_ido"],
        "adagido":        ["adagido","adag_ido","ossz_ido","total_ido","futasi_ido"],
        "kezdet_combo":   ["kezdet","kezdet_ts","kezdet_datumido","kezdet_dt","start","start_ts","start_dt","indulas","inditas"],
        "vege_combo":     ["vege","vege_ts","vege_datumido","vege_dt","end","end_ts","end_dt","befejezes"],
    }

    def pick(df, keys):
        for k in keys:
            if k in df.columns:
                return df[k]
        return pd.Series([None]*len(df))

    raw_adag_szam  = pick(adagok, CAND["adag_szam"])
    raw_kd         = pick(adagok, CAND["kezdet_datum"])
    raw_ki         = pick(adagok, CAND["kezdet_ido"])
    raw_vd         = pick(adagok, CAND["vege_datum"])
    raw_vi         = pick(adagok, CAND["vege_ido"])
    raw_ak         = pick(adagok, CAND["adagkozi_ido"])
    raw_ai         = pick(adagok, CAND["adagido"])
    raw_kcombo     = pick(adagok, CAND["kezdet_combo"])
    raw_vcombo     = pick(adagok, CAND["vege_combo"])

    kd_src = raw_kd.astype("object")
    ki_src = raw_ki.astype("object")
    vd_src = raw_vd.astype("object")
    vi_src = raw_vi.astype("object")

    def split_dt(x):
        if pd.isna(x): return (None, None)
        s = str(x).strip()
        if " " in s:
            d, t = s.split(" ", 1)
            return (d, t)
        return (s, None)

    missing_kd = kd_src.isna() | (kd_src.astype("string").str.strip().fillna("") == "")
    kd_from_combo = pd.Series([split_dt(v)[0] for v in raw_kcombo], dtype="object")
    ki_from_combo = pd.Series([split_dt(v)[1] for v in raw_kcombo], dtype="object")
    kd_src = kd_src.where(~missing_kd, kd_from_combo)
    ki_src = ki_src.where(~(ki_src.isna() | (ki_src.astype("string").str.strip().fillna("") == "")), ki_from_combo)

    missing_vd = vd_src.isna() | (vd_src.astype("string").str.strip().fillna("") == "")
    vd_from_combo = pd.Series([split_dt(v)[0] for v in raw_vcombo], dtype="object")
    vi_from_combo = pd.Series([split_dt(v)[1] for v in raw_vcombo], dtype="object")
    vd_src = vd_src.where(~missing_vd, vd_from_combo)
    vi_src = vi_src.where(~(vi_src.isna() | (vi_src.astype("string").str.strip().fillna("") == "")), vi_from_combo)

    kezdet_list = [_join_dt(kd_src.iat[i] if i < len(kd_src) else None,
                             ki_src.iat[i] if i < len(ki_src) else None)
                   for i in range(len(adagok))]
    vege_list   = [_join_dt(vd_src.iat[i] if i < len(vd_src) else None,
                             vi_src.iat[i] if i < len(vi_src) else None)
                   for i in range(len(adagok))]
    kezdet_ts = pd.to_datetime(kezdet_list, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    vege_ts   = pd.to_datetime(vege_list,   format="%Y-%m-%d %H:%M:%S", errors="coerce")

    adag_szam    = pd.to_numeric(raw_adag_szam, errors="coerce").astype("Int64")
    adagkozi_ido = pd.Series([_hms_to_minutes(v) for v in raw_ak], dtype="float")
    adagido      = pd.Series([_hms_to_minutes(v) for v in raw_ai], dtype="float")

    adag_dim = pd.DataFrame({
        "adag_szam":     adag_szam,
        "kezdet_datum":  kd_src,
        "kezdet_ido":    ki_src,
        "vege_datum":    vd_src,
        "vege_ido":      vi_src,
        "kezdet_ts":     kezdet_ts,
        "vege_ts":       vege_ts,
        "adagkozi_ido":  adagkozi_ido,
        "adagido":       adagido,
    })

    src_cols = ["adag_szam","kezdet_datum","kezdet_ido","vege_datum","vege_ido",
                "kezdet_ts","vege_ts","adagkozi_ido","adagido"]
    def is_blank_df(df: pd.DataFrame) -> pd.Series:
        s = df.apply(lambda col: col.astype("string").str.strip().fillna(""), axis=0)
        return s.eq("").all(axis=1)
    before = len(adag_dim)
    mask_all_blank = is_blank_df(adag_dim[src_cols])
    adag_dim = adag_dim.loc[~mask_all_blank].copy()
    after = len(adag_dim)
    log(f"Adag üres-sor szűrés (forrás): {before} → {after} (eldobva: {before-after})")

    log(f"Adag diag: sorok={len(adag_dim)} | NULL kezdet_ts={int(adag_dim['kezdet_ts'].isna().sum())} | NULL vege_ts={int(adag_dim['vege_ts'].isna().sum())}")
    log("i Adag első 3 sor normalizálva:")
    for _, r in adag_dim.head(3).iterrows():
        log(f"  adag_szam={r['adag_szam']} | kezdet={r['kezdet_ts']} | vege={r['vege_ts']}")

    panelek.rename(columns={panelek.columns[0]: "timestamp_raw"}, inplace=True)
    value_cols = [c for c in panelek.columns if c != "timestamp_raw"]

    ts_norm = []
    for raw in panelek["timestamp_raw"].astype(str).tolist():
        if " " in raw:
            d_part, t_part = raw.split(" ", 1)
            ts_norm.append(_join_dt(d_part, t_part))
        else:
            ts_norm.append(_join_dt(raw, None))
    panelek["ts"] = pd.to_datetime(ts_norm, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    melted = panelek.melt(id_vars=["ts"], value_vars=value_cols, var_name="panel_raw", value_name="ertek_text")
    melted["panel_szam"] = melted["panel_raw"].astype(str).str.extract(r"(\d+)").astype(float).astype("Int64")
    melted["ertek_num"] = pd.to_numeric(melted["ertek_text"].str.replace(",", ".", regex=False), errors="coerce")
    meres_pre = melted[["panel_szam","ts","ertek_num","ertek_text"]].dropna(subset=["panel_szam","ts"]).copy()
    meres_pre["panel_szam"] = meres_pre["panel_szam"].astype(int)

    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    DROP TABLE IF EXISTS {T('meres')} CASCADE;
    DROP TABLE IF EXISTS {T('adag')} CASCADE;
    DROP TABLE IF EXISTS {T('panel')} CASCADE;

    CREATE TABLE {T('panel')} (
      panel_id   serial PRIMARY KEY,
      panel_szam text UNIQUE NOT NULL,
      panel_nev  text
    );

    CREATE TABLE {T('adag')} (
      adag_id        bigserial PRIMARY KEY,
      adag_szam      integer UNIQUE,
      kezdet_ts      timestamp,
      vege_ts        timestamp,
      kezdet_datum   text,
      kezdet_ido     text,
      vege_datum     text,
      vege_ido       text,
      adagkozi_ido   double precision,
      adagido        double precision
    );

    CREATE TABLE {T('meres')} (
      meres_id    bigserial PRIMARY KEY,
      panel_id    integer NOT NULL REFERENCES {T('panel')}(panel_id),
      ts          timestamp NOT NULL,
      ertek_num   double precision,
      ertek_text  text,
      adag_id     bigint NULL REFERENCES {T('adag')}(adag_id)
    );

    CREATE INDEX idx_meres_ts ON {T('meres')}(ts);
    CREATE INDEX idx_adag_ts  ON {T('adag')}(kezdet_ts, vege_ts);
    """
    conn = psycopg2.connect(host=PGHOST, port=PGPORT, dbname=PGDB, user=PGUSER, password=PGPASS)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    log("✔ 3 tábla létrehozva")

    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE tmp_panel (panel_szam text, panel_nev text);")
        panels = meres_pre[["panel_szam"]].drop_duplicates().sort_values(by="panel_szam")
        tmp_panel_df = pd.DataFrame({
            "panel_szam": panels["panel_szam"].astype(str).tolist(),
            "panel_nev": [f"Panel {p}" for p in panels["panel_szam"].astype(str).tolist()]
        })
        cur.copy_from(df_to_copy_buffer(tmp_panel_df), "tmp_panel", sep="\t", null="")
        cur.execute(f"""
            INSERT INTO {T('panel')}(panel_szam, panel_nev)
            SELECT panel_szam, panel_nev FROM tmp_panel
            ON CONFLICT (panel_szam) DO UPDATE SET panel_nev = EXCLUDED.panel_nev;
        """)
        cur.execute("DROP TABLE tmp_panel;")
    conn.commit()
    log("✔ panel betöltve")

    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TEMP TABLE tmp_adag (
                adag_szam integer,
                kezdet_ts timestamp,
                vege_ts   timestamp,
                kezdet_datum text,
                kezdet_ido   text,
                vege_datum   text,
                vege_ido     text,
                adagkozi_ido double precision,
                adagido      double precision
            );
        """)

        def to_copy_ts(val) -> str:
            if pd.isna(val):
                return ""
            s = str(val).strip().replace("T", " ")
            if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):
                return ""
            ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
            if pd.isna(ts):
                return ""
            return ts.strftime("%Y-%m-%d %H:%M:%S")

        ad_fmt = adag_dim.copy()
        ad_fmt["kezdet_ts_str"] = ad_fmt["kezdet_ts"].apply(to_copy_ts)
        ad_fmt["vege_ts_str"]   = ad_fmt["vege_ts"].apply(to_copy_ts)

        no_end_date_mask = (
            ad_fmt["vege_datum"].astype("string").str.strip().fillna("").eq("") &
            ad_fmt["vege_ido"].astype("string").str.strip().fillna("").ne("")
        )
        ad_fmt.loc[no_end_date_mask, "vege_ts_str"] = ""

        to_copy = pd.DataFrame({
            "adag_szam":     ad_fmt["adag_szam"],
            "kezdet_ts":     ad_fmt["kezdet_ts_str"],
            "vege_ts":       ad_fmt["vege_ts_str"],
            "kezdet_datum":  ad_fmt["kezdet_datum"],
            "kezdet_ido":    ad_fmt["kezdet_ido"],
            "vege_datum":    ad_fmt["vege_datum"],
            "vege_ido":      ad_fmt["vege_ido"],
            "adagkozi_ido":  ad_fmt["adagkozi_ido"],
            "adagido":       ad_fmt["adagido"],
        })

        if len(to_copy) == 0:
            log("to_copy ÜRES! Diagnosztika (első 10 sor a forrásból):")
            dbg = ad_fmt[[
                "adag_szam","kezdet_datum","kezdet_ido","vege_datum","vege_ido",
                "kezdet_ts","vege_ts","kezdet_ts_str","vege_ts_str","adagkozi_ido","adagido"
            ]].head(10)
            for i, r in dbg.iterrows():
                log(f"[{i}] adag_szam={r['adag_szam']} | kd='{r['kezdet_datum']}' ki='{r['kezdet_ido']}' "
                    f"| vd='{r['vege_datum']}' vi='{r['vege_ido']}' "
                    f"| kezdet_ts='{r['kezdet_ts_str']}' vege_ts='{r['vege_ts_str']}' "
                    f"| adagkozi_ido={r['adagkozi_ido']} adagido={r['adagido']}")
            raise SystemExit("Megálltam, mert üres lenne a betöltés. A fenti minták alapján látszik az ok.")

        log("— tmp_adag mintasorok (kezdet_ts | vege_ts | adag_szam):")
        for _, row in to_copy.head(5).iterrows():
            log(f"   {row['kezdet_ts']} | {row['vege_ts']} | {row['adag_szam']}")

        cur.copy_from(df_to_copy_buffer(to_copy), "tmp_adag", sep="\t", null="")
        cur.execute("SELECT COUNT(*) FROM tmp_adag;")
        n_tmp = cur.fetchone()[0]
        log(f"— tmp_adag sorok: {n_tmp}")

        cur.execute(f"""
            INSERT INTO {T('adag')} (
                adag_szam, kezdet_ts, vege_ts, kezdet_datum, kezdet_ido,
                vege_datum, vege_ido, adagkozi_ido, adagido
            )
            SELECT
                adag_szam, kezdet_ts, vege_ts, kezdet_datum, kezdet_ido,
                vege_datum, vege_ido, adagkozi_ido, adagido
            FROM tmp_adag
            ON CONFLICT (adag_szam) DO UPDATE SET
                kezdet_ts = EXCLUDED.kezdet_ts,
                vege_ts   = EXCLUDED.vege_ts,
                kezdet_datum = EXCLUDED.kezdet_datum,
                kezdet_ido   = EXCLUDED.kezdet_ido,
                vege_datum   = EXCLUDED.vege_datum,
                vege_ido     = EXCLUDED.vege_ido,
                adagkozi_ido = EXCLUDED.adagkozi_ido,
                adagido      = EXCLUDED.adagido;
        """)
        cur.execute(f"SELECT COUNT(*) FROM {T('adag')};")
        n_adag_after = cur.fetchone()[0]
        log(f"— adag összes sor az INSERT után: {n_adag_after}")

        cur.execute("DROP TABLE tmp_adag;")
    conn.commit()
    log("✔ adag betöltve")

    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE tmp_meres_src (panel_szam text, ts timestamp, ertek_num double precision, ertek_text text);")
        m_fmt = meres_pre.copy()
        s = pd.to_datetime(m_fmt["ts"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        s = s.where(m_fmt["ts"].notna(), "")
        m_fmt["ts"] = s
        cur.copy_from(df_to_copy_buffer(m_fmt[["panel_szam","ts","ertek_num","ertek_text"]]), "tmp_meres_src", sep="\t", null="")
        cur.execute(f"""
            INSERT INTO {T('meres')}(panel_id, ts, ertek_num, ertek_text)
            SELECT p.panel_id, s.ts, s.ertek_num, s.ertek_text
            FROM tmp_meres_src s
            JOIN {T('panel')} p ON p.panel_szam = s.panel_szam;
        """)
        cur.execute("DROP TABLE tmp_meres_src;")
    conn.commit()
    log("✔ meres betöltve")

    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {T('meres')} m
            SET adag_id = a.adag_id
            FROM {T('adag')} a
            WHERE m.ts IS NOT NULL
              AND a.kezdet_ts IS NOT NULL
              AND a.vege_ts   IS NOT NULL
              AND m.ts >= a.kezdet_ts AND m.ts < a.vege_ts
              AND m.adag_id IS NULL;
        """)
        updated = cur.rowcount
    conn.commit()
    log(f"✔ adag_id kitöltve a meres-ben | frissített sorok: {updated}")

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {T('panel')};"); n_panel = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {T('adag')};");  n_adag  = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {T('meres')};"); n_meres = cur.fetchone()[0]
    log(f"✅ KÉSZ | panel: {n_panel} | adag: {n_adag} | meres: {n_meres}")

    conn.close()

if __name__ == "__main__":
    main()

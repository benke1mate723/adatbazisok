# main.py
"""
CSV adat import és tisztítási folyamat a hőmérséklet monitoring rendszerhez.

Ez a modul ezeket kezeli:
- CSV fájlok betöltése (Adagok és Hűtőpanelek)
- Adat normalizálás és tisztítás
- IQR-alapú outlier detektálás
- PostgreSQL adatbázis betöltés
"""

from __future__ import annotations

import io
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Alkalmazás konfiguráció."""
    base_dir: Path
    adagok_stems: List[str]
    hutopanelek_stems: List[str]

    # Adatbázis
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    db_schema: str

    # Adattisztítás
    temp_min: float = -20.0
    temp_max: float = 100.0
    spike_threshold: float = 3.0
    iqr_multiplier: float = 3.0
    iqr_min_samples: int = 10

    # Rolling window (csúszóablakos) anomália detektálás
    rolling_window_size: int = 900  # 900 másodperc (15 perc)
    rolling_threshold: float = 5.0  # Mediántól való max eltérés (°C)

    # Median smoothing (simítás)
    smoothing_enabled: bool = True
    smoothing_window_size: int = 600  # 600 másodperc mozgó medián

    @classmethod
    def from_env(cls) -> 'Config':
        """Konfiguráció létrehozása környezeti változókból."""
        return cls(
            base_dir=Path(__file__).parent.resolve(),
            adagok_stems=["Adagok"],
            hutopanelek_stems=["Hűtőpanelek", "Hutopanelek"],
            db_host=os.getenv("PGHOST", "localhost"),
            db_port=int(os.getenv("PGPORT", "5432")),
            db_name=os.getenv("PGDATABASE", "beadando"),
            db_user=os.getenv("PGUSER", "postgres"),
            db_password=os.getenv("PGPASSWORD", ""),
            db_schema="public",
        )


def log(msg: str) -> None:
    """Log üzenet kiírása flush-sal."""
    print(msg, flush=True)


def df_to_copy_buffer(df: pd.DataFrame, sep: str = "\t") -> io.StringIO:
    """DataFrame konvertálása StringIO buffer-ré COPY parancshoz."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, sep=sep, na_rep="")
    buf.seek(0)
    return buf


class CSVLoader:
    """CSV fájlok betöltése automatikus kódolás-detektálással."""

    @staticmethod
    def sniff_separator(text_sample: str) -> str:
        """CSV elválasztó detektálása minta szövegből."""
        return max([",", ";", "\t", "|"], key=lambda ch: text_sample.count(ch))

    @staticmethod
    def try_read_csv(path: Path) -> pd.DataFrame:
        """CSV beolvasása automatikus kódolás-detektálással."""
        with open(path, "rb") as f:
            raw = f.read(65536)
        sample = raw.decode("utf-8", errors="ignore")
        sep = CSVLoader.sniff_separator(sample)

        for enc in ("utf-8-sig", "utf-8", "cp1250", "latin1", "iso-8859-2"):
            try:
                return pd.read_csv(
                    path, sep=sep, encoding=enc, engine="python",
                    dtype=str, skip_blank_lines=True, keep_default_na=False
                )
            except (UnicodeDecodeError, Exception):
                continue

        return pd.read_csv(
            path, sep=sep, encoding="utf-8", engine="python",
            dtype=str, skip_blank_lines=True, keep_default_na=False
        )

    @staticmethod
    def read_table_any(base: Path, stems: List[str]) -> Tuple[pd.DataFrame, str]:
        """Tábla beolvasása CSV-ből vagy Excel-ből több név opció alapján."""
        for stem in stems:
            xlsx = base / f"{stem}.xlsx"
            csv = base / f"{stem}.csv"
            if xlsx.exists():
                return pd.read_excel(xlsx, dtype=str), xlsx.name
            if csv.exists():
                return CSVLoader.try_read_csv(csv), csv.name
        raise FileNotFoundError(f"Nem található fájl a {base} könyvtárban a következő nevekkel: {stems}")


class DataNormalizer:
    """Segédfüggvények dátumok, idők és numerikus értékek normalizálásához."""

    @staticmethod
    def normalize_time(t: Optional[str]) -> str:
        """Idő string normalizálása HH:MM:SS formátumra."""
        if not t:
            return "00:00:00"
        t = str(t).strip().replace(".", ":")
        m = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", t)
        if not m:
            return "00:00:00"
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    @staticmethod
    def normalize_date(d: Optional[str]) -> Optional[str]:
        """Dátum string normalizálása YYYY-MM-DD formátumra."""
        if not d:
            return None
        d = str(d).strip()
        d = re.sub(r"[^\d./\-]", "", d)
        d = re.sub(r"[./]", "-", d)
        d = re.sub(r"-{2,}", "-", d).strip("-")

        # YYYY-M-D formátum
        m1 = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", d)
        if m1:
            y, m, dd = map(int, m1.groups())
            if 1 <= m <= 12 and 1 <= dd <= 31:
                return f"{y:04d}-{m:02d}-{dd:02d}"

        # D-M-YYYY formátum
        m2 = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", d)
        if m2:
            dd, m, y = map(int, m2.groups())
            if 1 <= m <= 12 and 1 <= dd <= 31:
                return f"{y:04d}-{m:02d}-{dd:02d}"

        return None

    @staticmethod
    def join_datetime(date_s: Optional[str], time_s: Optional[str]) -> Optional[str]:
        """Dátum és idő stringek kombinálása datetime stringgé."""
        d = DataNormalizer.normalize_date(date_s)
        if d is None:
            return None
        t = DataNormalizer.normalize_time(time_s)
        return f"{d} {t}"

    @staticmethod
    def hms_to_minutes(v: Optional[str]) -> Optional[float]:
        """HH:MM:SS vagy decimális string konvertálása percekre."""
        if v is None:
            return None
        s = str(v).strip()
        m = re.match(r"^(\d{1,3}):(\d{2})(?::(\d{2}))?$", s)
        if not m:
            return DataNormalizer.normalize_decimal(s)
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        return hh * 60 + mm + ss / 60.0

    @staticmethod
    def normalize_decimal(x: Optional[str]) -> Optional[float]:
        """Decimális string normalizálása (kezeli a vessző/pont kétértelműséget)."""
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

    @staticmethod
    def normalize_header(s: str) -> str:
        """Oszlop fejléc normalizálása konzisztens illesztéshez."""
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"[\s\.]+", "_", s)
        s = re.sub(r"[^a-z0-9_]", "", s)
        return s.strip("_")


class TemperatureCleaner:
    """Hőmérséklet adatok tisztítása outlier detektálással."""

    def __init__(self, config: Config):
        """Inicializálás konfigurációval."""
        self.config = config

    def apply_hard_limits(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Fizikai hőmérséklet határok alkalmazása."""
        outlier_mask = (series < self.config.temp_min) | (series > self.config.temp_max)
        outlier_mask = outlier_mask.fillna(False)
        cleaned = series.where(~outlier_mask, None)
        return cleaned, outlier_mask

    def detect_spikes(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Hirtelen hőmérséklet ugrások detektálása."""
        diff_prev = series.diff().abs()
        diff_next = series.diff(-1).abs()
        spike_mask = (diff_prev > self.config.spike_threshold) & (diff_next > self.config.spike_threshold)
        spike_mask = spike_mask.fillna(False)
        cleaned = series.where(~spike_mask, None)
        return cleaned, spike_mask

    def detect_rolling_window_anomalies(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Rolling window medián alapú anomália detektálás.

        Tartós oszcillációk és anomáliák szűrésére, ahol az érték jelentősen
        eltér a környező értékek mediánjától.
        """
        if series.notna().sum() < self.config.rolling_window_size:
            # Nincs elég adat a rolling windowhoz
            return series, pd.Series(False, index=series.index)

        # Rolling medián számítása (center=True: szimmetrikus ablak)
        rolling_median = series.rolling(
            window=self.config.rolling_window_size,
            center=True,
            min_periods=max(1, self.config.rolling_window_size // 2)
        ).median()

        # Eltérés a mediántól
        deviation = (series - rolling_median).abs()

        # Anomália maszk: ha az eltérés > threshold
        anomaly_mask = deviation > self.config.rolling_threshold
        anomaly_mask = anomaly_mask.fillna(False)

        # Értékek törlése az anomáliáknál
        cleaned = series.where(~anomaly_mask, None)

        return cleaned, anomaly_mask

    def iqr_outlier_filter(self, series: pd.Series) -> pd.Series:
        """Statisztikai outlierek detektálása IQR módszerrel."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr
        outlier_mask = (series < lower) | (series > upper)
        outlier_mask = outlier_mask.fillna(False)
        return outlier_mask

    def apply_median_smoothing(self, series: pd.Series) -> pd.Series:
        """
        Mozgó medián simítás.

        A tisztított értékeket lecseréli a környező értékek mediánjára,
        így simább görbét eredményez magas frekvenciájú zaj nélkül.
        """
        if not self.config.smoothing_enabled:
            return series

        if series.notna().sum() < self.config.smoothing_window_size:
            return series

        # Mozgó medián számítása
        smoothed = series.rolling(
            window=self.config.smoothing_window_size,
            center=True,
            min_periods=1
        ).median()

        return smoothed

    def clean_temperature_data(self, df: pd.DataFrame,
                               value_column: str = "ertek_num",
                               panel_column: str = "panel_szam") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Többszintű hőmérséklet adat tisztítás IQR flaggeléssel.

        Fizikai hibák törlésre kerülnek, statisztikai outlierek flagelve de megtartva.
        """
        result_df = df.copy()
        result_df["ertek_num_original"] = result_df[value_column].copy()
        result_df["fizikai_hiba"] = False
        result_df["statisztikai_kiugro"] = False

        stats = {
            "total_records": len(result_df),
            "panels": {},
            "total_physical_errors": 0,
            "total_statistical_outliers": 0
        }

        # Panel-enként tisztítás
        for panel_id in result_df[panel_column].unique():
            panel_mask = result_df[panel_column] == panel_id
            panel_data = result_df.loc[panel_mask, value_column].copy()
            original_count = panel_data.notna().sum()

            # 1. Fizikai határok - TÖRLÉS
            panel_data, hard_mask = self.apply_hard_limits(panel_data)

            # 2. Spike detektálás - TÖRLÉS
            panel_data, spike_mask = self.detect_spikes(panel_data)

            # 3. Rolling window anomália detektálás - TÖRLÉS
            panel_data, rolling_mask = self.detect_rolling_window_anomalies(panel_data)

            # 4. IQR filter - CSAK FLAGELÉS
            physical_error_mask = hard_mask | spike_mask | rolling_mask
            iqr_mask = pd.Series(False, index=panel_data.index)
            if panel_data.notna().sum() > self.config.iqr_min_samples:
                iqr_mask = self.iqr_outlier_filter(panel_data)
            statistical_outlier_mask = iqr_mask & ~physical_error_mask

            # 5. Median smoothing (simitas) - MÓDOSÍTÁS, NEM TÖRLÉS
            if self.config.smoothing_enabled:
                panel_data = self.apply_median_smoothing(panel_data)

            # Eredmények visszaírása
            result_df.loc[panel_mask, value_column] = panel_data
            result_df.loc[panel_mask, "fizikai_hiba"] = physical_error_mask
            result_df.loc[panel_mask, "statisztikai_kiugro"] = statistical_outlier_mask

            # Statisztikák frissítése
            physical_errors = physical_error_mask.sum()
            statistical_outliers = statistical_outlier_mask.sum()

            stats["panels"][int(panel_id)] = {
                "original": original_count,
                "removed_hard": hard_mask.sum(),
                "removed_spike": spike_mask.sum(),
                "removed_rolling": rolling_mask.sum(),
                "physical_errors": physical_errors,
                "statistical_outliers": statistical_outliers,
                "total_flagged": physical_errors + statistical_outliers,
                "remaining_clean": original_count - physical_errors
            }
            stats["total_physical_errors"] += physical_errors
            stats["total_statistical_outliers"] += statistical_outliers

        result_df["kiugro_ertek"] = result_df["fizikai_hiba"] | result_df["statisztikai_kiugro"]
        return result_df, stats


def print_cleaning_report(stats: Dict[str, Any]) -> None:
    """Adattisztítási riport kiírása."""
    log("=" * 80)
    log("ADATTISZTÍTÁSI RIPORT - IQR FLAGGING MÓDSZER")
    log("=" * 80)
    log(f"Összes mérés: {stats['total_records']:,}")
    log("")
    log(f"[HIBA] Fizikai hibák (TÖRLÉS):       {stats['total_physical_errors']:7,} "
        f"({stats['total_physical_errors'] / stats['total_records'] * 100:5.2f}%)")
    log(f"[FIGYELEM] Statisztikai outlierek (FLAG): {stats['total_statistical_outliers']:7,} "
        f"({stats['total_statistical_outliers'] / stats['total_records'] * 100:5.2f}%)")
    clean_count = stats['total_records'] - stats['total_physical_errors'] - stats['total_statistical_outliers']
    log(f"[OK] Tiszta adatok:                 {clean_count:7,} "
        f"({clean_count / stats['total_records'] * 100:5.2f}%)")
    log("")
    log("Panel-enkénti részletek:")
    log("-" * 80)

    for panel_id in sorted(stats["panels"].keys()):
        panel_stat = stats["panels"][panel_id]
        log(f"Panel {panel_id:2d}: "
            f"eredeti={panel_stat['original']:6,} | "
            f"fizikai_hiba={panel_stat['physical_errors']:5,} | "
            f"stat_outlier={panel_stat['statistical_outliers']:5,} | "
            f"tiszta={panel_stat['remaining_clean']:6,}")
        if panel_stat["physical_errors"] > 0 or panel_stat["statistical_outliers"] > 0:
            log(f"           [hard_limit={panel_stat['removed_hard']:4,} | "
                f"spike={panel_stat['removed_spike']:4,} | "
                f"rolling={panel_stat['removed_rolling']:4,}]")
    log("=" * 80)
    log("")
    log("[TIP] MAGYARAZAT:")
    log("   [HIBA] Fizikai hibak: >100C vagy <-20C vagy spike vagy rolling anomalia -> ertek_num = NULL")
    log("   [FIGYELEM] Statisztikai outlierek: IQR outlier -> ertek_num MEGMARAD, csak flagelve")
    log("   [SIMITAS] Median smoothing: 600 mp mozgo median -> simabb gorbe")
    log("   [OK] Tiszta adatok: Sem fizikai hiba, sem statisztikai outlier")
    log("=" * 80)


class AdagProcessor:
    """'Adag' (batch) adatok feldolgozása és normalizálása."""

    # Elvárt oszlop minták
    COLUMN_CANDIDATES = {
        "adag_szam": ["adag_szam", "adagszam", "adag", "batch", "sorszam"],
        "kezdet_datum": ["kezdet_datum", "kezdet_datuma", "kezdet_date", "kezdet", "start_datum", "start_date"],
        "kezdet_ido": ["kezdet_ido", "kezdet_ideje", "kezdet_time", "start_ido", "start_time", "ido_kezd"],
        "vege_datum": ["vege_datum", "vege_datuma", "vege_date", "veg", "end_datum", "end_date"],
        "vege_ido": ["vege_ido", "vege_ideje", "vege_time", "end_ido", "end_time", "ido_veg"],
        "adagkozi_ido": ["adagkozi_ido", "adagkozi", "koztes_ido", "intervallum"],
        "adagido": ["adagido", "adag_ido", "ossz_ido", "total_ido"],
        "kezdet_combo": ["kezdet", "kezdet_ts", "kezdet_datumido", "start", "indulas"],
        "vege_combo": ["vege", "vege_ts", "vege_datumido", "end", "befejezes"],
    }

    EXPECTED_ORDER = ["adag_szam", "kezdet_datum", "kezdet_ido", "vege_datum", "vege_ido", "adagkozi_ido", "adagido"]

    def __init__(self):
        """Feldolgozó inicializálása."""
        pass

    def normalize_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame oszlopfejlécek normalizálása."""
        df_copy = df.copy()
        df_copy.columns = [DataNormalizer.normalize_header(c) for c in df_copy.columns]

        # Ha nincs elég találat, pozíciós elnevezés használata
        hits = self._count_header_hits(df_copy.columns)
        if hits < 2 and len(df_copy.columns) >= 7:
            rename_map = {df_copy.columns[i]: self.EXPECTED_ORDER[i] for i in range(7)}
            df_copy.rename(columns=rename_map, inplace=True)
            log(f"i Fejlecek nem passzoltak -> pozicios atnevezes: {self.EXPECTED_ORDER}")
        return df_copy

    def _count_header_hits(self, cols: List[str]) -> int:
        """Megszámolja hány elvárt oszlop található a fejlécekben."""
        count = 0
        for name in self.EXPECTED_ORDER:
            for col in cols:
                if name in col:
                    count += 1
                    break
        return count

    def _pick_column(self, df: pd.DataFrame, keys: List[str]) -> pd.Series:
        """Első illeszkedő oszlop kiválasztása a DataFrame-ből."""
        for k in keys:
            if k in df.columns:
                return df[k]
        return pd.Series([None] * len(df))

    def process_adag_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adag (batch) adatok feldolgozása és normalizálása."""
        # Üres sorok eltávolítása
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        mask_blank = df.isna().all(axis=1)
        before = len(df)
        df = df.loc[~mask_blank].copy()
        after = len(df)
        log(f"Adag globalis üressor-szűrés: {before} -> {after} (eldobva: {before - after})")

        # Fejlécek normalizálása
        df = self.normalize_headers(df)
        log(f"i Adagok oszlopok (normalizálás után): {list(df.columns)}")

        # Oszlopok kinyerése
        raw_adag_szam = self._pick_column(df, self.COLUMN_CANDIDATES["adag_szam"])
        raw_kd = self._pick_column(df, self.COLUMN_CANDIDATES["kezdet_datum"])
        raw_ki = self._pick_column(df, self.COLUMN_CANDIDATES["kezdet_ido"])
        raw_vd = self._pick_column(df, self.COLUMN_CANDIDATES["vege_datum"])
        raw_vi = self._pick_column(df, self.COLUMN_CANDIDATES["vege_ido"])
        raw_ak = self._pick_column(df, self.COLUMN_CANDIDATES["adagkozi_ido"])
        raw_ai = self._pick_column(df, self.COLUMN_CANDIDATES["adagido"])
        raw_kcombo = self._pick_column(df, self.COLUMN_CANDIDATES["kezdet_combo"])
        raw_vcombo = self._pick_column(df, self.COLUMN_CANDIDATES["vege_combo"])

        # Kombinált oszlopok kezelése (dátum + idő egy mezőben)
        kd_src, ki_src = self._split_combo_column(raw_kd, raw_ki, raw_kcombo)
        vd_src, vi_src = self._split_combo_column(raw_vd, raw_vi, raw_vcombo)

        # Időbélyegek létrehozása
        kezdet_list = [DataNormalizer.join_datetime(kd_src.iat[i], ki_src.iat[i]) for i in range(len(df))]
        vege_list = [DataNormalizer.join_datetime(vd_src.iat[i], vi_src.iat[i]) for i in range(len(df))]

        # Eredmény DataFrame létrehozása
        result = pd.DataFrame({
            "adag_szam": pd.to_numeric(raw_adag_szam, errors="coerce").astype("Int64"),
            "kezdet_datum": kd_src,
            "kezdet_ido": ki_src,
            "vege_datum": vd_src,
            "vege_ido": vi_src,
            "kezdet_ts": pd.to_datetime(kezdet_list, format="%Y-%m-%d %H:%M:%S", errors="coerce"),
            "vege_ts": pd.to_datetime(vege_list, format="%Y-%m-%d %H:%M:%S", errors="coerce"),
            "adagkozi_ido": pd.Series([DataNormalizer.hms_to_minutes(v) for v in raw_ak], dtype="float"),
            "adagido": pd.Series([DataNormalizer.hms_to_minutes(v) for v in raw_ai], dtype="float"),
        })

        # Teljesen üres sorok eltávolítása
        src_cols = ["adag_szam", "kezdet_datum", "kezdet_ido", "vege_datum", "vege_ido"]
        mask_blank = result[src_cols].apply(lambda col: col.astype("string").str.strip().fillna("")).eq("").all(axis=1)
        before = len(result)
        result = result.loc[~mask_blank].copy()
        after = len(result)
        log(f"Adag ures-sor szures: {before} -> {after} (eldobva: {before - after})")

        # Diagnosztika
        log(f"Adag diag: sorok={len(result)} | NULL kezdet_ts={int(result['kezdet_ts'].isna().sum())} | NULL vege_ts={int(result['vege_ts'].isna().sum())}")
        log("i Adag első 3 sor normalizálva:")
        for _, r in result.head(3).iterrows():
            log(f"  adag_szam={r['adag_szam']} | kezdet={r['kezdet_ts']} | vege={r['vege_ts']}")

        return result

    def _split_combo_column(self, date_col: pd.Series, time_col: pd.Series,
                            combo_col: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Kombinált datetime oszlop szétbontása külön dátum és idő oszlopokra."""

        def split_dt(x):
            if pd.isna(x):
                return None, None
            s = str(x).strip()
            if " " in s:
                d, t = s.split(" ", 1)
                return d, t
            return s, None

        # Kombóból kinyerés ha az elsődleges oszlopok hiányoznak
        missing_date = date_col.isna() | (date_col.astype("string").str.strip().fillna("") == "")
        date_from_combo = pd.Series([split_dt(v)[0] for v in combo_col], dtype="object")
        time_from_combo = pd.Series([split_dt(v)[1] for v in combo_col], dtype="object")

        result_date = date_col.where(~missing_date, date_from_combo)
        result_time = time_col.where(~(time_col.isna() | (time_col.astype("string").str.strip().fillna("") == "")),
                                     time_from_combo)

        return result_date, result_time


class PanelProcessor:
    """Panel hőmérséklet adatok feldolgozása és normalizálása."""

    def process_panel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Panel adatok feldolgozása szélesről hosszú formátumra."""
        # Első oszlop átnevezése timestamp_raw-ra
        df_copy = df.copy()
        df_copy.rename(columns={df_copy.columns[0]: "timestamp_raw"}, inplace=True)
        value_cols = [c for c in df_copy.columns if c != "timestamp_raw" and "Time" not in c]

        # Időbélyegek normalizálása
        ts_norm = []
        for raw in df_copy["timestamp_raw"].astype(str).tolist():
            if " " in raw:
                d_part, t_part = raw.split(" ", 1)
                ts_norm.append(DataNormalizer.join_datetime(d_part, t_part))
            else:
                ts_norm.append(DataNormalizer.join_datetime(raw, None))

        df_copy["ts"] = pd.to_datetime(ts_norm, format="%Y-%m-%d %H:%M:%S", errors="coerce")

        # Szélesről hosszú formátumra alakítás (melt)
        melted = df_copy.melt(id_vars=["ts"], value_vars=value_cols,
                              var_name="panel_raw", value_name="ertek_text")

        # Panel szám kinyerése
        melted["panel_szam"] = melted["panel_raw"].astype(str).str.extract(r"(\d+)").astype(float).astype("Int64")

        # Értékek numerikussá alakítása
        melted["ertek_num"] = pd.to_numeric(
            melted["ertek_text"].str.replace(",", ".", regex=False),
            errors="coerce"
        )

        # Tisztítás
        result = melted[["panel_szam", "ts", "ertek_num", "ertek_text"]].dropna(subset=["panel_szam", "ts"]).copy()
        result["panel_szam"] = result["panel_szam"].astype(int)

        return result


class DatabaseManager:
    """Adatbázis séma létrehozása és adat betöltés kezelése."""

    def __init__(self, config: Config):
        """Inicializálás konfigurációval."""
        self.config = config
        self.schema = config.db_schema
        self.conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        """Adatbázis kapcsolat létrehozása."""
        self.conn = psycopg2.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password
        )
        self.conn.autocommit = False

    def close(self) -> None:
        """Adatbázis kapcsolat lezárása."""
        if self.conn:
            self.conn.close()

    def _table_name(self, name: str) -> str:
        """Teljes táblázat név lekérése."""
        return f"{self.schema}.{name}"

    def create_schema(self) -> None:
        """Adatbázis séma létrehozása (táblák és indexek)."""
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {self.schema};

        DROP TABLE IF EXISTS {self._table_name('meres')} CASCADE;
        DROP TABLE IF EXISTS {self._table_name('adag')} CASCADE;
        DROP TABLE IF EXISTS {self._table_name('panel')} CASCADE;

        CREATE TABLE {self._table_name('panel')} (
          panel_id   serial PRIMARY KEY,
          panel_szam text UNIQUE NOT NULL,
          panel_nev  text
        );

        CREATE TABLE {self._table_name('adag')} (
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

        CREATE TABLE {self._table_name('meres')} (
          meres_id                bigserial PRIMARY KEY,
          panel_id                integer NOT NULL REFERENCES {self._table_name('panel')}(panel_id),
          ts                      timestamp NOT NULL,
          ertek_num               double precision,
          ertek_text              text,
          ertek_num_original      double precision,
          fizikai_hiba            boolean DEFAULT FALSE,
          statisztikai_kiugro     boolean DEFAULT FALSE,
          kiugro_ertek            boolean DEFAULT FALSE,
          adag_id                 bigint NULL REFERENCES {self._table_name('adag')}(adag_id)
        );

        CREATE INDEX idx_meres_ts ON {self._table_name('meres')}(ts);
        CREATE INDEX idx_adag_ts  ON {self._table_name('adag')}(kezdet_ts, vege_ts);
        """

        with self.conn.cursor() as cur:
            cur.execute(ddl)
        self.conn.commit()
        log("[OK] 3 tabla letrehozva")

    def load_panel_data(self, meres_df: pd.DataFrame) -> None:
        """Panel adatok betöltése az adatbázisba."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE tmp_panel (panel_szam text, panel_nev text);")
            panels = meres_df[["panel_szam"]].drop_duplicates().sort_values(by="panel_szam")
            tmp_panel_df = pd.DataFrame({
                "panel_szam": panels["panel_szam"].astype(str).tolist(),
                "panel_nev": [f"Panel {p}" for p in panels["panel_szam"].astype(str).tolist()]
            })
            cur.copy_from(df_to_copy_buffer(tmp_panel_df), "tmp_panel", sep="\t", null="")
            cur.execute(f"""
                INSERT INTO {self._table_name('panel')}(panel_szam, panel_nev)
                SELECT panel_szam, panel_nev FROM tmp_panel
                ON CONFLICT (panel_szam) DO UPDATE SET panel_nev = EXCLUDED.panel_nev;
            """)
            cur.execute("DROP TABLE tmp_panel;")
        self.conn.commit()
        log("[OK] panel betoltve")

    def load_adag_data(self, adag_df: pd.DataFrame) -> None:
        """Adag (batch) adatok betöltése az adatbázisba."""
        with self.conn.cursor() as cur:
            cur.execute("""
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

            # Időbélyegek formázása COPY-hoz
            ad_fmt = adag_df.copy()
            ad_fmt["kezdet_ts_str"] = ad_fmt["kezdet_ts"].apply(self._format_ts_for_copy)
            ad_fmt["vege_ts_str"] = ad_fmt["vege_ts"].apply(self._format_ts_for_copy)

            # Hiányzó vég dátum kezelése amikor van vég idő
            no_end_date_mask = (
                    ad_fmt["vege_datum"].astype("string").str.strip().fillna("").eq("") &
                    ad_fmt["vege_ido"].astype("string").str.strip().fillna("").ne("")
            )
            ad_fmt.loc[no_end_date_mask, "vege_ts_str"] = ""

            to_copy = pd.DataFrame({
                "adag_szam": ad_fmt["adag_szam"],
                "kezdet_ts": ad_fmt["kezdet_ts_str"],
                "vege_ts": ad_fmt["vege_ts_str"],
                "kezdet_datum": ad_fmt["kezdet_datum"],
                "kezdet_ido": ad_fmt["kezdet_ido"],
                "vege_datum": ad_fmt["vege_datum"],
                "vege_ido": ad_fmt["vege_ido"],
                "adagkozi_ido": ad_fmt["adagkozi_ido"],
                "adagido": ad_fmt["adagido"],
            })

            log("- tmp_adag mintasorok (kezdet_ts | vege_ts | adag_szam):")
            for _, row in to_copy.head(5).iterrows():
                log(f"   {row['kezdet_ts']} | {row['vege_ts']} | {row['adag_szam']}")

            cur.copy_from(df_to_copy_buffer(to_copy), "tmp_adag", sep="\t", null="")

            cur.execute(f"""
                INSERT INTO {self._table_name('adag')} (
                    adag_szam, kezdet_ts, vege_ts, kezdet_datum, kezdet_ido,
                    vege_datum, vege_ido, adagkozi_ido, adagido
                )
                SELECT
                    adag_szam, kezdet_ts, vege_ts, kezdet_datum, kezdet_ido,
                    vege_datum, vege_ido, adagkozi_ido, adagido
                FROM tmp_adag
                ON CONFLICT (adag_szam) DO UPDATE SET
                    kezdet_ts = EXCLUDED.kezdet_ts,
                    vege_ts   = EXCLUDED.vege_ts;
            """)
            cur.execute("DROP TABLE tmp_adag;")
        self.conn.commit()
        log("[OK] adag betoltve")

    def load_meres_data(self, meres_df: pd.DataFrame) -> None:
        """Mérés (measurement) adatok betöltése az adatbázisba."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE tmp_meres_src (
                    panel_szam text,
                    ts timestamp,
                    ertek_num double precision,
                    ertek_text text,
                    ertek_num_original double precision,
                    fizikai_hiba boolean,
                    statisztikai_kiugro boolean,
                    kiugro_ertek boolean
                );
            """)

            m_fmt = meres_df.copy()
            s = pd.to_datetime(m_fmt["ts"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            s = s.where(m_fmt["ts"].notna(), "")
            m_fmt["ts"] = s

            cur.copy_from(
                df_to_copy_buffer(m_fmt[["panel_szam", "ts", "ertek_num", "ertek_text", "ertek_num_original",
                                         "fizikai_hiba", "statisztikai_kiugro", "kiugro_ertek"]]),
                "tmp_meres_src",
                sep="\t",
                null=""
            )

            cur.execute(f"""
                INSERT INTO {self._table_name('meres')}(panel_id, ts, ertek_num, ertek_text, ertek_num_original,
                                     fizikai_hiba, statisztikai_kiugro, kiugro_ertek)
                SELECT p.panel_id, s.ts, s.ertek_num, s.ertek_text, s.ertek_num_original,
                       s.fizikai_hiba, s.statisztikai_kiugro, s.kiugro_ertek
                FROM tmp_meres_src s
                JOIN {self._table_name('panel')} p ON p.panel_szam = s.panel_szam;
            """)
            cur.execute("DROP TABLE tmp_meres_src;")
        self.conn.commit()
        log("[OK] meres betoltve")

    def link_adag_to_meres(self) -> int:
        """Adag_id összekapcsolása a mérési rekordokkal időbélyeg tartományok alapján."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {self._table_name('meres')} m
                SET adag_id = a.adag_id
                FROM {self._table_name('adag')} a
                WHERE m.ts IS NOT NULL
                  AND a.kezdet_ts IS NOT NULL
                  AND a.vege_ts   IS NOT NULL
                  AND m.ts >= a.kezdet_ts AND m.ts < a.vege_ts
                  AND m.adag_id IS NULL;
            """)
            updated = cur.rowcount
        self.conn.commit()
        log(f"[OK] adag_id kitoltve a meres-ben | frissitett sorok: {updated}")
        return updated

    @staticmethod
    def _format_ts_for_copy(val) -> str:
        """Időbélyeg formázása PostgreSQL COPY parancshoz."""
        if pd.isna(val):
            return ""
        s = str(val).strip().replace("T", " ")
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):
            return ""
        ts = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S")
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    """Fő végrehajtási pipeline."""
    config = Config.from_env()
    log(f"Working dir: {config.base_dir}")

    # 1) CSV fájlok betöltése
    loader = CSVLoader()
    adagok, adag_file = loader.read_table_any(config.base_dir, config.adagok_stems)
    log(f"[OK] Adagok: {adag_file} | {len(adagok)} sor")
    panelek, panel_file = loader.read_table_any(config.base_dir, config.hutopanelek_stems)
    log(f"[OK] Hutopanelek: {panel_file} | {len(panelek)} sor")

    # 2) Adag adatok feldolgozása
    adag_processor = AdagProcessor()
    adag_dim = adag_processor.process_adag_data(adagok)

    # 3) Panel adatok feldolgozása
    panel_processor = PanelProcessor()
    meres_pre = panel_processor.process_panel_data(panelek)

    # 4) Hőmérséklet adatok tisztítása
    log("[INFO] Adattisztitas folyamatban...")
    cleaner = TemperatureCleaner(config)
    meres_pre, cleaning_stats = cleaner.clean_temperature_data(meres_pre)
    print_cleaning_report(cleaning_stats)

    # 5) Adatbázis műveletek
    db_manager = DatabaseManager(config)
    db_manager.connect()

    try:
        db_manager.create_schema()
        db_manager.load_panel_data(meres_pre)
        db_manager.load_adag_data(adag_dim)
        db_manager.load_meres_data(meres_pre)
        db_manager.link_adag_to_meres()

        # 6) Összegzés
        with db_manager.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {config.db_schema}.panel")
            n_panel = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {config.db_schema}.adag")
            n_adag = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {config.db_schema}.meres")
            n_meres = cur.fetchone()[0]

        log(f"[OK] KESZ | panel: {n_panel} | adag: {n_adag} | meres: {n_meres}")
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Adatbázis wrapper modul PostgreSQL kapcsolatokhoz.

Egyszerű interfészt biztosít lekérdezések és parancsok végrehajtásához,
automatikus kapcsolatkezeléssel és hibakezeléssel.
"""

from typing import Any, List, Optional
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

# Adatbázis konfiguráció
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDB = os.getenv("PGDATABASE", "beadando")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASS = os.getenv("PGPASSWORD", "")

SCHEMA = "public"


class DB:
    """Adatbázis wrapper kapcsolatkezeléssel és lekérdezés-végrehajtással."""

    def __init__(self):
        """Adatbázis kapcsolat inicializálása."""
        self.login_params = {
            "database": PGDB,
            "user": PGUSER,
            "password": PGPASS,
            "host": PGHOST,
            "port": PGPORT
        }
        self.conn: Optional[psycopg2.extensions.connection] = None
        self.reconnect()

    def __enter__(self):
        """Context manager belépés."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager kilépés - lezárja a kapcsolatot."""
        self.end_connection()

    def reconnect(self) -> None:
        """Kapcsolódás vagy újrakapcsolódás az adatbázishoz."""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**self.login_params)

    def end_connection(self) -> None:
        """Adatbázis kapcsolat lezárása."""
        if self.conn and not self.conn.closed:
            self.conn.close()
        self.conn = None

    def get_data(self, sql_text: str, params: List[Any] = None) -> List[Any]:
        """
        SELECT lekérdezés végrehajtása és eredmények visszaadása.

        Args:
            sql_text: SQL lekérdezés string
            params: Opcionális lekérdezési paraméterek listája

        Returns:
            Lekérdezés eredményeinek listája
        """
        if params is None:
            params = []

        if self.conn is None or self.conn.closed:
            self.reconnect()

        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(sql_text, params)
                return cur.fetchall()
        except Exception as e:
            print(f"Hiba a lekérdezés végrehajtásakor: {e}")
            print(traceback.format_exc())
            return []

    def do_command(self, sql_text: str, params: List[Any] = None) -> bool:
        """
        Adatbázis parancs végrehajtása (INSERT, UPDATE, DELETE).

        Args:
            sql_text: SQL parancs string
            params: Opcionális parancs paraméterek listája

        Returns:
            True ha sikeres, False egyébként
        """
        if params is None:
            params = []

        if self.conn is None or self.conn.closed:
            self.reconnect()

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql_text, params)
            return True
        except Exception as e:
            print(f"Hiba a parancs végrehajtásakor: {e}")
            print(traceback.format_exc())
            self.rollback()
            return False

    def commit(self) -> bool:
        """
        Függőben lévő tranzakciók véglegesítése.

        Returns:
            True ha sikeres
        """
        if self.conn and not self.conn.closed:
            self.conn.commit()
        return True

    def rollback(self) -> bool:
        """
        Függőben lévő tranzakciók visszavonása.

        Returns:
            True ha sikeres
        """
        if self.conn and not self.conn.closed:
            self.conn.rollback()
        return True

    def get_panel_data(self, panel_id: int, column: str = "ertek_num",
                       include_outliers: bool = False, limit: Optional[int] = None) -> List[Any]:
        """
        Adatok lekérése egy adott panelhez.

        Args:
            panel_id: Panel azonosító
            column: Lekérendő oszlop ('ertek_num' vagy 'ts')
            include_outliers: Outlierek is bekerüljenek-e
            limit: Opcionális limit a sorok számára

        Returns:
            Adat sorok listája
        """
        outlier_filter = "" if include_outliers else "AND ertek_num IS NOT NULL"
        limit_clause = f"LIMIT {limit}" if limit else ""

        sql = f"""
            SELECT {column}
            FROM {SCHEMA}.meres
            WHERE panel_id = %s {outlier_filter}
            ORDER BY ts
            {limit_clause}
        """
        return self.get_data(sql, [panel_id])

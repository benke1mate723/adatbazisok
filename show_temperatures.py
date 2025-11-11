"""
Fejlett hőmérséklet vizualizációs eszköz több megjelenítési móddal.

Funkciók:
1. Időalapú megjelenítés (egy vagy több panel)
2. Több panel összehasonlítása
3. Adag-alapú szűrés
4. Panel statisztikák
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from db import DB


@dataclass
class PlotConfig:
    """Konfiguráció a grafikon stílushoz és korlátokhoz."""
    # Alapértelmezett színek több panelhez
    COLORS: List[str] = None
    # Alapértelmezett limitek
    DEFAULT_LIMIT: int = 100000  # 24+ ora adathoz (86400 masodperc)
    COMPARISON_LIMIT: int = 100000  # Tobb panel osszehasonlitashoz (teljes 24 ora)
    OUTLIER_LIMIT: int = 5000  # Outlierek megjelenitesehez
    # Ábra méret
    FIGURE_SIZE: Tuple[int, int] = (14, 7)
    # Betűméretek
    FONT_SIZE_LABEL: int = 12
    FONT_SIZE_TITLE: int = 14
    FONT_SIZE_LEGEND: int = 10

    def __post_init__(self):
        """Alapértelmezett színek inicializálása ha nincsenek megadva."""
        if self.COLORS is None:
            self.COLORS = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]


# Globális konfiguráció
plot_config = PlotConfig()


class QueryBuilder:
    """Paraméterezett SQL lekérdezések építése SQL injection megelőzésére."""

    @staticmethod
    def get_panel_data(panel_szam: str, include_outliers: bool = False,
                       limit: Optional[int] = None) -> Tuple[str, List[Any]]:
        """
        Lekérdezés építése panel hőmérséklet adatokhoz.

        Args:
            panel_szam: Panel azonosító
            include_outliers: Outlierek is bekerüljenek-e
            limit: Maximum sorok száma

        Returns:
            (sql_lekérdezés, paraméterek) tuple
        """
        outlier_filter = "" if include_outliers else "AND m.kiugro_ertek = FALSE"
        limit_clause = "LIMIT %s" if limit else ""
        params = [panel_szam]
        if limit:
            params.append(limit)

        sql = f"""
            SELECT m.ts, m.ertek_num
            FROM meres m
            JOIN panel p ON m.panel_id = p.panel_id
            WHERE p.panel_szam = %s
              AND m.ertek_num IS NOT NULL
              {outlier_filter}
            ORDER BY m.ts
            {limit_clause}
        """
        return sql, params

    @staticmethod
    def get_outlier_data(panel_szam: str, limit: int = 500) -> Tuple[str, List[Any]]:
        """Lekérdezés építése outlier adatokhoz."""
        sql = """
            SELECT m.ts, m.ertek_num_original
            FROM meres m
            JOIN panel p ON m.panel_id = p.panel_id
            WHERE p.panel_szam = %s
              AND m.kiugro_ertek = TRUE
            ORDER BY m.ts
            LIMIT %s
        """
        return sql, [panel_szam, limit]

    @staticmethod
    def get_adag_data(adag_szam: int, panel_szam: str) -> Tuple[str, List[Any]]:
        """Lekérdezés építése adag (batch) adatokhoz."""
        sql = """
            SELECT m.ts, m.ertek_num
            FROM meres m
            JOIN panel p ON m.panel_id = p.panel_id
            JOIN adag a ON m.adag_id = a.adag_id
            WHERE a.adag_szam = %s
              AND p.panel_szam = %s
              AND m.kiugro_ertek = FALSE
              AND m.ertek_num IS NOT NULL
            ORDER BY m.ts
        """
        return sql, [adag_szam, panel_szam]

    @staticmethod
    def get_adag_info(adag_szam: int) -> Tuple[str, List[Any]]:
        """Lekérdezés építése adag információkhoz."""
        sql = """
            SELECT adag_szam, kezdet_ts, vege_ts
            FROM adag
            WHERE adag_szam = %s
        """
        return sql, [adag_szam]

    @staticmethod
    def get_available_adagok(limit: int = 10) -> Tuple[str, List[Any]]:
        """Lekérdezés építése elérhető adagokhoz."""
        sql = """
            SELECT adag_szam, kezdet_ts, vege_ts,
                   EXTRACT(EPOCH FROM (vege_ts - kezdet_ts))/60 as idotartam_perc
            FROM adag
            WHERE kezdet_ts IS NOT NULL AND vege_ts IS NOT NULL
            ORDER BY adag_szam
            LIMIT %s
        """
        return sql, [limit]

    @staticmethod
    def get_panels_in_adag(adag_szam: int) -> Tuple[str, List[Any]]:
        """Lekérdezés építése egy adott adagban lévő panelekhez."""
        sql = """
            SELECT p.panel_szam
            FROM meres m
            JOIN panel p ON m.panel_id = p.panel_id
            JOIN adag a ON m.adag_id = a.adag_id
            WHERE a.adag_szam = %s
            GROUP BY p.panel_szam
            ORDER BY p.panel_szam::int
        """
        return sql, [adag_szam]

    @staticmethod
    def get_panel_statistics(panel_szam: str) -> Tuple[str, List[Any]]:
        """Lekérdezés építése panel statisztikákhoz."""
        sql = """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN kiugro_ertek = FALSE THEN 1 END) as clean,
                COUNT(CASE WHEN kiugro_ertek = TRUE THEN 1 END) as outliers,
                MIN(CASE WHEN kiugro_ertek = FALSE THEN ertek_num END) as min_temp,
                MAX(CASE WHEN kiugro_ertek = FALSE THEN ertek_num END) as max_temp,
                AVG(CASE WHEN kiugro_ertek = FALSE THEN ertek_num END) as avg_temp,
                STDDEV(CASE WHEN kiugro_ertek = FALSE THEN ertek_num END) as std_temp
            FROM meres m
            JOIN panel p ON m.panel_id = p.panel_id
            WHERE p.panel_szam = %s
        """
        return sql, [panel_szam]


class ChartBuilder:
    """Segédosztály grafikonok építéséhez közös stílussal."""

    def __init__(self, config: PlotConfig = plot_config):
        """Inicializálás konfigurációval."""
        self.config = config

    def create_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """Ábra létrehozása beállított mérettel."""
        return plt.subplots(figsize=self.config.FIGURE_SIZE)

    def format_chart(self, ax: plt.Axes, title: str, xlabel: str = "Időpont",
                     ylabel: str = "Homerseklet (C)") -> None:
        """Közös formázás alkalmazása grafikonra."""
        ax.set_xlabel(xlabel, fontsize=self.config.FONT_SIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=self.config.FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=self.config.FONT_SIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Csak akkor jelenítsük meg a legendát, ha vannak labelezett elemek
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best', fontsize=self.config.FONT_SIZE_LEGEND)

        # X-tengely idoformatum javitva: tobb cimke, jobb formatummal
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Minden 2. ora
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Minor tick minden oraban
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    def plot_panel_data(self, ax: plt.Axes, df: pd.DataFrame, panel_id: str,
                        color: str, label: Optional[str] = None) -> None:
        """Panel adatok ábrázolása a tengelyen."""
        if label is None:
            label = f'Panel {panel_id}'
        ax.plot(df['ts'], df['temp'], linewidth=1.0, alpha=0.8,
                label=label, color=color)

    def plot_outliers(self, ax: plt.Axes, df: pd.DataFrame, panel_id: str,
                      color: str) -> None:
        """Outlier pontok ábrázolása a tengelyen."""
        ax.scatter(df['ts'], df['temp'], s=30, alpha=0.6,
                   color=color, marker='x', label=f'Panel {panel_id} (outlierek)')

    def add_vertical_line(self, ax: plt.Axes, x_pos: Any, color: str,
                          linestyle: str, label: str) -> None:
        """Függőleges vonal hozzáadása a grafikonhoz."""
        ax.axvline(x=x_pos, color=color, linestyle=linestyle, linewidth=2,
                   alpha=0.7, label=label)


class TemperatureVisualizer:
    """Fő osztály hőmérséklet adatok vizualizációjához."""

    def __init__(self, db: DB, config: PlotConfig = plot_config):
        """Vizualizáló inicializálása."""
        self.db = db
        self.config = config
        self.query_builder = QueryBuilder()
        self.chart_builder = ChartBuilder(config)

    def show_panel_with_time(self, panel_ids: List[str] | str, limit: int = None,
                             show_outliers: bool = False) -> None:
        """
        Egy vagy több panel hőmérséklet adatainak megjelenítése időbélyegekkel.

        Args:
            panel_ids: Panel azonosítók listája vagy egyetlen string
            limit: Maximum mérések száma panelonként
            show_outliers: Outlierek megjelenítése
        """
        if isinstance(panel_ids, str):
            panel_ids = [panel_ids]

        if limit is None:
            limit = self.config.DEFAULT_LIMIT

        fig, ax = self.chart_builder.create_figure()

        plotted_count = 0
        missing_panels = []

        for idx, panel_id in enumerate(panel_ids):
            color = self.config.COLORS[idx % len(self.config.COLORS)]

            # Tiszta adatok lekérése
            sql, params = self.query_builder.get_panel_data(panel_id, include_outliers=False, limit=limit)
            clean_data = self.db.get_data(sql, params)

            if clean_data:
                df = pd.DataFrame([(row[0], row[1]) for row in clean_data], columns=['ts', 'temp'])
                self.chart_builder.plot_panel_data(ax, df, panel_id, color)
                plotted_count += 1

                # Outlierek megjelenítése ha engedélyezett
                if show_outliers:
                    sql_outlier, params_outlier = self.query_builder.get_outlier_data(
                        panel_id, limit=self.config.OUTLIER_LIMIT
                    )
                    outlier_data = self.db.get_data(sql_outlier, params_outlier)

                    if outlier_data:
                        df_outlier = pd.DataFrame([(row[0], row[1]) for row in outlier_data],
                                                  columns=['ts', 'temp'])
                        self.chart_builder.plot_outliers(ax, df_outlier, panel_id, color)
            else:
                missing_panels.append(panel_id)

        # Ellenőrizzük hogy van-e megjeleníthető adat
        if plotted_count == 0:
            plt.close(fig)
            print(f"[HIBA] Nincs megjelenitheto adat a megadott panel(ek)re!")
            if missing_panels:
                print(f"   Nem található adat a következő panel(ek)re: {', '.join(missing_panels)}")
            return

        # Ha vannak hiányzó panelek, jelezzük
        if missing_panels:
            print(f"[FIGYELEM] Nincs adat a kovetkezo panel(ek)re: {', '.join(missing_panels)}")

        # Grafikon formázása
        title = (f'Panel {panel_ids[0]} Hőmérséklet - Időbélyeggel' if len(panel_ids) == 1
                 else f'{len(panel_ids)} Panel Hőmérséklet Összehasonlítása')
        self.chart_builder.format_chart(ax, title)
        plt.show()

    def compare_panels(self, panel_ids: Optional[List[str]] = None,
                       limit: int = None) -> None:
        """
        Több panel hőmérsékleteinek összehasonlítása.

        Args:
            panel_ids: Panel azonosítók listája (interaktív ha None)
            limit: Maximum mérések panelonként
        """
        if panel_ids is None:
            # Interaktív panel választás
            available_panels = self.db.get_data(
                "SELECT panel_szam FROM panel ORDER BY panel_szam::int"
            )
            panel_list = [row[0] for row in available_panels]

            print("\n" + "=" * 60)
            print("PANEL ÖSSZEHASONLÍTÁS")
            print("=" * 60)
            print(f"Elérhető panelek: {', '.join(panel_list)}")
            panel_input = input("Melyik paneleket szeretnéd összehasonlítani? (Enter = osszes, vagy pl: 1,5,6,8): ")

            if panel_input.strip():
                panel_ids = [p.strip() for p in panel_input.split(',')]
            else:
                # Osszes panel
                panel_ids = panel_list
                print(f"[INFO] Osszes panel kivalasztva: {', '.join(panel_ids)}")

        if not panel_ids:
            print("[HIBA] Nem valasztottal panelt!")
            return

        if limit is None:
            limit = self.config.COMPARISON_LIMIT

        print(f"\n[INFO] {len(panel_ids)} panel osszehasonlitasa...")
        self.show_panel_with_time(panel_ids, limit=limit)

    def show_adag_temperature(self, adag_szam: Optional[int] = None,
                              panel_ids: Optional[List[str]] = None) -> None:
        """
        Egy adott adag (batch) hőmérsékleteinek megjelenítése.

        Args:
            adag_szam: Adag szám (interaktív ha None)
            panel_ids: Panel azonosítók listája (interaktív ha None)
        """
        if adag_szam is None:
            adag_szam = self._select_adag_interactive()
            if adag_szam is None:
                return

        # Adag info lekérése
        sql, params = self.query_builder.get_adag_info(adag_szam)
        adag_info = self.db.get_data(sql, params)

        if not adag_info:
            print(f"[HIBA] Nem található adag: {adag_szam}")
            return

        adag_start = adag_info[0][1]
        adag_end = adag_info[0][2]

        print(f"\n[INFO] Adag #{adag_szam} adatainak megjelenítése...")
        print(f"   Időtartam: {adag_start} -> {adag_end}")

        # Panel kiválasztás
        if panel_ids is None:
            panel_ids = self._select_panels_for_adag(adag_szam)

        if not panel_ids:
            # Ellenőrizzük van-e bármilyen mérés az adagban
            sql = "SELECT COUNT(*) FROM meres m JOIN adag a ON m.adag_id = a.adag_id WHERE a.adag_szam = %s"
            count_result = self.db.get_data(sql, [adag_szam])
            measurement_count = count_result[0][0] if count_result else 0

            if measurement_count == 0:
                print(f"[HIBA] Az Adag #{adag_szam} üres - nincs egyetlen mérés sem ebben az időszakban!")
                print(f"   [TIP] Válassz egy másik adagot, amely tartalmaz méréseket (nem üres).")
            else:
                print("[HIBA] Nincsenek elerheto panelek ehhez az adaghoz!")
            return

        # Grafikon létrehozása
        fig, ax = self.chart_builder.create_figure()

        plotted_count = 0
        missing_panels = []

        for idx, panel_id in enumerate(panel_ids):
            color = self.config.COLORS[idx % len(self.config.COLORS)]
            sql, params = self.query_builder.get_adag_data(adag_szam, panel_id)
            data = self.db.get_data(sql, params)

            if data:
                df = pd.DataFrame([(row[0], row[1]) for row in data], columns=['ts', 'temp'])
                self.chart_builder.plot_panel_data(ax, df, panel_id, color)
                plotted_count += 1
                print(f"   [OK] Panel {panel_id}: {len(data)} mérés")
            else:
                missing_panels.append(panel_id)

        # Ellenőrizzük hogy van-e megjeleníthető adat
        if plotted_count == 0:
            plt.close(fig)
            print(f"[HIBA] Nincs megjeleníthető adat az Adag #{adag_szam}-ban a megadott panel(ek)re!")
            if missing_panels:
                print(f"   Nem található adat a következő panel(ek)re: {', '.join(missing_panels)}")
            return

        # Ha vannak hiányzó panelek, jelezzük
        if missing_panels:
            print(f"[FIGYELEM] Nincs adat az adagban a kovetkező panel(ek)re: {', '.join(missing_panels)}")

        # Adag határ vonalak hozzáadása
        self.chart_builder.add_vertical_line(ax, adag_start, 'green', '--', 'Adag kezdet')
        self.chart_builder.add_vertical_line(ax, adag_end, 'red', '--', 'Adag vége')

        # Grafikon formázása
        title = f'Adag #{adag_szam} - Hőmérséklet Adatok ({plotted_count} panel)'
        self.chart_builder.format_chart(ax, title)
        plt.show()

    def show_panel_statistics(self, panel_id: str) -> None:
        """
        Panel statisztikák megjelenítése.

        Args:
            panel_id: Panel azonosító
        """
        sql, params = self.query_builder.get_panel_statistics(panel_id)
        stats = self.db.get_data(sql, params)

        if stats:
            row = stats[0]
            print(f"\n{'=' * 60}")
            print(f"PANEL {panel_id} STATISZTIKÁK")
            print(f"{'=' * 60}")
            print(f"Összes mérés:       {row[0]:,}")
            print(f"Tisztított mérés:   {row[1]:,} ({row[1] / row[0] * 100:.1f}%)")
            print(f"Outlierek:          {row[2]:,} ({row[2] / row[0] * 100:.1f}%)")
            print(f"-" * 60)
            print(f"Min hőmérséklet:    {row[3]:.2f} C")
            print(f"Max hőmérséklet:    {row[4]:.2f} C")
            print(f"Atlag hőmérséklet:  {row[5]:.2f} C")
            print(f"Szórás:             {row[6]:.2f} C")
            print(f"{'=' * 60}\n")

    def _select_adag_interactive(self) -> Optional[int]:
        """Interaktív adag választás."""
        sql, params = self.query_builder.get_available_adagok(limit=50)
        available_adagok = self.db.get_data(sql, params)

        print("\n" + "=" * 80)
        print("ADAG-ALAPÚ SZŰRÉS")
        print("=" * 80)
        print(f"{'Adag':<8} {'Kezdet':<20} {'Vég':<20} {'Időtartam':<15} {'Mérések':<12}")
        print("-" * 80)

        non_empty_count = 0
        for row in available_adagok:
            adag_num = row[0]
            start = row[1].strftime('%Y-%m-%d %H:%M') if row[1] else 'N/A'
            end = row[2].strftime('%Y-%m-%d %H:%M') if row[2] else 'N/A'
            duration = f"{int(row[3])} perc" if row[3] else 'N/A'

            # Mérések száma az adagban
            count_sql = "SELECT COUNT(*) FROM meres m JOIN adag a ON m.adag_id = a.adag_id WHERE a.adag_szam = %s"
            count_result = self.db.get_data(count_sql, [adag_num])
            measurement_count = count_result[0][0] if count_result else 0
            count_str = f"{measurement_count:,}" if measurement_count > 0 else "ÜRES"

            if measurement_count > 0:
                non_empty_count += 1

            print(f"{adag_num:<8} {start:<20} {end:<20} {duration:<15} {count_str:<12}")

        print(f"\n(Összesen {len(available_adagok)} adag mutatva, ebből {non_empty_count} tartalmaz méréseket)")
        print("[TIP] Válassz olyan adagot, amelyikben vannak meresek (nem üres)!")
        adag_input = input("\nMelyik adagot szeretnéd megjeleníteni? (adag szám): ")

        try:
            return int(adag_input)
        except ValueError:
            print("[HIBA] Érvénytelen adag szám!")
            return None

    def _select_panels_for_adag(self, adag_szam: int) -> List[str]:
        """Panelek kiválasztása egy adott adaghoz."""
        panel_input = input("Mely paneleket szeretnéd látni? (Enter = összes, vagy pl: 1,5,6): ")

        if panel_input.strip():
            return [p.strip() for p in panel_input.split(',')]
        else:
            # Összes panel az adagban
            sql, params = self.query_builder.get_panels_in_adag(adag_szam)
            panel_data = self.db.get_data(sql, params)
            return [row[0] for row in panel_data]


def main_menu() -> None:
    """Interaktív főmenü."""
    db = DB()
    visualizer = TemperatureVisualizer(db)

    while True:
        print("\n" + "=" * 60)
        print("HŐMÉRSÉKLET VIZUALIZÁCIÓ - FŐMENÜ")
        print("=" * 60)
        print("1. Egy panel megjelenítése (időbélyeggel)")
        print("2. Több panel összehasonlítása")
        print("3. Adag-alapú szűrés")
        print("4. Panel statisztikák")
        print("5. Kilépés")
        print("=" * 60)

        choice = input("\nVálassz egy opciót (1-5): ")

        if choice == '1':
            panel_id = input("Panel azonosító (pl: 5): ")
            show_outliers_input = input("Mutassa az outliereket is? (i/n): ").lower()
            show_outliers = show_outliers_input == 'i'
            limit_input = input(f"Maximum hány mérés? (Enter = {plot_config.DEFAULT_LIMIT}): ")
            limit = int(limit_input) if limit_input.strip() else None

            visualizer.show_panel_with_time(panel_id, limit=limit, show_outliers=show_outliers)

        elif choice == '2':
            visualizer.compare_panels()

        elif choice == '3':
            visualizer.show_adag_temperature()

        elif choice == '4':
            panel_id = input("Panel azonosító (pl: 5): ")
            visualizer.show_panel_statistics(panel_id)

        elif choice == '5':
            print("\nViszlát!")
            break

        else:
            print("[HIBA] Érvénytelen választás! Probáld újra!")


if __name__ == "__main__":
    print("\nHőmérséklet Vizualizációs Eszköz")
    print("=" * 60)

    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram megszakítva. Viszlát!")
    except Exception as e:
        print(f"\n[HIBA] Hiba történt: {e}")
        import traceback

        traceback.print_exc()

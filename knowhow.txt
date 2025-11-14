requirements.txt:

A projekt futtatásához szükséges Python-csomagok telepítése az alábbi parancsok valamelyikével végezhető el:

•	Windows: pip install -r requirements.txt
•	Linux: pip3 install -r requirements.txt / pip install -r requirements.txt

.env
A projekt tartalmaz egy .env.example nevű mintakonfigurációs fájlt. Ennek használatához azt .env névre át kell nevezni és a megfelelő konfigurációs értékekkel ki kell tölteni.

Ha a helyi gépen van az adatbázisszerver, akkor a localhost-ot kell átírni annak a címére
PGHOST=localhost
Az alapértelmezett port az 5432-es, ha a telepítéskor máshová tettük, itt átírható
PGPORT=5432
Az általunk választott adatbázis neve
PGDATABASE=beadando
A felhasználónév
PGUSER=postgres
A felhasználó jelszava. Ezt mindenképpen meg kell változtatni, mert nincs alapértelmezett jelszó:
PGPASSWORD=

main.py:

A program az adatok előfeldolgozás nélküli beolvasását, valamint azok adatbázisba történő mentését végzi.

db_cleanup.py:

A script beolvassa a rendelkezésre álló CSV-fájlokat, elvégzi az adattisztítást, majd a tisztított adatokat az adatbázisba tölti. A modul egyszer futtatandó.

db.py:

Ez a modul az alkalmazás adatbázis-kezelési rétege, amelyet a show_temperatures.py használ. Absztrakciós réteget biztosít annak érdekében, hogy a projekt többi komponense az adatbázist annak belső struktúrájának ismerete nélkül érhesse el. Valós projektben ezen a rétegen keresztül történne az adattisztítás és az adatfeltöltés is.

show_temperatures.py:

A modul egy egyszerű, szöveges, interaktív menürendszert biztosít, amelynek segítségével a már megtisztított és betöltött adatok diagramon jeleníthetők meg. A modul tetszőleges alkalommal lefuttatható, azonban használatát minden esetben meg kell előzze a db_cleanup.py futtatása. 
Az alkalmazás a vizuális megjelenítés révén hatékony támogatást nyújt az adattisztítási folyamat során, mivel a hibás vagy kiugró értékek ilyen módon könnyebben azonosíthatók.

#!/usr/bin/env bash

PASS="msSQL2017!"

echo "5 - Add backup to system"

docker exec -it sql2 /opt/mssql-tools/bin/sqlcmd -S localhost \
   -U SA -P "${PASS}" \
   -Q 'RESTORE FILELISTONLY FROM DISK = "/var/opt/mssql/backup/AdventureWorks2017.bak"' \
   | tr -s ' ' | cut -d ' ' -f 1-2

echo "6 - Restore DB"

docker exec -it sql2 /opt/mssql-tools/bin/sqlcmd \
   -S localhost -U SA -P "${PASS}" \
   -Q 'RESTORE DATABASE AdventureWorks2017 FROM DISK = "/var/opt/mssql/backup/AdventureWorks2017.bak" WITH MOVE "AdventureWorks2017" TO "/var/opt/mssql/data/AdventureWorks2017.mdf", MOVE "AdventureWorks2017_Log" TO "/var/opt/mssql/data/AdventureWorks2017.ldf", REPLACE'

echo "7 - Query databases"

docker exec -it sql2 /opt/mssql-tools/bin/sqlcmd \
	-S localhost -U SA -P "${PASS}" \
	-Q 'SELECT Name FROM sys.Databases'


#!/usr/bin/env bash

PASS="msSQL2017!"

echo "5 - Add backup to system"

docker exec -it sql1 /opt/mssql-tools/bin/sqlcmd -S localhost \
   -U SA -P "${PASS}" \
   -Q 'RESTORE FILELISTONLY FROM DISK = "/var/opt/mssql/backup/wwi.bak"' \
   | tr -s ' ' | cut -d ' ' -f 1-2

echo "6 - Restore DB"

docker exec -it sql1 /opt/mssql-tools/bin/sqlcmd \
   -S localhost -U SA -P "${PASS}" \
   -Q 'RESTORE DATABASE WideWorldImporters FROM DISK = "/var/opt/mssql/backup/wwi.bak" WITH MOVE "WWI_Primary" TO "/var/opt/mssql/data/WideWorldImporters.mdf", MOVE "WWI_UserData" TO "/var/opt/mssql/data/WideWorldImporters_userdata.ndf", MOVE "WWI_Log" TO "/var/opt/mssql/data/WideWorldImporters.ldf", MOVE "WWI_InMemory_Data_1" TO "/var/opt/mssql/data/WideWorldImporters_InMemory_Data_1"'

echo "7 - Query databases"

docker exec -it sql1 /opt/mssql-tools/bin/sqlcmd \
	-S localhost -U SA -P "${PASS}" \
	-Q 'SELECT Name FROM sys.Databases'


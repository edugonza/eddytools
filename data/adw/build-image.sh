#!/usr/bin/env bash

PASS="msSQL2017!"

echo "1 - Pulling image:"

docker pull microsoft/mssql-server-linux:2017-latest

echo "2 - Running image:"

docker run -e 'ACCEPT_EULA=Y' -e "MSSQL_SA_PASSWORD=${PASS}" \
   --name 'sql2' -p 1402:1433 \
   -d microsoft/mssql-server-linux:2017-latest
#   -v sql1data:/var/opt/mssql \

echo "3 - Creating backup dir"

docker exec -it sql2 mkdir /var/opt/mssql/backup

if ! md5sum -c AdventureWorks2017.md5
then
	echo "3.1 - Downloading backup"
	curl -L -o AdventureWorks2017.bak 'https://github.com/Microsoft/sql-server-samples/releases/download/adventureworks/AdventureWorks2017.bak'
fi

echo "4 - Copying backup into container"

docker cp AdventureWorks2017.bak sql2:/var/opt/mssql/backup


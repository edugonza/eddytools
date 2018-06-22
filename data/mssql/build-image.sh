#!/usr/bin/env bash

PASS="msSQL2017!"

echo "1 - Pulling image:"

docker pull microsoft/mssql-server-linux:2017-latest

echo "2 - Running image:"

docker run -e 'ACCEPT_EULA=Y' -e "MSSQL_SA_PASSWORD=${PASS}" \
   --name 'sql1' -p 1401:1433 \
   -v sql1data:/var/opt/mssql \
   -d microsoft/mssql-server-linux:2017-latest

echo "3 - Creating backup dir"

docker exec -it sql1 mkdir /var/opt/mssql/backup

cat wwi-pieces/* > wwi.bak

if ! md5sum -c wwi.md5
then
	echo "3.1 - Downloading backup"
	curl -L -o wwi.bak 'https://github.com/Microsoft/sql-server-samples/releases/download/wide-world-importers-v1.0/WideWorldImporters-Full.bak'
fi

echo "4 - Copying backup into container"

docker cp wwi.bak sql1:/var/opt/mssql/backup


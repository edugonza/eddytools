#!/usr/bin/env bash
BUILD_DATA=1
HOST_PORT=5556
POSTGRES_PASSWORD='postgres'
docker run --name ds2 -p $HOST_PORT:5432 \
	-e BUILD_DATA=$BUILD_DATA \
	-e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
	-d postgres/ds2
sleep 20
./check_health.sh
docker exec -u postgres ds2 /setup.sh

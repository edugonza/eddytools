BUILD_DATA=1
HOST_PORT=5556
POSTGRES_PASSWORD='postgres'
HOST_PGDATA=`pwd`'/postgresql'
docker run --name ds2 -p $HOST_PORT:5432 \
	-e BUILD_DATA=$BUILD_DATA \
	-e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
	-v $HOST_PGDATA:/var/lib/postgresql/data \
	-d postgres/ds2

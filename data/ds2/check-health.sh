#!/usr/bin/env bash

HEALTH="no"

while [ "$HEALTH" != "\"healthy\"" ]; do
	docker logs --tail 100 ds2
	sleep 5
	HEALTH=`docker inspect --format="{{json .State.Health.Status}}" ds2`
	echo "Health check for ds2: ${HEALTH}"
done

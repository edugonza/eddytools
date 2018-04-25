#!/bin/bash

HEALTH="no"

while [ "$HEALTH" != "\"healthy\"" ]; do
	sleep 5
	HEALTH=`docker inspect --format="{{json .State.Health.Status}}" ds2`
done

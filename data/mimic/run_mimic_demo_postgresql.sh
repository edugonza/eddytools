#!/usr/bin/env bash

PATH_MIMIC_DATA=$1

docker run \
	--name mimic \
	-p 5555:5432 \
	-e BUILD_MIMIC=0 \
	-e POSTGRES_PASSWORD=postgres \
	-e MIMIC_PASSWORD=mimic \
	-v $PATH_MIMIC_DATA/demo/physionet.org/works/MIMICIIIClinicalDatabaseDemo/files/version_1_4:/mimic_data \
	-v $PATH_MIMIC_DATA/demo/postgresql:/var/lib/postgresql/data \
	-d postgres/mimic

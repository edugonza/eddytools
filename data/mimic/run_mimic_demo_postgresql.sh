#!/usr/bin/env bash

docker run \
	--name mimic \
	-p 5555:5432 \
	-e BUILD_MIMIC=0 \
	-e POSTGRES_PASSWORD=postgres \
	-e MIMIC_PASSWORD=mimic \
	-v ../../mimiciii-data/demo/physionet.org/works/MIMICIIIClinicalDatabaseDemo/files/version_1_4:/mimic_data \
	-v ../../mimiciii-data/demo/postgresql:/var/lib/postgresql/data \
	-d postgres/mimic

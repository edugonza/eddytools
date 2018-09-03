#!/usr/bin/env bash

PATH_MIMIC_CODE=$1
PATH_MIMIC_DATA=$2
PWD=`pwd`
cd $PATH_MIMIC_CODE/buildmimic/docker
docker build -t postgres/mimic .
cd $PWD
docker run \
	--name mimic \
	-p 5555:5432 \
	-e BUILD_MIMIC=1 \
	-e POSTGRES_PASSWORD=postgres \
	-e MIMIC_PASSWORD=mimic \
	-v $PATH_MIMIC_DATA/demo/physionet.org/works/MIMICIIIClinicalDatabaseDemo/files/version_1_4:/mimic_data \
	-v $PATH_MIMIC_DATA/demo/postgresql:/var/lib/postgresql/data \
	-d postgres/mimic

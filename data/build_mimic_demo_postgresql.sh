#!/usr/bin/env bash

PWD=`pwd`
cd ../../mimic-code/buildmimic/docker
docker build -t postgres/mimic .
cd $PWD
docker run \
	--name mimic \
	-p 5555:5432 \
	-e BUILD_MIMIC=1 \
	-e POSTGRES_PASSWORD=postgres \
	-e MIMIC_PASSWORD=mimic \
	-v /home/edu/Code/workspace/mimiciii-data/demo/physionet.org/works/MIMICIIIClinicalDatabaseDemo/files/version_1_4:/mimic_data \
	-v /home/edu/Code/workspace/mimiciii-data/demo/postgresql:/var/lib/postgresql/data \
	-d postgres/mimic

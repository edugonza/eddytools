#!/usr/bin/env bash

cd ds21
if  ! md5sum -c ds21.md5
then
        echo "Downloading backup"
        curl -L -o ds21.tar.gz 'https://linux.dell.com/dvdstore/ds21.tar.gz'
fi
cd ..

docker build -t postgres/ds2 .

#!/usr/bin/env bash

#Download appropriate package for the OS version
#Choose only ONE of the following, corresponding to your OS version

DISTRIBUTION=`lsb_release -is`

if ! [ "$DISTRIBUTION" == "Ubuntu" ]
then
	echo "Your distro is: ${DISTRIBUTION}. This script only suports Ubuntu."
	echo "To obtain installation instruction for other platforms, go to https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-2017"
	exit 1
fi

sudo sh -c 'curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -'

sudo sh -c 'RELEASE=`lsb_release -rs`; curl https://packages.microsoft.com/config/ubuntu/${RELEASE}/prod.list > /etc/apt/sources.list.d/mssql-release.list'


sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install msodbcsql17
# optional: for bcp and sqlcmd
sudo ACCEPT_EULA=Y apt-get install mssql-tools
echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile
echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
source ~/.bashrc
# optional: for unixODBC development headers
sudo apt-get install unixodbc-dev

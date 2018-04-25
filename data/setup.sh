#/bin/bash

if [ $BUILD_DATA -eq 1 ]; then
  cd /data
  tar xvzf ds21_postgresql.tar.gz
  tar xvzf ds21.tar.gz

  cd ds2/pgsqlds2
  psql -c "create user ds2 with superuser;"
  psql -c "create database ds2;"
  sh pgsqlds2_create_all.sh
fi

echo "Done!"

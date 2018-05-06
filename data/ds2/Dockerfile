FROM postgres:9

# in the docker initialization, we do not build the data
ENV BUILD_DATA 0

RUN apt-get update
RUN apt-get install -y vim git

#RUN adduser newuser

# make a directory for the data
RUN mkdir /data
RUN chown postgres /data
#USER newuser

# copy the data
COPY ds21/* /data/

ADD setup.sh /docker-entrypoint-initdb.d/

HEALTHCHECK CMD pg_isready -U postgres

#!/usr/bin/bash

docker build -t ensembles_server -f ../Dockerfile ../.
docker compose up
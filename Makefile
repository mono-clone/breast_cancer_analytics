#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Docker commands
cbuild:
	docker build --tag bca-test:latest .
#	docker build --tag breast-cancer-analytics .

crun:
	docker container run \
	-it \
	--mount type=bind,source="${PWD}",target=/home/breast-cancer-analytics \
	-p 8888:8888 \
	--name breast-cancer-analytics \
	bca-test:latest


cclear:
	docker rm breast-cancer-analytics

crestart:
	docker restart breast-cancer-analytics

cexec
	docker container exec -it breast-cancer-analytics bash

# launch jupyter server
jupyter:
	jupyter-lab --ip 0.0.0.0 --allow-root
#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

ifeq (${OS},Windows_NT)
    BASEDIR=$(subst /,\\,$(subst /c,C:,${CURDIR}))
else
    BASEDIR=${CURDIR}
endif

CONTAINERNAME=breast-cancer-analytics

# Docker commands
dbuild:
	docker build --tag ${CONTAINERNAME} .

drun:
	docker container run \
	-it \
	--mount type=bind,source=${BASEDIR},target=/home/breast-cancer-analytics \
	-p 8888:8888 \
	--name ${CONTAINERNAME} \
	${CONTAINERNAME}:latest

dclear:
	docker rm ${CONTAINERNAME}

drestart:
	docker restart ${CONTAINERNAME}

dexec:
	docker container exec -it ${CONTAINERNAME} bash

# launch jupyter server
jupyter:
	jupyter-lab --ip 0.0.0.0 --allow-root
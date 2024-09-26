# Makefile
# IMPORTANT: Please find the right cuda dev container for your environment
SHELL := /bin/bash


BASE_IMG=ubuntu:20.04
# USER INPUT (TODO: PLEASE MODIFY)
# Look for the correct dockerfile matched to your cuda version.
DOCKERFILE := docker/Dockerfile

CODE_PATH ?= $(shell pwd)
DISK_MOUNT_PATH ?= $(shell pwd)

# Custom Image
DOCKER_IMG := vsm
MY_DOCKER_IMG := $(shell echo ${USER} | tr '@.' '_')_${DOCKER_IMG}
TAG := latest

pull_docker_image:
	docker pull ${BASE_IMG}

build_docker_image:
	docker build --build-arg BASE_IMG=${BASE_IMG} . -f ${DOCKERFILE} --network=host --tag ${DOCKER_IMG}:${TAG}

run_dev_docker_container:
	docker run --interactive \
			   --detach \
			   --tty \
			   --name ${MY_DOCKER_IMG} \
			   --cap-add=SYS_PTRACE \
			   --ulimit core=0:0 \
			   --volume ${CODE_PATH}:/opt/vsm \
			   ${DOCKER_IMG}:${TAG} \
			   /bin/bash

exec_docker_container:
	docker exec -it ${MY_DOCKER_IMG} /bin/bash

stop_docker_container:
	docker stop $(MY_DOCKER_IMG)
	docker rm $(MY_DOCKER_IMG)
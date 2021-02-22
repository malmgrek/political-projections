##
# Dimred-demo
#
# @file
# @version 0.1

IMAGENAME=dimred-demo

#
# Install package in virtual env
#
devenv:
	bash bin/setup_devenv.sh

#
# Docker
#

build:
	docker build -t ${IMAGENAME} .

run:
	@docker run -p 8050\:8050 --name demo ${IMAGENAME}

up: build run

stop:
	@docker stop demo


#
# Podman
#

build-podman:
	podman build -t ${IMAGENAME} .

run-podman:
	@podman run -p 8050\:8050 --name demo ${IMAGENAME}

up-podman: build-podman run-podman

stop-podman:
	@podman stop demo


# end

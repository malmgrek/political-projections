##
# Political projections
#
# @file
# @version 0.1

IMAGENAME=political-projections

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

# end

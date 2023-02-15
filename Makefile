#################################################################################
# GLOBALS                                                                       #
#################################################################################
.ONESHELL:

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = Chat-response-suggestions

#################################################################################
# COMMANDS    DEV                                                                  #
#################################################################################
#update the environment after manually changing environment.yml
environment.lock.yml: environment.yml
	conda env update -n $(PROJECT_NAME) -f $< --prune
	conda env export -n $(PROJECT_NAME) | grep -v "^prefix: " > $@  


#shortcuts for Creating the lock file
def_lock: environment.lock.yml

lock: def_lock 

# Create environment from lock file

def_env_create: environment.lock.yml
	conda env update -n $(PROJECT_NAME) -f $< --prune


env_create: def_env_create

# delete environment

def_env_delete:
	conda env remove -n $(PROJECT_NAME)

env_delete: def_env_delete

#################################################################################
# COMMANDS    PROD                                                                  #
#################################################################################

run:
	docker-compose up --build --scale autocomplete=2 --scale semantic_search=2
down:
	docker-compose down



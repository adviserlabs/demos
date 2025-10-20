#!/bin/bash

successfulRun=1

( conda env create -f environment.yml >> output/conda_env_create.log ) || successfulRun=0
( conda activate ctc_test ) || successfulRun=0

if [[ $successfulRun == 1 ]] ; then
	echo " + Adviser: All Parts of Job are Successful."
else
	echo " x Adviser: Some Parts of Job Failed."
	cat output/conda_env_create.log
	exit 1
fi

successfulPython=1

( which python >> output/python_which.log ) || successfulPython=0
( python --version >> output/python_version.log ) || successfulPython=0

if [[ $successfulPython == 1 ]] ; then
	echo " + Adviser: Python sanity check passed."
else
	echo " x Adviser: Python sanity check failed."
	cat output/python*.log
	exit 1
fi


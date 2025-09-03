#!/bin/bash

echo " - This is a sample script to make progress bars happen on adviser status output."

for i in {1..20} ; do
	pctComplete=$(( i * 5 ))
	echo "AdviserRunner::UserJobProgress=$pctComplete"
	sleep 2
done

echo " - This sample script has finished echoing things."


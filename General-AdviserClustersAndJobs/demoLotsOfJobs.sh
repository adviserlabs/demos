#!/bin/bash

echo " - Before doing this, go do this in another terminal:"
echo " adviser status --watch 1"
echo ""
echo " - Running a buncha Jobs!"

sleep 3

# run a buncha jobs
sleep 1 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 2 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 3 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 4 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 3 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 2 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 1 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &

#!/bin/bash

echo " - Before doing this, go do this in another terminal:"
echo " adviser status --watch 1"
echo ""
echo " - Running a buncha Jobs!"

sleep 5

# run a buncha jobs
sleep 5 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 6 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 7 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 8 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 7 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 6 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &
sleep 5 ; adviser run  "bash ./demoAdviserCommandProgress.sh && ls -al" &

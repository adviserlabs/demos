#!/bin/bash

echo " - Before doing this, go do this in another terminal:"
echo " adviser status --watch 1"
echo ""
echo " - Setting up a buncha Clusters!"

sleep 5

# set up 6 clusters
sleep 2 ; adviser  cluster create  --short-name anni     --cloud aws --instance-type t2.medium  --workdir . --region us-west-2  --setup 'echo Adviser' &
sleep 2 ; adviser  cluster create  --short-name brynne   --cloud aws --instance-type t2.medium  --workdir . --region us-west-2  --setup 'echo Adviser' &
sleep 2 ; adviser  cluster create  --short-name cephisto --cloud aws --instance-type t2.medium  --workdir . --region us-west-2  --setup 'echo Adviser' &
sleep 2 ; adviser  cluster create  --short-name darren   --cloud aws --instance-type t2.medium  --workdir . --region us-west-2  --setup 'echo Adviser' &
sleep 2 ; adviser  cluster create  --short-name ellis    --cloud aws --instance-type t2.medium  --workdir . --region us-west-2  --setup 'echo Adviser' &
sleep 8 ; adviser  cluster create  --short-name fez      --cloud aws --instance-type t2.medium  --workdir . --region us-west-2  --setup 'echo Adviser'

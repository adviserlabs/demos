#!/bin/bash

set -e

numClusters=${1-6}
# Array of cluster short names
short_names=(advistar tim david krish brian annya bethany christy denise erica fez grant harold ilya jefferson kevin lenny manny nandy olyvya pandies quendan rita sia tandy uvmap vertex wyn xander ze)

# Check if requested number exceeds available names
if [ $numClusters -gt ${#short_names[@]} ]; then
    echo "Error: Requested $numClusters clusters, but only ${#short_names[@]} short names are available."
    echo "Maximum supported clusters: ${#short_names[@]}"
    exit 1
fi

echo " - Before doing this, go do this in another terminal:"
echo " adviser status --watch 1"
echo ""
echo " - Setting up $numClusters Clusters! (You can specify a different number if you want)"
echo " - Like so: \"$0 10\" for 10."
echo ""

sleep 3

# Use only the first numClusters names from the array
clusters_to_create=("${short_names[@]:0:$numClusters}")

# Create clusters in a loop
for i in "${!clusters_to_create[@]}"; do
    short_name="${clusters_to_create[$i]}"

    if [ $i -eq $((${#clusters_to_create[@]} - 1)) ]; then
        # Last cluster - don't background it
        sleep 5
        adviser cluster create --short-name "$short_name" --cloud aws --instance-type t2.medium --workdir . --region us-west-2 --setup 'echo Adviser'
    else
        # Background all other clusters
        sleep 0.5
        adviser cluster create --short-name "$short_name" --cloud aws --instance-type t2.medium --workdir . --region us-west-2 --setup 'echo Adviser' &
    fi
done

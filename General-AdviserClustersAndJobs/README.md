# Clusters and Jobs Visual Demo
These are meant to be run while doing `adviser status --watch 1` in some terminal so that you can see
clusters being built, then jobs being run.

`demoLotsOfClusters.sh` - Creates many clusters (6 by default, up to about 40).

`demoLotsOfJobs.sh` - Creates 6 jobs. Could be easily modified to do many more simultaneously.

`demoAdviserCommandProgress.sh` - A simple command that returns progress % to show job state in `adviser status`.

`demoCondaSetup.sh` - A simple script to set up Conda environment.

## Running the Demonstration
`bash ./demoLotsOfClusters.sh`


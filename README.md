# Demos
Demos for Clients, Potential Clients, and Whatever

### Clusters and Jobs Visual Demo
These are meant to be run while doing `adviser status --watch 1` in some terminal so that you can see
clusters being built, then jobs being run.

`demoLotsOfClusters.sh` - Creates 6 clusters (by default, up to about 40)

`demoLotsOfJobs.sh` - Creates 6 jobs. Could be easily modified to do many more simultaneously.

### Stock Analyzer
`stockAnalyzer.py` - AI wrote this. It has zero dependencies. Creates a bunch of fake tickers, generates trade
data for the over a long period, then does some analysis on the biggest winners/losers. Outputs charts so
it actually looks like something.

### Protein/Ligand Binding Affinities
`biomedMolecularDynamicsSim.py` - Another AI-written simulation. Creates a bunch of proteins and ligands and
then creates some mock data for binding affinity then generates charts and graphs. Data is written into a
TSV in case we wanted to replace it with some actual data?

### MKL
`./mklSampleProggy.py` - AI wrote this. Looks like it does a 500x500 matrix multiplication then 500x500 Eigen decomposition.

On the server, required:

```bash
sudo apt-get update
sudo apt-get install intel-mkl
pip install mkl
pip install mkl-service
```

Example cluster setup command:

```bash
adviser cluster create --cloud=aws --setup="apt-get update && DEBIAN_FRONTEND=noninteractive apt install -y intel-mkl && pip install mkl mkl-service"
```

### Mosek
`./mosekLinearOptimisation.py` - AI wrote this. A simple linear optimisation problem.

On the server, required:

```bash
pip install mosek
mkdir ~/mosek
vim ~/mosek/mosek.lic   # paste a valid Mosek license in
```

### Fenics
`./fenicsDemo.py` - Retrieved from the [Fenics/Dolfinx](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos/demo_poisson.html) page.

On the server, required:

```bash
conda create -n fenicsx-env
conda activate fenicsx-env
conda env list
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

This generates an image locally. Unsure how to convert this into a demo. Probably modify the script to output the PNG/image onto S3 bucket.

### Starting with the CLI
This gets a functional cluster for running Mosek+MKL demos:

```bash
adviser cluster create --cloud=azure \
    --setup="mkdir ~/mosek ; mv mosek.lic ~/mosek/ \
             && sudo apt-get update \
             && sudo DEBIAN_FRONTEND=\"noninteractive\" apt-get -y install intel-mkl \
             && pip install -r requirements.txt"
```

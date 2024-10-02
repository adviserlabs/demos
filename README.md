# Demos
Demos for Clients, Potential Clients, and Whatever

### MKL
./mklSampleProggy.py - AI wrote this. Looks like it does a 500x500 matrix multiplication then 500x500 Eigen decomposition.

On the server, required:

```bash
sudo apt-get update
sudo apt-get install intel-mkl
pip install mkl
pip install mkl-service
```

### Mosek
./mosekLinearOptimisation.py - AI wrote this. A simple linear optimisation problem.

On the server, required:

```bash
pip install mosek
mkdir ~/mosek
vim ~/mosek/mosek.lic   # paste a valid Mosek license in
```

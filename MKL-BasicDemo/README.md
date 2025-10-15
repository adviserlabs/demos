# MKL
`./mklSampleProggy.py` - A 500x500 matrix multiplication then 500x500 Eigen decomposition.

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

# Starting with the CLI
This gets a functional cluster for running Mosek+MKL demos:

```bash
adviser cluster create --cloud=azure \
    --setup="mkdir ~/mosek ; mv mosek.lic ~/mosek/ \
             && sudo apt-get update \
             && sudo DEBIAN_FRONTEND=\"noninteractive\" apt-get -y install intel-mkl \
             && pip install -r requirements.txt"
```

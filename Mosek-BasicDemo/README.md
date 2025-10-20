### Mosek
`./mosekLinearOptimisation.py` - A simple linear optimisation problem.

On the server, required:

```bash
pip install mosek
mkdir ~/mosek
vim ~/mosek/mosek.lic   # paste a valid Mosek license in
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

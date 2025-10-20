# Biomed-NextFlow Demo
A simple NextFlow pipeline demonstrating protein-ligand binding simulation for
biomedical research.

## Overview
This pipeline simulates a basic drug discovery workflow:
1. **Parse Ligands** - Extract molecular properties from candidate drug molecules
2. **Calculate Binding** - Compute binding energies between protein and ligands
3. **Visualize Results** - Generate plots and identify top binding candidates

## Pipeline Structure
```
Biomed-NextFlow/
├── main.nf                 # NextFlow workflow definition
├── nextflow.config         # Pipeline configuration
├── bin/                    # Python scripts
│   ├── parse_ligands.py
│   ├── calculate_binding.py
│   └── visualize.py
├── data/                   # Input data
│   ├── ligands.csv        # Candidate drug molecules
│   └── protein.pdb        # Target protein structure
└── output/                # Results (created during execution)
```

## Requirements
- **Adviser CLI**
- **NextFlow** >= 22.0.0
- **Python** >= 3.9 (with standard library only - no external dependencies!)
- **Java** >= 11 (for NextFlow)

## Quick Start

### Adviser Setup Arg
```bash
curl -s https://get.nextflow.io | bash
```

### Run the Pipeline
```bash
nextflow run main.nf
```

### View Results
`adviser download ...` to download the results from the cluster.

```bash
# View top binding candidates
cat output/visualization/top_candidates.txt

# View binding report
cat output/binding_results/binding_report.txt

# View ASCII visualization
cat output/visualization/binding_plot.png
```

## Output Files
```
output/
├── parsed_ligands/
│   └── ligand_properties.json       # Molecular properties
├── binding_results/
│   ├── binding_energies.csv         # All binding energies
│   └── binding_report.txt           # Summary report
├── visualization/
│   ├── binding_plot.png             # ASCII bar chart
│   └── top_candidates.txt           # Top 3 candidates
├── timeline.html                     # Execution timeline
├── report.html                       # Pipeline report
├── trace.txt                         # Process trace
└── dag.html                          # Pipeline DAG
```

## Understanding the Results
- **Binding Energy** (kcal/mol): Lower (more negative) values indicate stronger binding
- **Kd** (μM): Dissociation constant - lower values indicate higher affinity
- **LogP**: Lipophilicity - affects drug absorption
- **Molecular Weight** (Da): Important for drug-like properties

## Customization

### Use Your Own Data
```bash
nextflow run main.nf \
  --ligands /path/to/your/ligands.csv \
  --protein /path/to/your/protein.pdb \
  --outdir /path/to/output
```

### Ligands CSV Format
```csv
id,name,formula
LIG001,CompoundName,C10H12N2O
```

## Pipeline Reports
NextFlow automatically generates execution reports:

```bash
# View execution timeline
open output/timeline.html

# View resource usage
open output/report.html

# View workflow DAG
open output/dag.html
```


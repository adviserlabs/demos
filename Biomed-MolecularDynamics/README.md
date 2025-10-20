# Protein/Ligand Binding Affinities
`biomedMolecularDynamicsSim.py` - Small molecule protein binding simulation.
Creates a bunch of proteins and ligands and then creates some mock data for
binding affinity then generates charts and graphs. Data is written into a TSV
in case we wanted to replace it with some actual data.

## Running the Demonstration
`adviser run "python biomedMolecularDynamicsSim.py"`

```
=== Generating Synthetic Data ===
Creating mock binding data for proteins and ligands...

✓ Data generation complete! Generated data for 15 proteins and 12 ligands

=== Exporting Data to TSV ===
✓ Exported 180 rows to biomedMolecularDynamicsSim.tsv

=== Running Biomolecular Simulation ===
Analyzing molecular binding dynamics...

=== Binding Affinity Matrix ===
Values: 0.0 (no binding) → 1.0 (strong binding)

Protein   Aspirin       Ibuprofen     Caffeine      Morphine      Dopamine      Serotonin     Acetaminophen Naproxen      Codeine       Theobromine   Epinephrine   Norepinephrine
ACE2      0.567         0.834         0.468         0.661         0.562         0.103         0.895         0.360         0.920         0.652         0.440         0.166
EGFR      0.538         0.561         0.763         0.598         0.791         0.404         0.335         0.362         0.834         0.580         0.698         0.493
p53       0.405         0.275         0.547         0.316         0.866         0.483         0.129         0.589         0.790         0.220         0.180         0.494
BRCA1     0.652         0.920         0.276         0.211         0.295         0.811         0.224         0.784         0.866         0.264         0.611         0.305
TNF-α     0.126         0.324         0.846         0.676         0.429         0.797         0.591         0.269         0.174         0.445         0.862         0.375
HER2      0.399         0.465         0.786         0.559         0.842         0.290         0.315         0.560         0.430         0.443         0.503         0.295
VEGFR     0.834         0.671         0.271         0.568         0.450         0.884         0.829         0.411         0.866         0.206         0.302         0.240
PDGFR     0.243         0.284         0.146         0.587         0.645         0.735         0.190         0.356         0.441         0.782         0.278         0.147
IGF1R     0.864         0.520         0.729         0.360         0.793         0.561         0.749         0.520         0.890         0.110         0.315         0.846
FGFR      0.107         0.302         0.523         0.420         0.573         0.735         0.459         0.744         0.467         0.830         0.573         0.161
MET       0.263         0.631         0.845         0.101         0.268         0.530         0.681         0.678         0.576         0.594         0.886         0.296
ALK       0.403         0.751         0.206         0.367         0.779         0.707         0.870         0.843         0.695         0.212         0.278         0.930
ROS1      0.338         0.623         0.305         0.316         0.931         0.671         0.566         0.611         0.417         0.896         0.259         0.930
BRAF      0.735         0.567         0.851         0.376         0.162         0.439         0.949         0.868         0.865         0.224         0.129         0.815
PIK3CA    0.735         0.115         0.568         0.368         0.385         0.143         0.826         0.134         0.640         0.136         0.648         0.551

=== Simulation Summary Statistics ===

Total protein-ligand pairs: 180
Average binding affinity: 0.522
Maximum binding affinity: 0.949
Minimum binding affinity: 0.101
Strong binders (≥0.7): 51

Top 5 Binding Pairs:
1. BRAF + Acetaminophen: 0.949
2. ROS1 + Dopamine: 0.931
3. ALK + Norepinephrine: 0.930
4. ROS1 + Norepinephrine: 0.930
5. ACE2 + Codeine: 0.920

=== Binding Affinity Distribution ===

0.0-0.2  |--------------- 19

=== Detailed Kinetics Analysis ===

=== Binding Kinetics: ACE2 + Aspirin ===
Initial Binding Affinity: 0.567

1.0 |                    o     oo oooo  oo o oo  o  oo       o   o     o   o   oo o o
    |                      o oo  o    oo  o o   o  o  o oo o   oo  oo o  oo o      o
    |                   o o o                  o  o    o  o o o   o  o  o    oo  o
    |                  +
    |             +  ++
    |            + ++
    |           +
    |          +
    |        -
    |      -- -
    |     -
    |   -
    |  - -
    |-
    | -
0.0 |________________________________________________________________________________
    0                                     Time (steps)                                     100

Legend: O High (≥0.8)  o Medium (≥0.6)  + Low (≥0.3)  - Minimal

=== Binding Kinetics: ACE2 + Ibuprofen ===
Initial Binding Affinity: 0.834

1.0 |                     OOOO    O O  O OOO OO     OO O  OOOO OOOO    O  O  O OOOO O
    |                  OOO    OO O O OO O   O  OOOOO  O OO    O    OOOO OO OO O    O
    |                 O         O
    |
    |               oo
    |             oo
    |           +o
    |         ++
    |
    |       -+
    |     --
    |   --
    | --
    |
    |-
0.0 |________________________________________________________________________________
    0                                     Time (steps)                                     100

Legend: O High (≥0.8)  o Medium (≥0.6)  + Low (≥0.3)  - Minimal

=== Binding Kinetics: EGFR + Aspirin ===
Initial Binding Affinity: 0.538

1.0 |                     oo   oo o   oo                  o     oooo oo oo     ooooo
    |                  o o   o   o oo   oooo o  oooo ooooo ooo      o  o  o  oo     o
    |                   o   o +      +      + ++    o         oo           ++
    |                ++
    |
    |               +
    |           ++++
    |
    |          -
    |      -- -
    |     -  -
    |   -
    |  - -
    |-
    | -
0.0 |________________________________________________________________________________
    0                                     Time (steps)                                     100

Legend: O High (≥0.8)  o Medium (≥0.6)  + Low (≥0.3)  - Minimal

=== Binding Kinetics: EGFR + Ibuprofen ===
Initial Binding Affinity: 0.561

1.0 |                      oo     o   o  o   o  oo  oooo    oo oo       o  o oooo   o
    |                   o o   o oo o    o  o   o  o       o   o  oooo oo  o o    oo
    |                  o o   o o    oo o  o o o    o    oo o         o   o         o
    |                ++
    |               +
    |              +
    |             +
    |          -+-
    |        --
    |       -
    |
    |    ---
    |   -
    | --
    |-
0.0 |________________________________________________________________________________
    0                                     Time (steps)                                     100

Legend: O High (≥0.8)  o Medium (≥0.6)  + Low (≥0.3)  - Minimal

=== Simulation Complete ===
Simulation to demonstrate biomolecular binding affinity dynamics.
Data saved to biomedMolecularDynamicsSim.tsv for future analysis.
```

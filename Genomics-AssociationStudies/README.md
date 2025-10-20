# Genomics - Genome-Wide Association Studies (GWAS)
Analyzes millions of genetic variants across samples on distributed clusters;
outputs SNP significance maps to pinpoint disease-linked genes.

## Running the Demonstration
`adviser run "pip install -r requirements.txt && python genomeWideAssociationStudies.py" --cloud aws --instance-type i3.large`

```
================================================================================
GWAS Demo - Genome-Wide Association Studies
================================================================================

Configuration:
  - Samples: 1,000
  - SNPs: 500,000
  - Causal SNPs: 20
  - Chromosomes: 22
  - Disease prevalence: 30.0%

Output directory: /home/ubuntu/sky_workdir/output

[1/6] Generating synthetic SNP data...
  ✓ Generated 500,000 SNPs across 22 chromosomes
  ✓ Genotype matrix shape: (1000, 500000)

[2/6] Simulating phenotype with causal SNPs...
  ✓ Selected 20 causal SNPs
  ✓ Example causal SNPs: rs202268, rs400082, rs91046, rs203188, rs489870
  ✓ Disease prevalence: 30.0%
  ✓ Cases: 300, Controls: 700

[3/6] Performing association testing...
  ... tested 100,000 SNPs
  ... tested 200,000 SNPs
  ... tested 300,000 SNPs
  ... tested 400,000 SNPs
  ... tested 500,000 SNPs
  ✓ Association testing complete
  ✓ Genomic inflation factor (λ): 1.013
  ✓ Significant SNPs (Bonferroni corrected): 0

[4/6] Creating Manhattan plot...
  ✓ Manhattan plot saved to output/manhattan_plot.png

[5/6] Creating Q-Q plot...
  ✓ Q-Q plot saved to output/qq_plot.png

[6/6] Saving results...
  ✓ Top 100 SNPs saved to output/top_100_snps.csv
  ✓ Causal SNP results saved to output/causal_snps_results.csv
  ✓ Full results saved to output/all_snps_results.csv.gz
  ✓ Summary statistics saved to output/summary_statistics.csv

================================================================================
GWAS Analysis Summary
================================================================================
Total SNPs tested: 500,000
Significant SNPs (Bonferroni): 0
True causal SNPs: 20
Causal SNPs detected: 0/20 (0.0%)
Most significant p-value: 3.02e-07

Top 5 most significant SNPs:
     SNP  Chromosome     Position      P_value  Odds_Ratio  Is_Causal
rs348625          16 8.430163e+07 3.015251e-07    2.817344       True
rs181813           8 2.499533e+08 8.118997e-07   13.596244      False
rs360755          16 2.183037e+08 4.363125e-06    3.046809      False
 rs19593           1 2.151300e+08 4.891402e-06    4.050000      False
rs445261          20 1.478071e+08 5.693987e-06    1.655187      False
================================================================================

================================================================================
GWAS Demo Complete!
================================================================================

All outputs saved to: /home/ubuntu/sky_workdir/output

Generated files:
  - manhattan_plot.png          : Genome-wide association visualization
  - qq_plot.png                 : Quality control plot
  - top_100_snps.csv            : Top 100 most significant SNPs
  - causal_snps_results.csv     : Statistics for true causal SNPs
  - all_snps_results.csv.gz     : Complete results (compressed)
  - summary_statistics.csv      : Overall analysis summary
```

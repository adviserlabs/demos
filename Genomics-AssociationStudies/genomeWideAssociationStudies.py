#!/usr/bin/env python3
"""
Genome-Wide Association Studies (GWAS) Demo

This demo simulates a GWAS analysis:
1. Generates synthetic genetic variant (SNP) data for multiple samples
2. Simulates phenotypes with causal SNPs
3. Performs association testing
4. Creates Manhattan plots, Q-Q plots, and significance tables
5. Outputs all results to the output/ directory
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
NUM_SAMPLES = 1000          # Number of individuals
NUM_SNPS = 500000          # Number of genetic variants to simulate
NUM_CAUSAL_SNPS = 20       # Number of SNPs that truly affect the phenotype
NUM_CHROMOSOMES = 22       # Number of chromosomes (excluding X, Y)
EFFECT_SIZE = 0.3          # Effect size of causal SNPs
DISEASE_PREVALENCE = 0.3   # 30% of samples have the disease

print("=" * 80)
print("GWAS Demo - Genome-Wide Association Studies")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  - Samples: {NUM_SAMPLES:,}")
print(f"  - SNPs: {NUM_SNPS:,}")
print(f"  - Causal SNPs: {NUM_CAUSAL_SNPS}")
print(f"  - Chromosomes: {NUM_CHROMOSOMES}")
print(f"  - Disease prevalence: {DISEASE_PREVALENCE:.1%}")
print()


def generate_snp_data():
    """
    Generate synthetic SNP data.
    SNPs are encoded as 0, 1, 2 (number of minor alleles).
    """
    print("[1/6] Generating synthetic SNP data...")

    # Minor allele frequencies (MAF) vary across SNPs
    maf = np.random.uniform(0.05, 0.5, NUM_SNPS)

    # Generate genotype data using binomial distribution
    # Each SNP is diploid: 0, 1, or 2 copies of minor allele
    genotypes = np.random.binomial(2, maf, size=(NUM_SAMPLES, NUM_SNPS))

    # Create SNP identifiers distributed across chromosomes
    snps_per_chr = NUM_SNPS // NUM_CHROMOSOMES
    chromosomes = np.repeat(range(1, NUM_CHROMOSOMES + 1), snps_per_chr)

    # Add remaining SNPs to last chromosome
    remaining = NUM_SNPS - len(chromosomes)
    if remaining > 0:
        chromosomes = np.concatenate([chromosomes, np.full(remaining, NUM_CHROMOSOMES)])

    # Generate positions along each chromosome
    positions = []
    for chr_num in range(1, NUM_CHROMOSOMES + 1):
        chr_snps = np.sum(chromosomes == chr_num)
        # Positions range from 0 to ~250Mb
        chr_positions = np.sort(np.random.uniform(0, 250_000_000, chr_snps))
        positions.extend(chr_positions)

    positions = np.array(positions)

    # Create SNP metadata
    snp_names = [f"rs{i}" for i in range(NUM_SNPS)]

    print(f"  ✓ Generated {NUM_SNPS:,} SNPs across {NUM_CHROMOSOMES} chromosomes")
    print(f"  ✓ Genotype matrix shape: {genotypes.shape}")

    return genotypes, snp_names, chromosomes, positions, maf


def simulate_phenotype(genotypes, snp_names):
    """
    Simulate a binary phenotype (case/control) with some causal SNPs.
    """
    print("\n[2/6] Simulating phenotype with causal SNPs...")

    # Randomly select causal SNPs
    causal_indices = np.random.choice(NUM_SNPS, NUM_CAUSAL_SNPS, replace=False)
    causal_snps = [snp_names[i] for i in causal_indices]

    print(f"  ✓ Selected {NUM_CAUSAL_SNPS} causal SNPs")
    print(f"  ✓ Example causal SNPs: {', '.join(causal_snps[:5])}")

    # Calculate genetic risk score based on causal SNPs
    risk_score = np.zeros(NUM_SAMPLES)
    for idx in causal_indices:
        # Each causal SNP contributes to risk
        risk_score += genotypes[:, idx] * EFFECT_SIZE

    # Add random noise
    risk_score += np.random.normal(0, 1, NUM_SAMPLES)

    # Convert to binary phenotype (disease status)
    # Use threshold to achieve desired disease prevalence
    threshold = np.percentile(risk_score, (1 - DISEASE_PREVALENCE) * 100)
    phenotype = (risk_score > threshold).astype(int)

    actual_prevalence = phenotype.mean()
    print(f"  ✓ Disease prevalence: {actual_prevalence:.1%}")
    print(f"  ✓ Cases: {phenotype.sum()}, Controls: {NUM_SAMPLES - phenotype.sum()}")

    return phenotype, causal_indices, causal_snps


def perform_association_testing(genotypes, phenotype, snp_names):
    """
    Perform association testing using chi-square test.
    """
    print("\n[3/6] Performing association testing...")

    p_values = []
    odds_ratios = []

    # Test each SNP for association with phenotype
    for snp_idx in range(NUM_SNPS):
        genotype_col = genotypes[:, snp_idx]

        # Create contingency table for chi-square test
        # Rows: genotype (0, 1, 2), Columns: phenotype (0, 1)
        contingency = np.zeros((3, 2))
        for gt in [0, 1, 2]:
            for phen in [0, 1]:
                contingency[gt, phen] = np.sum((genotype_col == gt) & (phenotype == phen))

        # Chi-square test
        chi2, p_val, _, _ = stats.chi2_contingency(contingency + 1e-10)  # Add small value to avoid zeros
        p_values.append(p_val)

        # Calculate odds ratio (for genotype 2 vs 0)
        cases_with_2 = contingency[2, 1]
        controls_with_2 = contingency[2, 0]
        cases_with_0 = contingency[0, 1]
        controls_with_0 = contingency[0, 0]

        if cases_with_0 > 0 and controls_with_2 > 0:
            or_val = (cases_with_2 * controls_with_0) / (cases_with_0 * controls_with_2 + 1e-10)
        else:
            or_val = 1.0

        odds_ratios.append(or_val)

        # Progress indicator
        if (snp_idx + 1) % 100000 == 0:
            print(f"  ... tested {snp_idx + 1:,} SNPs")

    p_values = np.array(p_values)
    odds_ratios = np.array(odds_ratios)

    # Calculate genomic inflation factor (lambda)
    median_chi2 = np.median(stats.chi2.ppf(1 - p_values, df=1))
    expected_median_chi2 = stats.chi2.ppf(0.5, df=1)
    lambda_gc = median_chi2 / expected_median_chi2

    print(f"  ✓ Association testing complete")
    print(f"  ✓ Genomic inflation factor (λ): {lambda_gc:.3f}")

    # Count significant SNPs
    bonferroni_threshold = 0.05 / NUM_SNPS
    num_significant = np.sum(p_values < bonferroni_threshold)
    print(f"  ✓ Significant SNPs (Bonferroni corrected): {num_significant}")

    return p_values, odds_ratios, lambda_gc


def create_manhattan_plot(chromosomes, positions, p_values, causal_indices, output_dir):
    """
    Create Manhattan plot showing -log10(p-values) across chromosomes.
    """
    print("\n[4/6] Creating Manhattan plot...")

    # Calculate -log10(p-values)
    log_p_values = -np.log10(p_values + 1e-300)  # Avoid log(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot each chromosome
    colors = ['#1f77b4', '#ff7f0e']
    x_pos = []
    x_ticks = []
    x_labels = []
    current_pos = 0

    for chr_num in range(1, NUM_CHROMOSOMES + 1):
        chr_mask = chromosomes == chr_num
        chr_log_p = log_p_values[chr_mask]
        chr_positions_plot = current_pos + np.arange(np.sum(chr_mask))

        # Plot SNPs
        color = colors[chr_num % 2]
        ax.scatter(chr_positions_plot, chr_log_p, c=color, s=2, alpha=0.6)

        # Highlight causal SNPs
        causal_in_chr = [i for i in causal_indices if chromosomes[i] == chr_num]
        if causal_in_chr:
            causal_mask = np.zeros(len(chr_log_p), dtype=bool)
            for causal_idx in causal_in_chr:
                local_idx = np.where(np.arange(NUM_SNPS)[chr_mask] == causal_idx)[0]
                if len(local_idx) > 0:
                    causal_mask[local_idx[0]] = True

            ax.scatter(chr_positions_plot[causal_mask], chr_log_p[causal_mask],
                      c='red', s=20, marker='*', zorder=5, label='Causal SNP' if chr_num == 1 else '')

        # Track positions for chromosome labels
        x_ticks.append(current_pos + np.sum(chr_mask) / 2)
        x_labels.append(str(chr_num))
        current_pos += np.sum(chr_mask)

    # Add significance thresholds
    bonferroni = -np.log10(0.05 / NUM_SNPS)
    suggestive = -np.log10(1e-5)

    ax.axhline(y=bonferroni, color='red', linestyle='--', linewidth=1,
               label=f'Bonferroni (p={0.05/NUM_SNPS:.2e})')
    ax.axhline(y=suggestive, color='blue', linestyle='--', linewidth=1,
               label=f'Suggestive (p=1e-5)')

    ax.set_xlabel('Chromosome', fontsize=12, fontweight='bold')
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12, fontweight='bold')
    ax.set_title('Manhattan Plot - GWAS Results', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'manhattan_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Manhattan plot saved to output/manhattan_plot.png")


def create_qq_plot(p_values, output_dir):
    """
    Create Q-Q plot to check for population stratification.
    """
    print("\n[5/6] Creating Q-Q plot...")

    # Sort observed p-values
    observed = np.sort(p_values)

    # Expected p-values under null hypothesis
    expected = np.linspace(1/len(p_values), 1, len(p_values))

    # Convert to -log10 scale
    observed_log = -np.log10(observed + 1e-300)
    expected_log = -np.log10(expected)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot observed vs expected
    ax.scatter(expected_log, observed_log, s=5, alpha=0.5, color='#1f77b4')

    # Add diagonal line (y=x)
    max_val = max(expected_log.max(), observed_log.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Expected (null)')

    # Calculate lambda (genomic inflation factor)
    median_chi2 = np.median(stats.chi2.ppf(1 - p_values, df=1))
    expected_median_chi2 = stats.chi2.ppf(0.5, df=1)
    lambda_gc = median_chi2 / expected_median_chi2

    ax.set_xlabel('Expected -log₁₀(p-value)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed -log₁₀(p-value)', fontsize=12, fontweight='bold')
    ax.set_title(f'Q-Q Plot (λ = {lambda_gc:.3f})', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'qq_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Q-Q plot saved to output/qq_plot.png")


def save_results(snp_names, chromosomes, positions, p_values, odds_ratios,
                 causal_snps, output_dir):
    """
    Save top results to CSV files.
    """
    print("\n[6/6] Saving results...")

    # Create results dataframe
    results_df = pd.DataFrame({
        'SNP': snp_names,
        'Chromosome': chromosomes,
        'Position': positions,
        'P_value': p_values,
        'Odds_Ratio': odds_ratios,
        'Neg_log10_P': -np.log10(p_values + 1e-300),
        'Is_Causal': [snp in causal_snps for snp in snp_names]
    })

    # Sort by p-value
    results_df = results_df.sort_values('P_value')

    # Save top 100 SNPs
    top_snps = results_df.head(100)
    top_snps.to_csv(output_dir / 'top_100_snps.csv', index=False)
    print(f"  ✓ Top 100 SNPs saved to output/top_100_snps.csv")

    # Save all causal SNPs and their statistics
    causal_df = results_df[results_df['Is_Causal']]
    causal_df.to_csv(output_dir / 'causal_snps_results.csv', index=False)
    print(f"  ✓ Causal SNP results saved to output/causal_snps_results.csv")

    # Save full results (compressed)
    results_df.to_csv(output_dir / 'all_snps_results.csv.gz', index=False, compression='gzip')
    print(f"  ✓ Full results saved to output/all_snps_results.csv.gz")

    # Create summary statistics
    bonferroni_threshold = 0.05 / NUM_SNPS
    num_significant = (results_df['P_value'] < bonferroni_threshold).sum()
    num_causal_detected = causal_df[causal_df['P_value'] < bonferroni_threshold].shape[0]

    summary = {
        'Total_SNPs': NUM_SNPS,
        'Total_Samples': NUM_SAMPLES,
        'True_Causal_SNPs': NUM_CAUSAL_SNPS,
        'Significant_SNPs': num_significant,
        'Causal_SNPs_Detected': num_causal_detected,
        'Detection_Rate': num_causal_detected / NUM_CAUSAL_SNPS,
        'Bonferroni_Threshold': bonferroni_threshold,
        'Min_P_value': results_df['P_value'].min(),
        'Median_P_value': results_df['P_value'].median()
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"  ✓ Summary statistics saved to output/summary_statistics.csv")

    # Print summary to console
    print("\n" + "=" * 80)
    print("GWAS Analysis Summary")
    print("=" * 80)
    print(f"Total SNPs tested: {NUM_SNPS:,}")
    print(f"Significant SNPs (Bonferroni): {num_significant}")
    print(f"True causal SNPs: {NUM_CAUSAL_SNPS}")
    print(f"Causal SNPs detected: {num_causal_detected}/{NUM_CAUSAL_SNPS} ({num_causal_detected/NUM_CAUSAL_SNPS:.1%})")
    print(f"Most significant p-value: {results_df['P_value'].min():.2e}")
    print("\nTop 5 most significant SNPs:")
    print(top_snps[['SNP', 'Chromosome', 'Position', 'P_value', 'Odds_Ratio', 'Is_Causal']].head().to_string(index=False))
    print("=" * 80)


def main():
    """
    Main function to run the GWAS demo.
    """
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")

    # Step 1: Generate SNP data
    genotypes, snp_names, chromosomes, positions, maf = generate_snp_data()

    # Step 2: Simulate phenotype
    phenotype, causal_indices, causal_snps = simulate_phenotype(genotypes, snp_names)

    # Step 3: Perform association testing
    p_values, odds_ratios, lambda_gc = perform_association_testing(genotypes, phenotype, snp_names)

    # Step 4: Create Manhattan plot
    create_manhattan_plot(chromosomes, positions, p_values, causal_indices, output_dir)

    # Step 5: Create Q-Q plot
    create_qq_plot(p_values, output_dir)

    # Step 6: Save results
    save_results(snp_names, chromosomes, positions, p_values, odds_ratios,
                 causal_snps, output_dir)

    print("\n" + "=" * 80)
    print("GWAS Demo Complete!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - manhattan_plot.png          : Genome-wide association visualization")
    print("  - qq_plot.png                 : Quality control plot")
    print("  - top_100_snps.csv            : Top 100 most significant SNPs")
    print("  - causal_snps_results.csv     : Statistics for true causal SNPs")
    print("  - all_snps_results.csv.gz     : Complete results (compressed)")
    print("  - summary_statistics.csv      : Overall analysis summary")
    print()


if __name__ == '__main__':
    main()

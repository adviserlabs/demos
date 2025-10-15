#!/usr/bin/env nextflow

/*
 * Simple Protein-Ligand Binding Simulation Pipeline
 * Demonstrates a basic biomedical workflow using NextFlow
 */

nextflow.enable.dsl=2

// Parameters
params.ligands = "${projectDir}/data/ligands.csv"
params.protein = "${projectDir}/data/protein.pdb"
params.outdir = "${projectDir}/output"

// Display pipeline info
log.info """\
    PROTEIN-LIGAND BINDING SIMULATION
    ===================================
    ligands    : ${params.ligands}
    protein    : ${params.protein}
    outdir     : ${params.outdir}
    """
    .stripIndent()

/*
 * Process 1: Parse ligand molecules
 */
process PARSE_LIGANDS {
    publishDir "${params.outdir}/parsed_ligands", mode: 'copy'

    input:
    path ligands

    output:
    path "ligand_properties.json"

    script:
    """
    python3 ${projectDir}/bin/parse_ligands.py ${ligands} ligand_properties.json
    """
}

/*
 * Process 2: Calculate binding energies
 */
process CALCULATE_BINDING {
    publishDir "${params.outdir}/binding_results", mode: 'copy'

    input:
    path protein
    path ligand_props

    output:
    path "binding_energies.csv"
    path "binding_report.txt"

    script:
    """
    python3 ${projectDir}/bin/calculate_binding.py ${protein} ${ligand_props} .
    """
}

/*
 * Process 3: Generate visualization
 */
process VISUALIZE_RESULTS {
    publishDir "${params.outdir}/visualization", mode: 'copy'

    input:
    path binding_energies

    output:
    path "binding_plot.png"
    path "top_candidates.txt"

    script:
    """
    python3 ${projectDir}/bin/visualize.py ${binding_energies} .
    """
}

/*
 * Main workflow
 */
workflow {
    // Create channels from input files
    ligands_ch = Channel.fromPath(params.ligands)
    protein_ch = Channel.fromPath(params.protein)

    // Execute pipeline
    ligand_props = PARSE_LIGANDS(ligands_ch)
    binding_results = CALCULATE_BINDING(protein_ch, ligand_props)
    viz_results = VISUALIZE_RESULTS(binding_results[0])

    // Print completion message
    viz_results[1].view {
        "\n✓ Pipeline completed! Check ${params.outdir} for results.\n"
    }
}

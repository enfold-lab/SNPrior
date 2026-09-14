# SNPrior

A Two-step Feature Selection Framework for Computationally Efficient Machine Learning on Genome-wide SNP Data, applied to predicting the population of origin of 1000 Genomes Project samples.

## Overview

Genome-wide data contain tens of millions of SNPs, far more than the number of samples, so machine-learning-based feature selection cannot be applied to the full feature space under realistic hardware. SNPrior splits selection into two stages:

1. **Preliminary selection** reduces the feature space with inexpensive filters (e.g., tens of millions of SNPs to ~1 million).
2. **Secondary selection** applies more expensive methods to the reduced set to pick a few hundred to ~100,000 SNPs.

## Evaluation protocol

- **Family-grouped nested cross-validation.** Samples are split with `StratifiedGroupKFold`, stratified by population code and grouped by pedigree-derived `family_id`, so related individuals never appear in both training and test sets.
    - **Outer loop (5 folds)**
    - **Inner loop (4 folds)**

## Dataset
- **Genotypes:** phased high-coverage (30x, GRCh38) VCFs for 3,202 samples from the [International Genome Sample Resource](https://www.internationalgenome.org/data-portal/data-collection/30x-grch38).
- **Sample metadata:** the IGSR sample table (`igsr-1000 genomes 30x on grch38.tsv`) and the pedigree file (`1kGP.3202_samples.pedigree_info.txt`).
- **Processing:** only autosomal SNVs are kept, and genotypes are encoded as alternate-allele dosage (0, 1, 2).

## Installation

```bash
conda env create -f environment.yml
conda activate SNPrior
```

## Usage

**Run from the repository root:**

```bash
python src/main.py
```

## Notes

- **Memory.** The full genotype matrix (3,202 samples × ~43 million SNPs, `int8`) occupies about 138 GB in memory, and selection on the full matrix needs several times that.


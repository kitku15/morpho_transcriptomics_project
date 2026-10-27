# Predicting Cortical Neuron Morphology from Single-Cell Gene Expression

This repository contains a reproducible Python pipeline addressing the technical interview task: predicting motor neuron morphology from Patch-seq transcriptomics (Scala et al., 2021). 

## Task Requirements

1. **Identify genes whose expression changes correspond with morphology:** 
   - Extracted scalar features are reduced via PCA to independently model Structural Size/Reach (PC1) and Geometry (PC2).
   - XGBoost is utilized to capture complex, non-linear gene-morphology relationships. SHAP (SHapley Additive exPlanations) is applied to extract exact, directional feature importance for every gene, avoiding black-box outcomes.

2. **Represent the potential function of these genes:** 
   - Pre-ranked Gene Set Enrichment Analysis (GSEA) is run using continuous signed SHAP scores to map transcriptomic drivers to known biological processes (GO terms).
   - High-ranking genes are plotted directly against raw morphological features to visualize continuous biological gradients.

3. **Provide statistical support:**
   - Cross-validation uses `GroupKFold` grouped by Animal ID to prevent the model from memorizing batch effects.
   - A 1,000-iteration parallel permutation test explicitly validates the true Out-Of-Fold (OOF) $R^2$ significance against a random null distribution.
   - A sparse linear model (`ElasticNet`) is evaluated symmetrically against XGBoost to statistically justify the use of non-linear methods.

## How to Run

*Note: The transcriptomic count matrices in the `data/` directory are compressed to comply with GitHub file size limits. Please unzip `transcriptomics.zip` inside the `data/` folder before proceeding.*

```bash
# 1. Extract transcriptomic data
cd data
python -m zipfile -e transcriptomics.zip .
cd ..

# 2. Setup environment (Python 3.8+)
python -m venv .venv
source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Execute the pipeline
python main.py
```
*Outputs (including execution logs, metric tables, and statistical plots) are  saved to a timestamped folder inside `results/run_YYYYMMDD_HHMMSS/`.*



import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_and_merge_data(
    config: dict, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads and reconciles raw multimodal datasets.
    
    Enforces strict data contracts for intersecting cells across morphological 
    and transcriptomic modalities prior to downstream analysis.

    Args:
        config (dict): Configuration dictionary containing data paths.
        logger (logging.Logger): Logger instance for execution tracking.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            Filtered morphological features, morphological Z-profiles, 
            exon counts, intron counts, and metadata.
            
    Raises:
        AssertionError: If data modalities fail to align or contain invalid counts.
    """
    logger.info("Loading and reconciling raw datasets...")
    paths = config['data']
    
    morph_feat = pd.read_csv(paths['morph_feat_path'], index_col=0)
    morph_z = pd.read_csv(paths['morph_z_path'], index_col=0)
    expr_exon = pd.read_csv(paths['expr_exon_path'], index_col=0).T
    expr_intron = pd.read_csv(paths['expr_intron_path'], index_col=0).T
    meta = pd.read_csv(paths['meta_path'], sep='\t', index_col='Cell')

    meta = meta.dropna(subset=['RNA type'])

    common_cells = (
        morph_feat.index
        .intersection(morph_z.index)
        .intersection(expr_exon.index)
        .intersection(expr_intron.index)
        .intersection(meta.index)
    )
    
    assert len(common_cells) > 0, "Data Contract Error: Zero overlapping cells found across modalities!"
    assert (expr_exon.values >= 0).all(), "Data Contract Error: Negative counts detected in exon matrix!"
    assert (expr_intron.values >= 0).all(), "Data Contract Error: Negative counts detected in intron matrix!"
    assert 'RNA family' in meta.columns, "Data Contract Error: Required 'RNA family' column missing from metadata!"
    
    m_feat = morph_feat.loc[common_cells]
    m_z = morph_z.loc[common_cells]
    e_exon = expr_exon.loc[common_cells]
    e_intron = expr_intron.loc[common_cells]
    m_meta = meta.loc[common_cells]

    assert (m_feat.index == e_exon.index).all(), "Data Contract Error: Index misalignment detected post-intersection!"
    
    logger.info(f"-> Retained {len(common_cells)} cells across all modalities with verified alignment.")
    return m_feat, m_z, e_exon, e_intron, m_meta


def filter_lineage(
    morph_feat: pd.DataFrame, 
    morph_z: pd.DataFrame, 
    expr_exon: pd.DataFrame, 
    expr_intron: pd.DataFrame, 
    meta: pd.DataFrame, 
    lineage: str, 
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filters consolidated datasets for a specific neuronal lineage.
    
    Args:
        morph_feat (pd.DataFrame): Morphological feature matrix.
        morph_z (pd.DataFrame): Z-profile morphological matrix.
        expr_exon (pd.DataFrame): Exonic transcript counts.
        expr_intron (pd.DataFrame): Intronic transcript counts.
        meta (pd.DataFrame): Cell metadata.
        lineage (str): Target cell class ('exc' or 'inh').
        logger (logging.Logger): Logger instance.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            Lineage-specific filtered DataFrames.
    """
    mask = morph_feat['cell class'] == lineage
    assert mask.sum() > 0, f"Data Contract Error: No cells found for lineage '{lineage}'!"
    
    logger.info(f"-> Isolated {mask.sum()} {lineage.upper()} neurons.")
    return morph_feat[mask], morph_z[mask], expr_exon[mask], expr_intron[mask], meta[mask]


def build_independent_targets(
    morph_feat: pd.DataFrame, 
    lineage: str, 
    random_seed: int, 
    logger: logging.Logger
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Computes Principal Components on scalar morphology features.
    
    Implements feature selection matching Scala et al. (2021) to isolate
    PC1 (Size/Reach) and PC2 (Geometry/Branching Shape).
    
    Args:
        morph_feat (pd.DataFrame): Morphological feature matrix.
        lineage (str): Target cell class ('exc' or 'inh').
        random_seed (int): Seed for reproducible PCA.
        logger (logging.Logger): Logger instance.
        
    Returns:
        Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, np.ndarray]:
            - Dictionary of target arrays (PC1 and PC2).
            - PC1 loading DataFrame.
            - PC2 loading DataFrame.
            - Explained variance ratio array.
    """
    logger.info("Building morphology targets (Scala et al. method)...")
    feat_num = morph_feat.select_dtypes(include=[np.number]).fillna(0)

    if lineage == 'exc':
        feat_cols = [c for c in feat_num.columns if 'axon' not in c.lower()]
    else:
        feat_cols = [c for c in feat_num.columns if 'apical' not in c.lower()]
        
    feat_num = feat_num[feat_cols]
    
    cols_to_keep = []
    cv = feat_num.std() / feat_num.mean().replace(0, np.nan)
    
    for c in feat_num.columns:
        if feat_num[c].std() < 1e-6:
            continue
        if feat_num[c].mean() > 0 and cv[c] < 0.25:
            continue
        cols_to_keep.append(c)
        
    feat_num = feat_num[cols_to_keep]
    logger.info(f"-> Applied Scala feature filter: Retained {len(cols_to_keep)} morphological features.")

    feat_scaled = pd.DataFrame(StandardScaler().fit_transform(feat_num), index=feat_num.index, columns=feat_num.columns)

    n_pcs = min(10, feat_scaled.shape[1])
    pca_feat = PCA(n_components=n_pcs, random_state=random_seed)
    morph_pca = pca_feat.fit_transform(feat_scaled)
    
    var_feat_pc1 = pca_feat.explained_variance_ratio_[0] * 100
    var_feat_pc2 = pca_feat.explained_variance_ratio_[1] * 100
    
    logger.info(f"-> Scalar Morphometrics PC1 explains {var_feat_pc1:.2f}% of variance (Size/Reach).")
    logger.info(f"-> Scalar Morphometrics PC2 explains {var_feat_pc2:.2f}% of variance (Geometry/Branching Shape).")

    targets = {
        'Morphology_PC1_Size': morph_pca[:, 0],
        'Morphology_PC2_Geometry': morph_pca[:, 1]
    }
    
    loadings_pc1 = pd.DataFrame({'Feature': feat_scaled.columns, 'Loading': pca_feat.components_[0]})
    loadings_pc2 = pd.DataFrame({'Feature': feat_scaled.columns, 'Loading': pca_feat.components_[1]})
    
    return targets, loadings_pc1, loadings_pc2, pca_feat.explained_variance_ratio_


def preprocess_rna(
    expr_exon: pd.DataFrame, 
    expr_intron: pd.DataFrame, 
    n_hvgs: int, 
    config: dict, 
    logger: logging.Logger
) -> Tuple[pd.DataFrame, sc.AnnData]:
    """Applies Scala cmin filtering, sums Exon and Intron counts, and normalizes.
    
    Args:
        expr_exon (pd.DataFrame): Exonic transcript counts.
        expr_intron (pd.DataFrame): Intronic transcript counts.
        n_hvgs (int): Number of Highly Variable Genes to select.
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.
        
    Returns:
        Tuple[pd.DataFrame, sc.AnnData]: Processed HVG matrix and corresponding AnnData object.
    """
    logger.info(f"Preprocessing transcriptomics ({n_hvgs} HVGs)...")
    
    cmin = config['preprocessing']['cmin']
    min_cells = config['preprocessing']['min_cells']
    
    valid_genes_mask = (expr_exon >= cmin).sum(axis=0) >= min_cells
    clean_mask = ~expr_exon.columns.str.startswith(('Gm', 'ERCC', 'Rik'))
    final_mask = valid_genes_mask & clean_mask
    
    expr_exon_filt = expr_exon.loc[:, final_mask]
    expr_intron_filt = expr_intron.loc[:, final_mask]
    
    logger.info(f"-> Scala Gene Filter (cmin={cmin}, cells>={min_cells}): Retained {final_mask.sum()} robust genes.")
    
    expr_total = expr_exon_filt + expr_intron_filt
    
    adata = sc.AnnData(X=expr_total.values)
    adata.obs.index = expr_total.index
    adata.var.index = expr_total.columns

    target_sum = config['preprocessing']['target_sum']
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if n_hvgs < adata.shape[1]:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor='seurat')
        X = pd.DataFrame(
            adata[:, adata.var.highly_variable].X,
            index=adata.obs.index,
            columns=adata.var[adata.var.highly_variable].index
        )
    else:
        X = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)
        
    logger.info(f"-> Clean Feature Matrix (Exon+Intron): {X.shape[0]} cells by {X.shape[1]} HVGs.")
    return X, adata
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_qc_suite(
    morph_feat: pd.DataFrame, 
    morph_z: pd.DataFrame, 
    expr_df: pd.DataFrame, 
    meta_df: pd.DataFrame, 
    lin_name: str, 
    qc_dir: str, 
    random_seed: int = 42
) -> None:
    """Generates automated structural and transcriptomic UMAPs prior to modeling.
    
    Args:
        morph_feat (pd.DataFrame): Physical morphology metrics.
        morph_z (pd.DataFrame): Z-profile morphology metrics.
        expr_df (pd.DataFrame): Transcriptomic profiles.
        meta_df (pd.DataFrame): Cell metadata.
        lin_name (str): Lineage string identifier.
        qc_dir (str): Destination directory for resulting plots.
        random_seed (int): Seed for UMAP/PCA reproducibility.
    """
    os.makedirs(qc_dir, exist_ok=True)
    meta = meta_df.copy()
    meta.loc[:, 'RNA_Family'] = meta['RNA family'].fillna('Unknown')

    adata = sc.AnnData(X=expr_df.values)
    adata.obs = meta.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='seurat')
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack', random_state=random_seed)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=random_seed)
    sc.tl.umap(adata, random_state=random_seed)

    sc.pl.umap(adata, color='RNA_Family', show=False, title=f"[{lin_name.upper()}] Transcriptomic UMAP")
    plt.savefig(os.path.join(qc_dir, "01_transcriptomic_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    feat_num = morph_feat.select_dtypes(include=[np.number]).fillna(0)
    if 'exc' in lin_name.lower():
        feat_cols = [c for c in feat_num.columns if 'axon' not in c.lower()]
    else:
        feat_cols = [c for c in feat_num.columns if 'apical' not in c.lower()]
    feat_num = feat_num[feat_cols]
    
    cols_to_keep = [c for c in feat_num.columns if feat_num[c].std() > 1e-6]
    feat_scaled = StandardScaler().fit_transform(feat_num[cols_to_keep])
    pca_feat = PCA(n_components=2, random_state=random_seed).fit_transform(feat_scaled)
    
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=pca_feat[:, 0], y=pca_feat[:, 1], hue=meta['RNA_Family'], palette='tab10', alpha=0.8)
    plt.title(f"[{lin_name.upper()}] Morphology PCA (PC1 vs PC2)")
    plt.xlabel("Morphology PC1 (Size)")
    plt.ylabel("Morphology PC2 (Geometry)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "02_morphology_pca.png"), dpi=300)
    plt.close()

    z_scaled = StandardScaler().fit_transform(morph_z.select_dtypes(include=[np.number]).fillna(0))
    pca_z = PCA(n_components=2, random_state=random_seed).fit_transform(z_scaled)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=pca_z[:, 0], y=pca_z[:, 1], hue=meta['RNA_Family'], palette='tab10', alpha=0.8)
    plt.title(f"[{lin_name.upper()}] Z-Profile PCA (PC1 vs PC2)")
    plt.xlabel("Z-Profile PC1 (Soma Depth)")
    plt.ylabel("Z-Profile PC2 (Spatial Distribution)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "03_zprofile_pca.png"), dpi=300)
    plt.close()

    if 'exc' in lin_name.lower():
        target_feats = ['total length', 'height', 'width', 'branch points']
    else:
        target_feats = ['axon height', 'branch points', 'above soma', 'tortuosity']

    matched_cols = []
    for tf in target_feats:
        matches = [c for c in morph_feat.columns if tf in c.lower() and ('apical' in c.lower() if 'exc' in lin_name.lower() else 'axon' in c.lower())]
        if matches:
            matched_cols.append(matches[0])
            
    if len(matched_cols) >= 2:
        fig, axes = plt.subplots(1, len(matched_cols), figsize=(4 * len(matched_cols), 4.5), sharey=False)
        for idx, col in enumerate(matched_cols):
            sns.boxplot(data=morph_feat, x=meta['RNA_Family'], y=col, ax=axes[idx], palette='tab10', fliersize=2)
            axes[idx].set_title(col.replace('"', '').title(), fontsize=11, pad=10)
            axes[idx].set_xlabel("")
            axes[idx].tick_params(axis='x', rotation=45)
        plt.suptitle(f"[{lin_name.upper()}] Physical Morphology Baselines Across Families", y=1.03, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(qc_dir, "04_morphology_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if 'exc' in lin_name.lower():
        candidate_genes = ['Coro6', 'Tox', 'Calb1', 'Crym', 'Ptprm', 'Cux2']
    else:
        candidate_genes = ['Sst', 'Pvalb', 'Vip', 'Kit', 'Npy', 'Lamp5']

    genes_to_plot = [g for g in candidate_genes if g in adata.var_names]
    if genes_to_plot:
        sc.pl.umap(adata, color=genes_to_plot, ncols=3, show=False, cmap='magma')
        plt.suptitle(f"[{lin_name.upper()}] Expression Landscape of Top Biological Drivers", y=1.02, fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(qc_dir, "05_marker_gene_umaps.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_pca_loadings(loadings: pd.DataFrame, title: str, save_path: str) -> None:
    """Visualizes the top ten features defining a principal component."""
    plt.figure(figsize=(9, 4.5))
    top_loadings = loadings.reindex(loadings['Loading'].abs().sort_values(ascending=False).index).head(10)
    sns.barplot(data=top_loadings, x='Loading', y='Feature', hue='Feature', palette='coolwarm', legend=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_permutation_test(null_r2s: List[float], true_r2: float, perm_p_val: float, title: str, save_path: str) -> None:
    """Graphs the empirical p-value validation distribution."""
    plt.figure(figsize=(6, 4))
    sns.histplot(null_r2s, bins=15, kde=True, color='gray', stat='density')
    plt.axvline(true_r2, color='red', linestyle='--', label=f'True R² ({true_r2:.2f}, p={perm_p_val:.3f})')
    plt.xlabel("OOF R² Score")
    plt.title(f"{title}\n(Permutation p = {perm_p_val:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_cv_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str) -> None:
    """Plots cross-validated model predictions versus true targets."""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor=None)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted (Unseen Mice)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_shap_summary(shap_values: pd.DataFrame, X: pd.DataFrame, title: str, save_path: str) -> None:
    """Outputs the global feature importance dot plot."""
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values.values, X, max_display=15, show=False)
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_stratified_shap(
    shap_values: pd.DataFrame, 
    X: pd.DataFrame, 
    meta_df: pd.DataFrame, 
    lineage: str, 
    target_name: str, 
    out_dir: str
) -> None:
    """Renders localized SHAP summaries specific to individual RNA families.
    
    Args:
        shap_values (pd.DataFrame): Derived SHAP values.
        X (pd.DataFrame): Input transcriptomics matrix.
        meta_df (pd.DataFrame): Metadata mapping containing RNA family assignments.
        lineage (str): Lineage string identifier.
        target_name (str): Label of modeled parameter.
        out_dir (str): Base plot directory.
    """
    families = meta_df.loc[X.index, 'RNA family']

    for family in families.dropna().unique():
        family_mask = families == family
        cells = X.index[family_mask]
        
        if len(cells) < 10:
            continue
            
        plt.figure(figsize=(7, 5))
        shap.summary_plot(
            shap_values.loc[cells].values, 
            X.loc[cells], 
            max_display=10, 
            show=False
        )
        clean_name = str(family).replace('/', '_').replace(' ', '_')
        plt.title(f"[{lineage.upper()} | {family}] SHAP: {target_name}", pad=15)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"04_shap_{clean_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_shap_dependence(
    shap_values: pd.DataFrame, 
    X: pd.DataFrame, 
    meta_df: pd.DataFrame, 
    lineage: str, 
    target_gene: str, 
    interaction_gene: str, 
    out_dir: str
) -> None:
    """Creates subclass-stratified SHAP dependence plots revealing biological activation thresholds."""
    if target_gene not in X.columns:
        return

    plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        ind=target_gene,
        shap_values=shap_values.loc[X.index].values,
        features=X,
        interaction_index=interaction_gene if interaction_gene in X.columns else 'auto',
        show=False
    )
    plt.title(f"[{lineage.upper()} | GLOBAL] Activation Threshold: {target_gene}", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"04c_threshold_{target_gene}_GLOBAL.png"), dpi=300, bbox_inches='tight')
    plt.close()
        
    families = meta_df.loc[X.index, 'RNA family']

    for family in families.dropna().unique():
        family_mask = families == family
        cells = X.index[family_mask]
        
        if len(cells) < 10:
            continue
            
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            ind=target_gene,
            shap_values=shap_values.loc[cells].values,
            features=X.loc[cells],
            interaction_index=interaction_gene if interaction_gene in X.columns else 'auto',
            show=False
        )
        clean_name = str(family).replace('/', '_').replace(' ', '_')
        plt.title(f"[{lineage.upper()} | {family}] Activation Threshold: {target_gene}", pad=15)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"04c_threshold_{target_gene}_{clean_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_linear_coefficients(coef_df: pd.DataFrame, title: str, save_path: str) -> None:
    """Plots the ElasticNet standardized regression coefficients."""
    top_coefs = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index).head(15)
    top_coefs = top_coefs.sort_values(by='Coefficient')
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_coefs, x='Coefficient', y='Gene', hue='Gene', palette='coolwarm', legend=False)
    plt.title(title, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pca_scree(exp_var_ratio: np.ndarray, title: str, save_path: str) -> None:
    """Visualizes cumulative morphology variance retention."""
    plt.figure(figsize=(7, 4.5))
    pcs = range(1, len(exp_var_ratio) + 1)
    var_pct = np.array(exp_var_ratio) * 100
    cum_pct = np.cumsum(var_pct)
    
    plt.bar(pcs, var_pct, color='#4C72B0', alpha=0.8, label='Individual Variance')
    plt.plot(pcs, cum_pct, color='#C44E52', marker='o', linewidth=2, label='Cumulative Variance')
    
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained (%)")
    plt.title(title, pad=15)
    plt.xticks(pcs)
    plt.ylim(0, 105)
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_cv_stability(fold_r2_xgb: List[float], fold_r2_linear: List[float], title: str, save_path: str) -> None:
    """Visualizes model generalizability through a cross-fold dumbbell plot."""
    plt.figure(figsize=(6, 5))
    folds = len(fold_r2_xgb)
    
    for i in range(folds):
        plt.plot([0, 1], [fold_r2_linear[i], fold_r2_xgb[i]], color='gray', linestyle='--', alpha=0.7)
        plt.scatter(0, fold_r2_linear[i], color='#4C72B0', s=80, zorder=3)
        plt.scatter(1, fold_r2_xgb[i], color='#C44E52', s=80, zorder=3)
        
    plt.xticks([0, 1], ['ElasticNet\n(Sparse Linear)', 'XGBoost\n(Non-Linear)'])
    plt.ylabel("Test Fold R² (Unseen Mice)")
    plt.title(title, pad=15)
    plt.axhline(0, color='black', linestyle=':', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Calculates False Discovery Rate (FDR) q-values using Benjamini-Hochberg."""
    p_arr = np.array(p_values)
    n = len(p_arr)
    if n == 0:
        return p_arr
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]
    q_vals = np.zeros(n)
    q_vals[-1] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        q_vals[i] = min(sorted_p[i] * n / (i + 1), q_vals[i + 1])
    out = np.zeros(n)
    out[sorted_idx] = q_vals
    return np.clip(out, 0, 1)

def _clean_feature_name(name: str) -> str:
    """Converts code-friendly variables into publication-grade labels."""
    clean = str(name).replace('"', '').replace("'", '').replace('_', ' ')
    clean = clean.replace('log1p', '').replace('pct', '%').strip()
    return clean.title()

def plot_feature_gene_heatmap(
    X: pd.DataFrame, 
    morph_feat: pd.DataFrame, 
    top_genes: List[str], 
    top_features: List[str], 
    title: str, 
    save_path: str
) -> None:
    """Evaluates top model features directly against physical morphology distributions.
    
    Generates a publication-grade Spearman correlation matrix with inline statistical
    reporting, strictly governed by Benjamini-Hochberg FDR adjustments.
    
    Args:
        X (pd.DataFrame): Input transcriptomics matrix.
        morph_feat (pd.DataFrame): Physical morphology feature table.
        top_genes (List[str]): Extracted target genes.
        top_features (List[str]): Original morphological PC drivers.
        title (str): Chart title.
        save_path (str): Destination disk path.
    """
    valid_feats = []
    for f in top_features:
        if f in morph_feat.columns and morph_feat.loc[X.index, f].std() > 1e-6:
            valid_feats.append(f)
    valid_feats = valid_feats[:10]

    valid_genes = [g for g in top_genes if g in X.columns][:10]
    
    if not valid_genes or not valid_feats:
        return

    n_feats, n_genes = len(valid_feats), len(valid_genes)
    corr_mat = np.zeros((n_feats, n_genes))
    raw_p_mat = np.zeros((n_feats, n_genes))
    
    for i, f in enumerate(valid_feats):
        for j, g in enumerate(valid_genes):
            r, p = spearmanr(morph_feat.loc[X.index, f], X[g])
            corr_mat[i, j] = r
            raw_p_mat[i, j] = p

    flat_p = raw_p_mat.flatten()
    flat_q = _benjamini_hochberg(flat_p)
    q_mat = flat_q.reshape((n_feats, n_genes))

    annot_mat = np.empty((n_feats, n_genes), dtype=object)
    for i in range(n_feats):
        for j in range(n_genes):
            r_val = corr_mat[i, j]
            q_val = q_mat[i, j]
            stars = "***" if q_val < 0.001 else "**" if q_val < 0.01 else "*" if q_val < 0.05 else ""
            annot_mat[i, j] = f"{r_val:+.2f}"

    clean_row_labels = [_clean_feature_name(f) for f in valid_feats]
    df_corr = pd.DataFrame(corr_mat, index=clean_row_labels, columns=valid_genes)

    plt.figure(figsize=(10, 7.5))
    sns.heatmap(
        df_corr, 
        annot=annot_mat, 
        fmt='', 
        cmap='vlag', 
        center=0, 
        vmin=-0.6, 
        vmax=0.6, 
        annot_kws={'fontsize': 9, 'weight': 'bold'},
        linewidths=0.5, 
        linecolor='#ECEFF1',
        cbar_kws={'label': 'Spearman Correlation (r)', 'shrink': 0.8}
    )
    plt.title(title, pad=15, fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _render_gsea_bar(df_subset: pd.DataFrame, color: str, title: str, save_path: str) -> None:
    """Renders standalone GSEA directional enrichment components."""
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_subset, x='NES', y='Term', hue='Term', palette=[color] * len(df_subset), legend=False)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(title, pad=15)
    plt.xlabel("Normalized Enrichment Score (NES)")
    plt.ylabel("Biological Pathway")
    plt.subplots_adjust(left=0.45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gsea_suite(gsea_df: pd.DataFrame, title_prefix: str, out_dir: str) -> None:
    """Exports structural GSEA pathway metrics spanning all regulation directions.
    
    Args:
        gsea_df (pd.DataFrame): Dataframe of calculated GSEA terms and Normalized Enrichment Scores.
        title_prefix (str): Prefix mapping the tested lineage parameter.
        out_dir (str): Base plot directory.
    """
    top_combined = gsea_df.head(10).sort_values(by='NES', ascending=True)
    plt.figure(figsize=(10, 5))
    colors = ['#4C72B0' if x < 0 else '#C44E52' for x in top_combined['NES']]
    sns.barplot(data=top_combined, x='NES', y='Term', hue='Term', palette=colors, legend=False)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f"{title_prefix} (Combined Top 10)", pad=15)
    plt.xlabel("Normalized Enrichment Score (NES)")
    plt.ylabel("Biological Pathway")
    plt.subplots_adjust(left=0.45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_gsea_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()

    pos_df = gsea_df[gsea_df['NES'] > 0].sort_values(by='NES', ascending=True).tail(10)
    if not pos_df.empty:
        _render_gsea_bar(
            pos_df, 
            '#C44E52', 
            f"{title_prefix} (Positive Drivers / Upregulated)", 
            os.path.join(out_dir, "05_gsea_positive.png")
        )

    neg_df = gsea_df[gsea_df['NES'] < 0].sort_values(by='NES', ascending=False).tail(10).sort_values(by='NES', ascending=True)
    if not neg_df.empty:
        _render_gsea_bar(
            neg_df, 
            '#4C72B0', 
            f"{title_prefix} (Negative Drivers / Downregulated)", 
            os.path.join(out_dir, "05_gsea_negative.png")
        )

def plot_intra_class_gradient(
    expr_norm: pd.DataFrame, 
    morph_pc1: pd.Series, 
    meta_df: pd.DataFrame, 
    lineage: str, 
    target_gene: str, 
    conf_thresh: float, 
    title: str, 
    save_path: str
) -> None:
    """Highlights intra-class gradients linking gene expression to structure variation.
    
    Args:
        expr_norm (pd.DataFrame): Normalized transcriptomic arrays.
        morph_pc1 (pd.Series): Morphological dimension measurements.
        meta_df (pd.DataFrame): Descriptive metadata annotations.
        lineage (str): Investigated cell line.
        target_gene (str): Highly variable target gene.
        conf_thresh (float): Minimum validation confidence required to retain a cell.
        title (str): Applied plot title.
        save_path (str): Destination serialization file path.
    """
    meta_df['RNA type confidence'] = pd.to_numeric(meta_df['RNA type confidence'], errors='coerce')
    high_conf_mask = meta_df['RNA type confidence'] >= conf_thresh
    
    valid_cells = meta_df[high_conf_mask].index.intersection(expr_norm.index)
    if len(valid_cells) == 0:
        return

    families = meta_df.loc[valid_cells, 'RNA family']

    plot_df = pd.DataFrame({
        'Gene Expression (Log CPM)': expr_norm.loc[valid_cells, target_gene],
        'Morphology PC1 (Structural Complexity)': morph_pc1[valid_cells],
        'RNA Family': families
    }).dropna()

    counts = plot_df['RNA Family'].value_counts()
    valid_sub = counts[counts >= 5].index
    plot_df = plot_df[plot_df['RNA Family'].isin(valid_sub)]

    stat_labels = {}
    for sub in valid_sub:
        sub_data = plot_df[plot_df['RNA Family'] == sub]
        r, p = spearmanr(sub_data['Gene Expression (Log CPM)'], sub_data['Morphology PC1 (Structural Complexity)'])
        stat_labels[sub] = f"{sub} (r={r:.2f}, p={p:.3f})"
        
    plot_df['Subclass (Stats)'] = plot_df['RNA Family'].map(stat_labels)

    g = sns.lmplot(
        data=plot_df, x='Gene Expression (Log CPM)', y='Morphology PC1 (Structural Complexity)', 
        hue='Subclass (Stats)', ci=None, height=6, aspect=1.3,
        scatter_kws={'alpha': 0.7, 's': 50}, line_kws={'linewidth': 3}
    )
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
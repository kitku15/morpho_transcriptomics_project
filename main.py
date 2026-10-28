import os
import pandas as pd
from typing import Dict, Any

from src.utils import load_config, setup_experiment_dir, get_logger
from src.data_prep import load_and_merge_data, filter_lineage, build_independent_targets, preprocess_rna
from src.modeling import run_evaluation, run_preranked_gsea
from src import visualization as viz

def main() -> None:
    """Executes the complete morpho-transcriptomic analytical pipeline."""
    config = load_config()
    dirs = setup_experiment_dir(config['general']['output_base_dir'])
    logger = get_logger("pipeline", dirs['base'])
    
    logger.info("=" * 80)
    logger.info("STARTING CORTICAL NEURON MORPHO-TRANSCRIPTOMIC ANALYSIS")
    logger.info("=" * 80)

    morph_feat_all, morph_z_all, expr_exon_all, expr_intron_all, meta_df_all = load_and_merge_data(config, logger)

    n_hvgs = config['experiments']['n_hvgs']
    lineage_map = {'exc': 'excitatory', 'inh': 'inhibitory'}

    for lineage in config['experiments']['lineages']:
        lin_name = lineage_map[lineage]
        logger.info(f"\n{'='*80}\nPROCESSING LINEAGE: {lin_name.upper()}\n{'='*80}")
        
        lin_dir = os.path.join(dirs['base'], lin_name)
        m_feat, m_z, e_exon, e_intron, meta = filter_lineage(
            morph_feat_all, morph_z_all, expr_exon_all, expr_intron_all, meta_df_all, lineage, logger
        )
        
        logger.info(f"[{lin_name.upper()}] Generating Exploratory QC Visualizations...")
        qc_dir = os.path.join(lin_dir, "00_qc")
        viz.plot_qc_suite(m_feat, m_z, e_exon, meta, lin_name, qc_dir, config['general']['random_seed'])

        targets, load_pc1, load_pc2, exp_var_ratio = build_independent_targets(
            m_feat, lineage, config['general']['random_seed'], logger
        )

        viz.plot_pca_scree(
            exp_var_ratio, 
            f"[{lin_name.upper()}] Morphology PCA Scree Plot (Elbow)", 
            os.path.join(lin_dir, "00_pca_scree.png")
        )

        X, _ = preprocess_rna(e_exon, e_intron, n_hvgs, config, logger)
        groups = meta.loc[X.index, 'Mouse'].values
        
        for name, y_target in targets.items():
            sub_folder = "pc1_size" if "PC1" in name else "pc2_geometry"
            target_dir = os.path.join(lin_dir, sub_folder)
            
            plots_dir = os.path.join(target_dir, "plots")
            tables_dir = os.path.join(target_dir, "tables")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            
            logger.info(f"   -> Processing: {lin_name.upper()} / {sub_folder.upper()} ({name})")
            
            current_loadings = load_pc1 if "PC1" in name else load_pc2
            viz.plot_pca_loadings(
                current_loadings, 
                f"[{lin_name.upper()}] Top Features Driving {name}", 
                os.path.join(plots_dir, "01_loadings.png")
            )

            res = run_evaluation(X, y_target, groups, name, config, logger)
            
            res['metrics_df'].to_csv(os.path.join(tables_dir, "00_model_metrics.csv"), index=False)
            res['gene_summary_df'].to_csv(os.path.join(tables_dir, "04_ranked_genes.csv"), index=False)
            
            viz.plot_permutation_test(
                res['null_r2s'], res['r2_xgb'], res['perm_p_val'], 
                f"[{lin_name.upper()}] Permutation Test: {sub_folder.upper()}", 
                os.path.join(plots_dir, "02_permutation.png")
            )
            viz.plot_cv_stability(
                res['fold_r2_xgb'], res['fold_r2_linear'], 
                f"[{lin_name.upper()}] Cross-Validation Stability Across Animal Folds", 
                os.path.join(plots_dir, "02b_cv_stability.png")
            )
            viz.plot_cv_scatter(
                y_target, res['preds'], 
                f"[{lin_name.upper()}] XGBoost (R² = {res['r2_xgb']:.2f})", 
                os.path.join(plots_dir, "03_cv_scatter.png")
            )
            viz.plot_shap_summary(
                res['shap_vals'], X, 
                f"[{lin_name.upper()}] Global SHAP: {sub_folder.upper()}", 
                os.path.join(plots_dir, "04_shap_global.png")
            )
            viz.plot_linear_coefficients(
                res['linear_coefs'], 
                f"[{lin_name.upper()}] ElasticNet Coefficients: {sub_folder.upper()}", 
                os.path.join(plots_dir, "04b_linear_coefs.png")
            )
            
            viz.plot_stratified_shap(res['shap_vals'], X, meta, lineage, name, plots_dir)
            
            if len(res['top_genes']) >= 2:
                gene1, gene2 = res['top_genes'][0], res['top_genes'][1]
                viz.plot_shap_dependence(
                    shap_values=res['shap_vals'], 
                    X=X, 
                    meta_df=meta, 
                    lineage=lineage, 
                    target_gene=gene1, 
                    interaction_gene=gene2, 
                    out_dir=plots_dir
                )

            top_feats = current_loadings.head(10)['Feature'].tolist()
            viz.plot_feature_gene_heatmap(
                X=X, 
                morph_feat=m_feat, 
                top_genes=res['top_genes'], 
                top_features=top_feats, 
                title=f"[{lin_name.upper()}] Feature-to-Gene Correlation Heatmap: {sub_folder.upper()}", 
                save_path=os.path.join(plots_dir, "06_feature_gene_heatmap.png")
            )
            
            if len(res['top_genes']) > 0:
                viz.plot_intra_class_gradient(
                    expr_norm=X,
                    morph_pc1=pd.Series(y_target, index=X.index),
                    meta_df=meta,
                    lineage=lineage,
                    target_gene=res['top_genes'][0],
                    conf_thresh=config['preprocessing']['confidence_threshold'],
                    title=f"[{lin_name.upper()}] Morpho-Transcriptomic Gradient: {res['top_genes'][0]}",
                    save_path=os.path.join(plots_dir, "07_intra_class_gradient.png")
                )

            gsea_res = run_preranked_gsea(X, res['shap_vals'], y_target, logger)
            if gsea_res is not None:
                gsea_res.to_csv(os.path.join(tables_dir, "05_gsea_pathways.csv"), index=True)
                viz.plot_gsea_suite(gsea_res, f"[{lin_name.upper()}] GSEA: {sub_folder.upper()}", plots_dir)

    logger.info("=" * 80)
    logger.info(f"ANALYSIS COMPLETE. All results saved to {dirs['base']}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
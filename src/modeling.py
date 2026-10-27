import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import gseapy as gp
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import ttest_rel

def _run_single_permutation(seed_idx: int, X_arr: np.ndarray, y_target: np.ndarray, splits: List) -> float:
    """Worker function for parallel empirical permutation testing.
    
    Args:
        seed_idx (int): Seed index for thread-safe RNG generation.
        X_arr (np.ndarray): Feature array.
        y_target (np.ndarray): True target array to be shuffled.
        splits (List): Pre-computed GroupKFold splits.
        
    Returns:
        float: Permuted out-of-fold R² score.
    """
    rng = np.random.default_rng(seed=seed_idx)
    y_shuf = rng.permutation(y_target)
    oof_null = np.zeros(len(X_arr))
    
    for tr, va in splits:
        m_null = xgb.XGBRegressor(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1, 
            n_jobs=1,
            random_state=seed_idx
        )
        m_null.fit(X_arr[tr], y_shuf[tr])
        oof_null[va] = m_null.predict(X_arr[va])
        
    return float(r2_score(y_shuf, oof_null))

def run_evaluation(
    X: pd.DataFrame, 
    y_target: np.ndarray, 
    groups: np.ndarray, 
    target_name: str, 
    config: dict, 
    logger: logging.Logger
) -> Dict[str, Any]:
    """Executes leakage-safe GroupKFold evaluation and model interpretation.
    
    Performs cross-validated XGBoost and ElasticNet baseline modeling, 
    parallel empirical permutation testing, and exact SHAP extraction.
    
    Args:
        X (pd.DataFrame): Gene expression matrix.
        y_target (np.ndarray): Target morphological vector.
        groups (np.ndarray): Animal IDs for stratified fold assignment.
        target_name (str): Label for the current modeled target.
        config (dict): Pipeline configuration parameters.
        logger (logging.Logger): Logger instance.
        
    Returns:
        Dict[str, Any]: Consolidated metrics, predictions, null distributions,
            and derived statistical dataframes.
    """
    n_splits = config['modeling']['n_cv_splits']
    xgb_params = config['modeling']['xgb_params']
    enet_params = config['modeling']['enet_params']
    
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds_xgb = np.zeros(len(X))
    oof_preds_linear = np.zeros(len(X))
    shap_list = []
    fold_linear_coefs = []
    fold_r2_xgb = []
    fold_r2_linear = []
    
    for train_idx, val_idx in gkf.split(X, groups=groups):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y_target[train_idx], y_target[val_idx]
        
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_train, y_train)
        preds_xgb = model_xgb.predict(X_val)
        oof_preds_xgb[val_idx] = preds_xgb
        fold_r2_xgb.append(r2_score(y_val, preds_xgb))
        
        linear_base = ElasticNet(**enet_params)
        linear_base.fit(X_train, y_train)
        preds_linear = linear_base.predict(X_val)
        oof_preds_linear[val_idx] = preds_linear
        fold_r2_linear.append(r2_score(y_val, preds_linear))
        fold_linear_coefs.append(linear_base.coef_)
        
        explainer = shap.TreeExplainer(model_xgb)
        shap_list.append(pd.DataFrame(explainer.shap_values(X_val), columns=X_val.columns, index=X_val.index))
        
    oof_shap = pd.concat(shap_list).sort_index()
    r2_xgb = r2_score(y_target, oof_preds_xgb)
    r2_enet = r2_score(y_target, oof_preds_linear)
    
    stat, p_model_diff = ttest_rel(fold_r2_xgb, fold_r2_linear)
    
    avg_linear_coefs = np.mean(fold_linear_coefs, axis=0)
    linear_coef_df = pd.DataFrame({'Gene': X.columns, 'Linear_Coefficient': avg_linear_coefs})
    
    logger.info(f"      -> OOF R² (XGBoost): {r2_xgb:.3f}")
    logger.info(f"      -> OOF R² (Sparse Linear Baseline): {r2_enet:.3f}")
    logger.info(f"      -> Paired Fold Test (XGBoost vs Linear): t = {stat:.2f}, p-value = {p_model_diff:.4f}")

    n_perm = config['modeling']['n_permutations']
    logger.info(f"      -> Running parallel permutation test (N={n_perm} across all CPU cores)...")
    
    X_arr = X.values
    splits = list(gkf.split(X_arr, groups=groups))
    
    null_r2s = Parallel(n_jobs=-1)(
        delayed(_run_single_permutation)(i, X_arr, y_target, splits) 
        for i in range(n_perm)
    )

    perm_p_val = (np.sum(np.array(null_r2s) >= r2_xgb) + 1) / (len(null_r2s) + 1)
    logger.info(f"      -> Empirical Permutation p-value: p = {perm_p_val:.4f}")

    mean_abs = np.abs(oof_shap.values).mean(axis=0)
    z_scores = stats.zscore(mean_abs)
    ranking = pd.DataFrame({'Gene': X.columns, 'Mean_Abs_SHAP': mean_abs, 'SHAP_ZScore': z_scores}).sort_values(by='Mean_Abs_SHAP', ascending=False)
    
    gene_summary_df = ranking.merge(linear_coef_df, on='Gene', how='left')
    
    z_thresh = config['interpretation']['shap_z_thresh']
    clean_genes = gene_summary_df[gene_summary_df['SHAP_ZScore'] > z_thresh]['Gene'].tolist()
    if len(clean_genes) < 10:
        clean_genes = gene_summary_df.head(20)['Gene'].tolist()
        logger.info(f"      -> Z > {z_thresh} too strict, falling back to top 20 genes.")
    else:
        logger.info(f"      -> Selected {len(clean_genes)} significant genes (Z > {z_thresh}).")

    metrics_df = pd.DataFrame([{
        'Target': target_name,
        'N_Cells': len(X),
        'OOF_R2_XGBoost': r2_xgb,
        'OOF_R2_ElasticNet': r2_enet,
        'Paired_T_Stat': stat,
        'Paired_P_Value': p_model_diff,
        'Permutation_P_Value': perm_p_val
    }])

    return {
        'r2_xgb': r2_xgb,
        'preds': oof_preds_xgb,
        'shap_vals': oof_shap,
        'null_r2s': null_r2s,
        'top_genes': clean_genes,
        'linear_coefs': linear_coef_df.rename(columns={'Linear_Coefficient': 'Coefficient'}),
        'perm_p_val': perm_p_val,
        'metrics_df': metrics_df,
        'gene_summary_df': gene_summary_df,
        'fold_r2_xgb': fold_r2_xgb,
        'fold_r2_linear': fold_r2_linear
    }

def run_preranked_gsea(
    X: pd.DataFrame, 
    shap_values: pd.DataFrame, 
    y_target: np.ndarray, 
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """Runs Pre-ranked Gene Set Enrichment Analysis (GSEA).
    
    Derives functional biological pathways by ranking tested genes using signed 
    SHAP importance scores, removing background universe bias.
    
    Args:
        X (pd.DataFrame): Gene expression matrix.
        shap_values (pd.DataFrame): Computed SHAP values mapping feature contributions.
        y_target (np.ndarray): Ground-truth target array.
        logger (logging.Logger): Logger instance.
        
    Returns:
        Optional[pd.DataFrame]: Summary DataFrame of enriched biological pathways.
    """
    logger.info("      -> Running Pre-ranked GSEA across all tested genes...")
    try:
        corrs = [stats.spearmanr(X[c], y_target)[0] for c in X.columns]
        corrs = np.nan_to_num(corrs)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        signed_metric = mean_abs_shap * np.sign(corrs)
        
        rnk = pd.Series(signed_metric, index=X.columns).dropna().sort_values(ascending=False)
        rnk = rnk[~rnk.index.duplicated(keep='first')]

        prerank_res = gp.prerank(
            rnk=rnk,
            gene_sets='GO_Biological_Process_2021',
            organism='mouse',
            permutation_num=500,
            min_size=5,
            max_size=500,
            seed=42,
            verbose=False
        )
        
        res_df = prerank_res.res2d
        if res_df is None or res_df.empty:
            logger.info("      -> No GSEA pathways enriched.")
            return None
            
        res_df = res_df.reindex(res_df['NES'].abs().sort_values(ascending=False).index).copy()
        logger.info(f"      -> Successfully evaluated GSEA across {len(res_df)} biological pathways.")
        return res_df
        
    except Exception as e:
        logger.error(f"      -> Error running GSEA: {e}")
        return None
import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from tabfm import TabFMRegressor
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0

def clean_col(c):
    return re.sub(r'[\[\]<>]', '', c)

OUT_DIR = 'tabfm_ablation_results'
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(OUT_DIR, 'summary.csv')
PRED_CSV = os.path.join(OUT_DIR, 'predictions.csv')
FOLD_CSV = os.path.join(OUT_DIR, 'fold_r2.csv')

df = pd.read_csv('final_df.csv')

race_cols = [c for c in df.columns if c.startswith('race_')]
df['race'] = df[race_cols].idxmax(axis=1).str.replace('race_', '', regex=False)

sex_cols = [c for c in df.columns if c.startswith('sex_')]
df['sex'] = df[sex_cols].idxmax(axis=1).str.replace('sex_', '', regex=False)

target_col = 'import_insulin, Insulin [Units/volume] in Serum o'

exclude = {
    'participant_id', 'study_group', 'split',
    target_col,
    'import_glucose, Glucose [Mass/volume] in Serum or',
    'tir',
    'tar',
    'tbr'
}

feature_cols = [
    c for c in df.columns
    if c not in exclude
    and not c.startswith('race_')
    and not c.startswith('sex_')
]
print('Features:', feature_cols)

tabfm_model = tabfm_v1_0_0.load(model_type='regression')

N_FOLDS = 5
RANDOM_STATE = 42
N_BOOT = 1000
divider = '=' * 50


def bootstrap_r2_se(y_true, y_pred, n_boot=N_BOOT, random_state=RANDOM_STATE):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    boot_r2 = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        boot_r2[i] = r2_score(yt, yp) if np.var(yt) > 0 else np.nan

    return np.nanstd(boot_r2, ddof=1)


results = {}
prediction_rows = []
fold_r2_rows = []

for group in df['study_group'].dropna().unique():
    print('')
    print(divider)
    print('Study Group:', group)
    print(divider)

    group_df = df[df['study_group'] == group].copy()
    group_df = group_df.replace([np.inf, -np.inf], np.nan)
    group_df = group_df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    if len(group_df) < 20:
        print('Skipping - only', len(group_df), 'rows after dropping NAs (too few to model)')
        continue

    X = group_df[feature_cols].copy()
    X.columns = [clean_col(c) for c in X.columns]
    X['sex'] = X['sex'].astype(str)
    X['race'] = X['race'].astype(str)
    y = group_df[target_col].values
    participant_ids = group_df['participant_id'].values

    train_mask = (group_df['split'] == 'train').values
    test_mask = (group_df['split'] == 'test').values

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y[train_mask]
    y_test = y[test_mask]
    test_ids = participant_ids[test_mask]

    # ---------- 1. Forward train -> test (split-based) ----------
    reg = TabFMRegressor(model=tabfm_model)
    reg.fit(X_train, y_train)
    test_preds = reg.predict(X_test)

    forward_r2 = r2_score(y_test, test_preds)
    forward_r2_se = bootstrap_r2_se(y_test, test_preds)

    print('')
    print('Forward train->test (split-based)')
    print('  R-squared:', forward_r2)
    print('  SE (bootstrap):', forward_r2_se)

    prediction_rows.append(pd.DataFrame({
        'participant_id': test_ids,
        'study_group': group,
        'eval': 'forward_train_test',
        'actual_insulin': y_test,
        'pred_insulin_tabfm': test_preds,
    }))

    # ---------- Fold setup on TRAIN ONLY (test set never touched below) ----------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = [test_idx for _, test_idx in kf.split(X_train)]

    # ---------- 2. Standard 5-fold CV (train on other 4 folds -> predict held-out fold), pooled ----------
    print('')
    print('Standard 5-fold CV (train on other 4 folds -> predict held-out fold)')
    standard_true, standard_pred = [], []
    standard_fold_r2 = []
    for fold_i in range(N_FOLDS):
        te_idx = fold_indices[fold_i]
        tr_idx = np.concatenate([fold_indices[j] for j in range(N_FOLDS) if j != fold_i])

        Xtr, Xte = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        ytr, yte = y_train[tr_idx], y_train[te_idx]

        reg = TabFMRegressor(model=tabfm_model)
        reg.fit(Xtr, ytr)
        preds = reg.predict(Xte)

        fold_r2 = r2_score(yte, preds)
        standard_fold_r2.append(fold_r2)
        standard_true.append(yte)
        standard_pred.append(preds)

        print(f'  Fold {fold_i}: R2={fold_r2:.4f}  (n_test={len(te_idx)})')

        fold_r2_rows.append({
            'study_group': group, 'cv_variant': 'standard', 'fold': fold_i, 'r2': fold_r2
        })

    standard_true = np.concatenate(standard_true)
    standard_pred = np.concatenate(standard_pred)
    standard_pooled_r2 = r2_score(standard_true, standard_pred)
    standard_pooled_r2_se = bootstrap_r2_se(standard_true, standard_pred)
    print('  Pooled R2 (all held-out folds combined):', standard_pooled_r2)
    print('  Pooled SE (bootstrap):', standard_pooled_r2_se)

    # ---------- 3. Reverse 5-fold CV (train on 1 fold -> predict union of other 4), pooled ----------
    print('')
    print('Reverse 5-fold CV (train on 1 fold -> predict union of other 4 folds)')
    reverse_true, reverse_pred = [], []
    reverse_fold_r2 = []
    for fold_i in range(N_FOLDS):
        ctx_idx = fold_indices[fold_i]
        query_idx = np.concatenate([fold_indices[j] for j in range(N_FOLDS) if j != fold_i])

        Xctx, Xq = X_train.iloc[ctx_idx], X_train.iloc[query_idx]
        yctx, yq = y_train[ctx_idx], y_train[query_idx]

        reg = TabFMRegressor(model=tabfm_model)
        reg.fit(Xctx, yctx)
        preds = reg.predict(Xq)

        fold_r2 = r2_score(yq, preds)
        reverse_fold_r2.append(fold_r2)
        reverse_true.append(yq)
        reverse_pred.append(preds)

        print(f'  Fold {fold_i} as context: R2={fold_r2:.4f}  (n_query={len(query_idx)})')

        fold_r2_rows.append({
            'study_group': group, 'cv_variant': 'reverse', 'fold': fold_i, 'r2': fold_r2
        })

    reverse_true = np.concatenate(reverse_true)
    reverse_pred = np.concatenate(reverse_pred)
    reverse_pooled_r2 = r2_score(reverse_true, reverse_pred)
    reverse_pooled_r2_se = bootstrap_r2_se(reverse_true, reverse_pred)
    print('  Pooled R2 (all query-fold predictions combined):', reverse_pooled_r2)
    print('  Pooled SE (bootstrap):', reverse_pooled_r2_se)

    results[group] = {
        'n': len(group_df),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'forward_r2': forward_r2,
        'forward_r2_se': forward_r2_se,
        'standard_cv_pooled_r2': standard_pooled_r2,
        'standard_cv_pooled_r2_se': standard_pooled_r2_se,
        'standard_cv_fold_r2': standard_fold_r2,
        'reverse_cv_pooled_r2': reverse_pooled_r2,
        'reverse_cv_pooled_r2_se': reverse_pooled_r2_se,
        'reverse_cv_fold_r2': reverse_fold_r2,
    }

    # Incremental save so partial progress survives a crash/timeout on this long CPU run.
    pd.DataFrame(results).T.to_csv(SUMMARY_CSV)
    pd.concat(prediction_rows, ignore_index=True).to_csv(PRED_CSV, index=False)
    pd.DataFrame(fold_r2_rows).to_csv(FOLD_CSV, index=False)
    print(f'  [Saved partial results to {OUT_DIR}/ after group "{group}"]')
    sys.stdout.flush()

print('')
print(divider)
print('Summary')
print(divider)
summary_df = pd.DataFrame(results).T
print(summary_df)

prediction_tabfm = pd.concat(prediction_rows, ignore_index=True)
print('')
print('Prediction dataframe:')
print(prediction_tabfm.head())

fold_r2_df = pd.DataFrame(fold_r2_rows)
print('')
print('Per-fold R2 (both CV variants):')
print(fold_r2_df)

# Human-readable summary (R2 +/- SE for each of the 3 analyses, per study group)
summary_txt_path = os.path.join(OUT_DIR, 'summary.txt')
with open(summary_txt_path, 'w') as f:
    for group, r in results.items():
        f.write(f'{group}  (n={r["n"]}, n_train={r["n_train"]}, n_test={r["n_test"]})\n')
        f.write(f'  Forward train->test:        R2 = {r["forward_r2"]:.4f} +/- {r["forward_r2_se"]:.4f}\n')
        f.write(f'  Standard 5-fold CV (pooled): R2 = {r["standard_cv_pooled_r2"]:.4f} +/- {r["standard_cv_pooled_r2_se"]:.4f}\n')
        f.write(f'  Reverse 5-fold CV (pooled):  R2 = {r["reverse_cv_pooled_r2"]:.4f} +/- {r["reverse_cv_pooled_r2_se"]:.4f}\n')
        f.write('\n')

print('')
print(divider)
print(f'Output files written to {OUT_DIR}/:')
print(f'  - {SUMMARY_CSV}')
print(f'  - {PRED_CSV}')
print(f'  - {FOLD_CSV}')
print(f'  - {summary_txt_path}')
print(divider)

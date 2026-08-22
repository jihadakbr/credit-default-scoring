"""
Modeling untuk Home Credit Default Risk.

Dua model:
  1. Logistic Regression   -> baseline interpretable, fitur berlogika bisnis, koefisien bisa dibaca.
  2. LightGBM (Boosting)   -> model utama, tangani non-linear + missing + kategori secara native.

Fokus tambahan (key success task): PERFORMANCE STABILITY.
Kita ukur stabilitas lewat Stratified K-Fold: laporkan mean +/- std AUC antar fold,
plus gap train vs valid untuk deteksi overfit.

Metrik: AUC-ROC (metrik kompetisi) sebagai utama, dilengkapi Gini, KS, dan PR-AUC
karena konteksnya kredit dengan kelas sangat tidak seimbang (~8% default).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import lightgbm as lgb


RANDOM_STATE = 42
N_FOLDS = 5


def ks_statistic(y_true, y_score):
    """
    KS (Kolmogorov-Smirnov) = jarak maksimum antara distribusi kumulatif skor
    nasabah default vs non-default. Metrik klasik scorecard kredit; makin tinggi makin baik.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def _encode_categoricals_for_lgbm(df):
    """
    Ubah kolom object jadi tipe 'category' agar LightGBM menanganinya secara native.
    Mengembalikan (df_encoded, daftar_kolom_kategori).
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df, cat_cols


def cross_val_stability(estimator_type, X, y, lgbm_params=None, cat_cols=None, n_folds=N_FOLDS):
    """
    Jalankan Stratified K-Fold dan kembalikan out-of-fold prediksi + skor per fold.

    estimator_type: "logreg" atau "lgbm".
    Mengembalikan dict berisi oof_pred, fold AUC (train & valid), plus ringkasan stability.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_valid_auc, fold_train_auc = [], []
    models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if estimator_type == "logreg":
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, class_weight="balanced",
                    C=0.1, solver="lbfgs", random_state=RANDOM_STATE)),
            ])
            model.fit(X_tr, y_tr)
            p_tr = model.predict_proba(X_tr)[:, 1]
            p_va = model.predict_proba(X_va)[:, 1]

        elif estimator_type == "lgbm":
            model = lgb.LGBMClassifier(**lgbm_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                categorical_feature=cat_cols or "auto",
                callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
            )
            p_tr = model.predict_proba(X_tr)[:, 1]
            p_va = model.predict_proba(X_va)[:, 1]
        else:
            raise ValueError(estimator_type)

        oof[va_idx] = p_va
        fold_train_auc.append(roc_auc_score(y_tr, p_tr))
        fold_valid_auc.append(roc_auc_score(y_va, p_va))
        models.append(model)
        print(f"  fold {fold}: train AUC={fold_train_auc[-1]:.4f}  valid AUC={fold_valid_auc[-1]:.4f}")

    result = {
        "oof_pred": oof,
        "fold_valid_auc": fold_valid_auc,
        "fold_train_auc": fold_train_auc,
        "valid_auc_mean": float(np.mean(fold_valid_auc)),
        "valid_auc_std": float(np.std(fold_valid_auc)),
        "train_auc_mean": float(np.mean(fold_train_auc)),
        "overfit_gap": float(np.mean(fold_train_auc) - np.mean(fold_valid_auc)),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ks": ks_statistic(y, oof),
        "oof_pr_auc": float(average_precision_score(y, oof)),
        "models": models,
    }
    return result


def default_lgbm_params():
    """
    Hyperparameter LightGBM yang diregularisasi untuk menekan overfit dan menjaga stabilitas.

    Catatan penting soal imbalance: kita SENGAJA tidak memakai scale_pos_weight / is_unbalance.
    Metrik utama kita AUC yang berbasis ranking, dan pembobotan kelas terbukti (lewat uji coba)
    merusak ranking sekaligus membuat early stopping berhenti di ~iterasi 3 (model underfit,
    valid AUC hanya ~0.735). Tanpa pembobotan, model mencapai valid AUC ~0.78 dengan ~1200 iterasi.
    Ketidakseimbangan kelas ditangani belakangan di tahap pemilihan threshold keputusan, bukan di ranking.
    """
    return dict(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def summarize(name, res):
    """Ringkas metrik satu model jadi dict yang enak disimpan ke JSON."""
    return {
        "model": name,
        "oof_auc": round(res["oof_auc"], 5),
        "oof_gini": round(2 * res["oof_auc"] - 1, 5),
        "oof_ks": round(res["oof_ks"], 5),
        "oof_pr_auc": round(res["oof_pr_auc"], 5),
        "cv_valid_auc_mean": round(res["valid_auc_mean"], 5),
        "cv_valid_auc_std": round(res["valid_auc_std"], 5),
        "cv_train_auc_mean": round(res["train_auc_mean"], 5),
        "overfit_gap": round(res["overfit_gap"], 5),
        "fold_valid_auc": [round(a, 5) for a in res["fold_valid_auc"]],
    }

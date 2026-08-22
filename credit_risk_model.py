"""
Home Credit Default Risk - pipeline end-to-end (Python Script).

Menjalankan seluruh alur sekali jalan:
  load data -> feature engineering -> latih Logistic Regression & LightGBM
  -> evaluasi + uji stabilitas (Stratified K-Fold) -> SHAP reason codes (Gen AI)
  -> simpan metrik, figur, dan file submission.

Deliverable ini berbagi logika inti dengan notebook lewat modul di folder src/,
supaya hasil script dan notebook konsisten (tidak ada kode yang diduplikasi).

Cara pakai:
    .venv/Scripts/python.exe credit_risk_model.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interaktif supaya bisa jalan headless
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from src import features as F
from src import modeling as M
from src import reason_codes as RC

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc(curves, path):
    """Gambar kurva ROC beberapa model dalam satu plot untuk perbandingan."""
    plt.figure(figsize=(6, 5))
    for name, (y, p, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Kurva ROC - Perbandingan Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_stability(summaries, path):
    """Bar chart AUC per fold untuk menunjukkan stabilitas antar fold."""
    plt.figure(figsize=(7, 4))
    width = 0.35
    folds = np.arange(1, len(summaries[0]["fold_valid_auc"]) + 1)
    for i, s in enumerate(summaries):
        plt.bar(folds + (i - 0.5) * width, s["fold_valid_auc"], width,
                label=f"{s['model']} (mean={s['cv_valid_auc_mean']:.3f}, std={s['cv_valid_auc_std']:.3f})")
    plt.xlabel("Fold")
    plt.ylabel("Valid AUC")
    plt.title("Stabilitas Antar Fold (Stratified 5-Fold)")
    plt.ylim(0.5, 0.85)
    plt.xticks(folds)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_lgbm_importance(model, feature_names, path, top_n=20):
    """
    Top fitur menurut split count LightGBM (importance_type default = 'split',
    yaitu berapa kali fitur dipakai memecah node). Metrik ini cenderung membesarkan
    fitur kategori berkardinalitas tinggi, jadi baca bersama SHAP importance di bawah.
    """
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(7, 6))
    imp[::-1].plot(kind="barh")
    plt.title(f"Top {top_n} Feature Importance - LightGBM (split count)")
    plt.xlabel("Importance (split count)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_shap_importance(model, X, path, top_n=20, sample_size=3000, random_state=42):
    """
    Top fitur menurut SHAP importance = rata-rata |kontribusi SHAP| ke prediksi.
    Lebih adil dari split count karena mengukur besar pengaruh ke prediksi, bukan
    frekuensi pemakaian, sehingga tidak bias oleh kardinalitas kategori.
    Dihitung pada sampel acak supaya cepat tapi rankingnya tetap stabil.
    """
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    _, shap_vals = RC.compute_shap_values(model, X.iloc[idx])
    imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(7, 6))
    imp[::-1].plot(kind="barh", color="#c0392b")
    plt.title(f"Top {top_n} SHAP Importance - LightGBM (mean |SHAP|)")
    plt.xlabel("Rata-rata |kontribusi SHAP| ke log-odds default")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_target_balance(y, path):
    """Distribusi kelas target untuk menegaskan ketidakseimbangan."""
    plt.figure(figsize=(4.5, 4))
    counts = y.value_counts().sort_index()
    rate = counts[1] / counts.sum() * 100
    plt.bar(["Lancar (0)", "Default (1)"], counts.values, color=["#4c72b0", "#c44e52"])
    plt.title(f"Distribusi TARGET (default rate = {rate:.2f}%)")
    plt.ylabel("Jumlah nasabah")
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    print("=" * 70)
    print("HOME CREDIT DEFAULT RISK - PIPELINE")
    print("=" * 70)

    # 1. Feature engineering
    X_all, y = F.build_features(DATA_DIR, which="train")
    print(f"\nData latih: {X_all.shape[0]:,} baris, {X_all.shape[1]} kolom fitur")
    print(f"Default rate: {y.mean()*100:.2f}%")
    plot_target_balance(y, FIG_DIR / "target_balance.png")

    # 2. Logistic Regression - fitur interpretable
    print("\n[LogReg] Cross-validation (fitur interpretable, business logic) ...")
    X_lr = F.get_logreg_features(X_all)
    res_lr = M.cross_val_stability("logreg", X_lr, y)
    sum_lr = M.summarize("Logistic Regression", res_lr)

    # 3. LightGBM - fitur lengkap
    print("\n[LightGBM] Cross-validation (fitur lengkap) ...")
    X_gb = F.get_lgbm_features(X_all)
    X_gb, cat_cols = M._encode_categoricals_for_lgbm(X_gb)
    res_gb = M.cross_val_stability("lgbm", X_gb, y,
                                   lgbm_params=M.default_lgbm_params(), cat_cols=cat_cols)
    sum_gb = M.summarize("LightGBM", res_gb)

    # 4. Figur evaluasi
    plot_roc(
        {"Logistic Regression": (y, res_lr["oof_pred"], res_lr["oof_auc"]),
         "LightGBM": (y, res_gb["oof_pred"], res_gb["oof_auc"])},
        FIG_DIR / "roc_comparison.png",
    )
    plot_stability([sum_lr, sum_gb], FIG_DIR / "stability.png")

    # Latih 1 LightGBM final di seluruh data (untuk importance, SHAP, dan submission).
    best_iters = [m.best_iteration_ or m.n_estimators for m in res_gb["models"]]
    final_params = M.default_lgbm_params()
    final_params["n_estimators"] = int(np.mean(best_iters))
    import lightgbm as lgb
    final_gb = lgb.LGBMClassifier(**final_params)
    final_gb.fit(X_gb, y, categorical_feature=cat_cols or "auto")
    plot_lgbm_importance(final_gb, X_gb.columns.tolist(), FIG_DIR / "lgbm_importance.png")
    plot_shap_importance(final_gb, X_gb, FIG_DIR / "shap_importance.png")

    # 5. SHAP reason codes (Gen AI) untuk beberapa nasabah berisiko tinggi
    print("\n[Gen AI] Membuat SHAP reason codes untuk contoh nasabah ...")
    oof = res_gb["oof_pred"]
    high_risk_idx = np.argsort(oof)[::-1][:5]  # 5 nasabah prob default tertinggi
    X_sample = X_gb.iloc[high_risk_idx].copy()
    prob_sample = oof[high_risk_idx]
    reason_list = RC.generate_reason_codes(final_gb, X_sample, prob_sample, top_k=3, use_llm=True)
    used_llm = any(r["text_llm"] for r in reason_list)
    print(f"  Reason codes dibuat untuk {len(reason_list)} nasabah "
          f"({'LLM qwen2.5:3b aktif' if used_llm else 'mode template, Ollama tidak aktif'}).")
    for r in reason_list:
        print(f"   - idx {r['index']} (prob={r['prob']}): {r['text_final']}")

    # 6. Submission untuk application_test
    print("\n[Submission] Prediksi application_test ...")
    X_test_all, _ = F.build_features(DATA_DIR, which="test")
    test_ids = X_test_all["SK_ID_CURR"].values
    X_test_gb = F.get_lgbm_features(X_test_all)
    X_test_gb, _ = M._encode_categoricals_for_lgbm(X_test_gb)
    X_test_gb = X_test_gb.reindex(columns=X_gb.columns, fill_value=np.nan)
    for c in cat_cols:
        X_test_gb[c] = X_test_gb[c].astype("category")
    test_pred = final_gb.predict_proba(X_test_gb)[:, 1]
    pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_pred}).to_csv(
        OUT_DIR / "submission.csv", index=False)

    # 7. Simpan metrik + reason codes
    metrics = {
        "n_train": int(X_all.shape[0]),
        "n_features_lgbm": int(X_gb.shape[1]),
        "n_features_logreg": int(X_lr.shape[1]),
        "default_rate": round(float(y.mean()), 5),
        "models": [sum_lr, sum_gb],
        "genai_used_llm": bool(used_llm),
        "reason_codes_sample": [
            {"index": r["index"], "prob": r["prob"], "text_final": r["text_final"]}
            for r in reason_list
        ],
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("RINGKASAN METRIK (out-of-fold)")
    print("=" * 70)
    for s in [sum_lr, sum_gb]:
        print(f"{s['model']:<22} AUC={s['oof_auc']:.4f}  Gini={s['oof_gini']:.4f}  "
              f"KS={s['oof_ks']:.4f}  CV std={s['cv_valid_auc_std']:.4f}  gap={s['overfit_gap']:.4f}")
    print(f"\nArtefak tersimpan di: {OUT_DIR}")
    print("Selesai.")
    return metrics


if __name__ == "__main__":
    main()

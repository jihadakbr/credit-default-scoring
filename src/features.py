"""
Feature engineering untuk Home Credit Default Risk.

Scope (sesuai keputusan): application_train/test + 3 tabel kunci:
  - bureau.csv                -> riwayat kredit di biro kredit eksternal
  - previous_application.csv  -> perilaku aplikasi kredit sebelumnya di Home Credit
  - installments_payments.csv -> disiplin pembayaran cicilan

Semua fitur turunan dibuat dengan alasan logika bisnis kredit yang jelas
(kemampuan bayar, beban utang, disiplin bayar, riwayat kredit).

Dua "view" fitur disediakan lewat helper di modeling:
  - view lengkap  -> dipakai LightGBM (boleh banyak kolom, tahan kolinearitas)
  - view ringkas  -> dipakai Logistic Regression (fitur interpretable, dipilih agar
                     maknanya jelas bagi analis kredit dan tidak saling menduplikasi)
"""

from pathlib import Path
import numpy as np
import pandas as pd


def _safe_div(a, b):
    """Bagi aman: hasil NaN kalau penyebut 0 atau NaN, supaya rasio tidak meledak jadi inf."""
    b = b.replace(0, np.nan)
    return a / b


def load_application(data_dir, which="train"):
    """Baca application_train.csv atau application_test.csv apa adanya."""
    fname = "application_train.csv" if which == "train" else "application_test.csv"
    return pd.read_csv(Path(data_dir) / fname)


def clean_application(df):
    """
    Bersihkan anomali yang sudah dikenal di tabel application, tanpa mengubah makna data.
    """
    df = df.copy()

    # DAYS_EMPLOYED punya sentinel 365243 (~1000 tahun) untuk pensiunan/tidak bekerja.
    # Ini bukan nilai nyata, jadi kita jadikan NaN dan tandai dengan flag agar sinyalnya tidak hilang.
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # CODE_GENDER punya sedikit nilai 'XNA' -> perlakukan sebagai missing.
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    # Pendapatan ekstrem sangat jarang tapi menarik ekor distribusi; biarkan, model pohon tahan outlier,
    # dan untuk LogReg kita pakai rasio + scaling, jadi tidak dibuang di sini.
    return df


def add_application_business_features(df):
    """
    Fitur turunan berbasis logika bisnis kredit dari tabel application.
    Setiap fitur punya interpretasi jelas untuk analis kredit.
    """
    df = df.copy()

    # Umur & masa kerja dalam tahun (lebih intuitif daripada 'days negatif').
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    # Beban kredit terhadap kemampuan bayar.
    df["CREDIT_INCOME_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])       # besar pinjaman vs penghasilan
    df["ANNUITY_INCOME_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])     # DTI: cicilan tahunan vs penghasilan
    df["CREDIT_GOODS_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])         # pinjaman vs harga barang (uang muka)
    df["CREDIT_TERM"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])                    # proxy tenor: makin kecil makin panjang
    df["GOODS_INCOME_RATIO"] = _safe_div(df["AMT_GOODS_PRICE"], df["AMT_INCOME_TOTAL"])

    # Struktur rumah tangga & pendapatan per kepala.
    df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["INCOME_PER_CHILD"] = _safe_div(df["AMT_INCOME_TOTAL"], (df["CNT_CHILDREN"] + 1))
    df["CHILDREN_RATIO"] = _safe_div(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])

    # Stabilitas: porsi hidup yang dihabiskan bekerja (proxy kestabilan penghasilan).
    df["EMPLOYED_AGE_RATIO"] = _safe_div(-df["DAYS_EMPLOYED"], -df["DAYS_BIRTH"])

    # Skor eksternal (paling prediktif di dataset ini) diringkas jadi beberapa agregat.
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_MAX"] = ext.max(axis=1)
    df["EXT_SOURCE_STD"] = ext.std(axis=1)
    df["EXT_SOURCE_PROD"] = ext.iloc[:, 0] * ext.iloc[:, 1] * ext.iloc[:, 2]

    # Total flag dokumen yang diberikan (kelengkapan berkas) & jumlah pertanyaan ke biro kredit.
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    req_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if req_cols:
        df["CREDIT_BUREAU_REQ_TOTAL"] = df[req_cols].sum(axis=1)

    return df


def aggregate_bureau(data_dir):
    """
    Ringkas bureau.csv ke level 1 baris per SK_ID_CURR.
    Logika bisnis: berapa banyak kredit eksternal, seberapa besar utang berjalan,
    apakah pernah/ sedang menunggak, seberapa baru aktivitas kreditnya.
    """
    bureau = pd.read_csv(Path(data_dir) / "bureau.csv")

    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["IS_CLOSED"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)
    bureau["HAS_OVERDUE"] = (bureau["CREDIT_DAY_OVERDUE"] > 0).astype(int)

    agg = bureau.groupby("SK_ID_CURR").agg(
        BURO_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BURO_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BURO_CLOSED_COUNT=("IS_CLOSED", "sum"),
        BURO_OVERDUE_COUNT=("HAS_OVERDUE", "sum"),
        BURO_DAYS_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BURO_AMT_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BURO_AMT_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BURO_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BURO_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BURO_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BURO_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BURO_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        BURO_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
    )

    # Utilisasi utang: total debt / total plafon kredit. Makin tinggi makin berisiko.
    agg["BURO_DEBT_CREDIT_RATIO"] = _safe_div(agg["BURO_DEBT_SUM"], agg["BURO_CREDIT_SUM"])
    agg["BURO_ACTIVE_RATIO"] = _safe_div(agg["BURO_ACTIVE_COUNT"], agg["BURO_CREDIT_COUNT"])

    return agg.reset_index()


def aggregate_previous(data_dir):
    """
    Ringkas previous_application.csv ke level SK_ID_CURR.
    Logika bisnis: seberapa sering mengajukan, tingkat approval/ penolakan,
    besar pinjaman sebelumnya, dan apakah jumlah yang diminta dipangkas saat approval.
    """
    prev = pd.read_csv(Path(data_dir) / "previous_application.csv")

    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    # Selisih antara yang diminta vs yang disetujui (dipangkas = sinyal risiko dari keputusan lampau).
    prev["APP_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["APP_CREDIT_RATIO"] = _safe_div(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_APP_CREDIT_DIFF_MEAN=("APP_CREDIT_DIFF", "mean"),
        PREV_APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    )
    agg["PREV_APPROVED_RATIO"] = _safe_div(agg["PREV_APPROVED_COUNT"], agg["PREV_COUNT"])
    agg["PREV_REFUSED_RATIO"] = _safe_div(agg["PREV_REFUSED_COUNT"], agg["PREV_COUNT"])

    return agg.reset_index()


def aggregate_installments(data_dir):
    """
    Ringkas installments_payments.csv ke level SK_ID_CURR.
    Ini file terbesar (~723MB), jadi dibaca dengan dtype hemat memori.
    Logika bisnis: disiplin bayar cicilan = prediktor default paling langsung.
      - keterlambatan hari  = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT (positif = telat)
      - rasio bayar         = AMT_PAYMENT / AMT_INSTALMENT (< 1 = kurang bayar)
    """
    dtypes = {
        "SK_ID_CURR": "int32",
        "DAYS_INSTALMENT": "float32",
        "DAYS_ENTRY_PAYMENT": "float32",
        "AMT_INSTALMENT": "float32",
        "AMT_PAYMENT": "float32",
    }
    usecols = list(dtypes.keys())
    ins = pd.read_csv(Path(data_dir) / "installments_payments.csv", usecols=usecols, dtype=dtypes)

    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)  # telat (hari), 0 kalau tepat/awal
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)  # bayar lebih awal (hari)
    ins["PAYMENT_RATIO"] = _safe_div(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]                 # > 0 = kurang bayar
    ins["IS_LATE"] = (ins["DPD"] > 0).astype("int8")
    ins["IS_UNDERPAID"] = (ins["PAYMENT_DIFF"] > 0).astype("int8")

    agg = ins.groupby("SK_ID_CURR").agg(
        INS_COUNT=("DPD", "size"),
        INS_DPD_MEAN=("DPD", "mean"),
        INS_DPD_MAX=("DPD", "max"),
        INS_DBD_MEAN=("DBD", "mean"),
        INS_LATE_COUNT=("IS_LATE", "sum"),
        INS_LATE_RATIO=("IS_LATE", "mean"),
        INS_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INS_PAYMENT_DIFF_MEAN=("PAYMENT_DIFF", "mean"),
        INS_UNDERPAID_COUNT=("IS_UNDERPAID", "sum"),
        INS_UNDERPAID_RATIO=("IS_UNDERPAID", "mean"),
    )
    return agg.reset_index()


def build_features(data_dir, which="train", verbose=True):
    """
    Bangun tabel fitur final: application (bersih + fitur bisnis) di-join dengan
    agregat bureau, previous_application, dan installments_payments.

    Mengembalikan (df_fitur, y). Untuk test, y = None.
    """
    data_dir = Path(data_dir)
    if verbose:
        print(f"[features] load application ({which}) ...")
    app = load_application(data_dir, which)
    app = clean_application(app)
    app = add_application_business_features(app)

    if verbose:
        print("[features] aggregate bureau ...")
    app = app.merge(aggregate_bureau(data_dir), on="SK_ID_CURR", how="left")

    if verbose:
        print("[features] aggregate previous_application ...")
    app = app.merge(aggregate_previous(data_dir), on="SK_ID_CURR", how="left")

    if verbose:
        print("[features] aggregate installments_payments (file besar) ...")
    app = app.merge(aggregate_installments(data_dir), on="SK_ID_CURR", how="left")

    # Flag "punya riwayat" untuk tiap sumber (missing agregat = tidak punya riwayat, bukan error).
    app["HAS_BUREAU"] = app["BURO_CREDIT_COUNT"].notna().astype(int)
    app["HAS_PREV"] = app["PREV_COUNT"].notna().astype(int)
    app["HAS_INSTALLMENTS"] = app["INS_COUNT"].notna().astype(int)

    # Bersihkan inf yang mungkin muncul dari rasio.
    app = app.replace([np.inf, -np.inf], np.nan)

    y = None
    if "TARGET" in app.columns:
        y = app["TARGET"].astype(int)
        app = app.drop(columns=["TARGET"])

    if verbose:
        print(f"[features] selesai. shape fitur = {app.shape}")
    return app, y


# Daftar fitur interpretable untuk Logistic Regression.
# Dipilih karena maknanya jelas bagi analis kredit dan saling melengkapi, bukan menduplikasi
# informasi yang sama (mis. dari skor eksternal hanya diambil mean/min/max, bukan semua turunannya).
LOGREG_FEATURES = [
    # kemampuan & beban bayar
    "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_MAX",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CREDIT_GOODS_RATIO",
    "INCOME_PER_PERSON",
    # profil demografi & stabilitas kerja
    "AGE_YEARS", "EMPLOYED_YEARS", "EMPLOYED_AGE_RATIO", "DAYS_EMPLOYED_ANOM",
    "REGION_RATING_CLIENT", "DOCUMENT_COUNT",
    # riwayat biro kredit
    "BURO_CREDIT_COUNT", "BURO_ACTIVE_COUNT", "BURO_DEBT_CREDIT_RATIO", "BURO_OVERDUE_COUNT",
    # perilaku aplikasi sebelumnya
    "PREV_APPROVED_RATIO", "PREV_REFUSED_RATIO",
    # disiplin bayar cicilan
    "INS_DPD_MEAN", "INS_LATE_RATIO", "INS_PAYMENT_RATIO_MEAN",
    # flag ketersediaan riwayat
    "HAS_BUREAU", "HAS_PREV", "HAS_INSTALLMENTS",
]


def get_logreg_features(df):
    """Ambil hanya kolom interpretable yang tersedia untuk Logistic Regression."""
    cols = [c for c in LOGREG_FEATURES if c in df.columns]
    return df[cols].copy()


def get_lgbm_features(df):
    """
    Untuk LightGBM: pakai semua fitur numerik + kategori (di-encode di modeling).
    Buang kolom ID yang tidak informatif.
    """
    drop = ["SK_ID_CURR"]
    return df.drop(columns=[c for c in drop if c in df.columns]).copy()

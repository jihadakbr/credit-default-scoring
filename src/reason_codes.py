"""
Reason codes berbasis SHAP (komponen Gen AI).

Alur:
  1. Hitung SHAP value per nasabah dari model LightGBM (shap.TreeExplainer).
  2. Ambil fitur-fitur yang paling mendorong prediksi risiko naik (kontribusi SHAP positif terbesar).
  3. Ubah jadi "reason codes" (mirip adverse action notice di dunia kredit) secara DETERMINISTIK
     lewat template Bahasa Indonesia. Bagian ini selalu jalan dan reproducible.
  4. OPSIONAL: perhalus jadi 1-3 kalimat natural pakai LLM open-source qwen2.5:3b lewat Ollama lokal.
     Kalau Ollama tidak aktif, otomatis fallback ke kalimat template (tidak pernah gagal).

Kenapa reason codes penting untuk kredit: keputusan penolakan kredit idealnya bisa dijelaskan
("kenapa ditolak"), baik untuk kepatuhan maupun untuk feedback ke nasabah. SHAP memberi kontribusi
per-fitur per-nasabah, dan LLM membantu menuliskannya jadi bahasa manusia.
"""

import numpy as np
import shap


# Penjelasan bisnis singkat tiap fitur, dipakai membuat kalimat reason code yang mudah dipahami.
FEATURE_EXPLAIN = {
    "EXT_SOURCE_MEAN": "skor kredit eksternal rata-rata rendah",
    "EXT_SOURCE_MIN": "salah satu skor kredit eksternal sangat rendah",
    "EXT_SOURCE_MAX": "skor kredit eksternal tertinggi pun masih rendah",
    "EXT_SOURCE_1": "skor kredit eksternal 1 rendah",
    "EXT_SOURCE_2": "skor kredit eksternal 2 rendah",
    "EXT_SOURCE_3": "skor kredit eksternal 3 rendah",
    "CREDIT_INCOME_RATIO": "nilai pinjaman besar dibanding penghasilan",
    "ANNUITY_INCOME_RATIO": "cicilan tahunan berat dibanding penghasilan (DTI tinggi)",
    "CREDIT_TERM": "struktur cicilan terhadap pinjaman kurang ideal",
    "CREDIT_GOODS_RATIO": "pinjaman tinggi relatif terhadap harga barang (uang muka kecil)",
    "INS_DPD_MEAN": "rata-rata keterlambatan bayar cicilan tinggi",
    "INS_LATE_RATIO": "sering telat membayar cicilan sebelumnya",
    "INS_PAYMENT_RATIO_MEAN": "sering membayar kurang dari jumlah cicilan",
    "BURO_DEBT_CREDIT_RATIO": "utang berjalan tinggi dibanding total plafon kredit",
    "BURO_OVERDUE_COUNT": "punya catatan tunggakan di biro kredit",
    "BURO_ACTIVE_COUNT": "banyak kredit aktif berjalan",
    "BURO_CREDIT_COUNT": "banyak kredit tercatat di biro kredit",
    "BURO_AMT_OVERDUE_SUM": "total tunggakan di biro kredit cukup besar",
    "BURO_AMT_OVERDUE_MAX": "pernah menunggak dalam jumlah besar di biro kredit",
    "BURO_DAYS_OVERDUE_MAX": "pernah menunggak cukup lama di biro kredit",
    "BURO_DEBT_SUM": "total utang berjalan di biro kredit besar",
    "BURO_CREDIT_SUM": "total plafon kredit di biro kredit besar",
    "BURO_DAYS_CREDIT_MAX": "baru saja membuka kredit baru (aktivitas kredit terkini)",
    "BURO_DAYS_CREDIT_MEAN": "pola pembukaan kredit relatif baru",
    "BURO_PROLONG_SUM": "kredit di biro pernah beberapa kali diperpanjang",
    "PREV_REFUSED_RATIO": "sering ditolak pada aplikasi kredit sebelumnya",
    "PREV_REFUSED_COUNT": "beberapa aplikasi kredit sebelumnya ditolak",
    "PREV_COUNT": "banyak riwayat pengajuan kredit sebelumnya",
    "PREV_APP_CREDIT_DIFF_MEAN": "jumlah yang disetujui sering dipangkas dari yang diminta",
    "INS_DPD_MAX": "pernah sangat terlambat membayar cicilan",
    "INS_UNDERPAID_RATIO": "sering membayar cicilan kurang dari jumlah seharusnya",
    "EXT_SOURCE_STD": "skor eksternal antar sumber tidak konsisten",
    "CREDIT_TERM": "struktur cicilan terhadap pinjaman kurang ideal",
    "INCOME_PER_PERSON": "penghasilan per anggota keluarga rendah",
    "AGE_YEARS": "faktor usia",
    "EMPLOYED_YEARS": "masa kerja relatif singkat",
    "EMPLOYED_AGE_RATIO": "porsi masa kerja terhadap usia rendah",
    "DAYS_EMPLOYED_ANOM": "status pekerjaan tidak standar (mis. pensiun/ tidak bekerja)",
}


def _humanize(feature):
    """Ambil penjelasan bisnis; kalau tidak terdaftar, pakai nama fitur apa adanya."""
    return FEATURE_EXPLAIN.get(feature, f"faktor {feature}")


def compute_shap_values(model, X_sample):
    """Hitung SHAP value untuk sekumpulan nasabah dengan TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # LightGBM biner: sebagian versi mengembalikan array, sebagian list [kelas0, kelas1].
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return explainer, np.asarray(shap_values)


def top_reason_features(shap_row, feature_names, top_k=3):
    """
    Ambil top-K fitur yang paling mendorong risiko NAIK untuk satu nasabah
    (SHAP positif = mendorong probabilitas default naik).
    """
    order = np.argsort(shap_row)[::-1]  # dari kontribusi positif terbesar
    reasons = []
    for idx in order[:top_k]:
        if shap_row[idx] <= 0:
            break
        reasons.append((feature_names[idx], float(shap_row[idx])))
    return reasons


def template_reason_text(prob, reasons):
    """Kalimat reason code deterministik (tanpa LLM). Selalu tersedia sebagai fallback."""
    risk = "TINGGI" if prob >= 0.5 else ("SEDANG" if prob >= 0.2 else "RENDAH")
    if not reasons:
        return f"Estimasi risiko default {risk} (prob={prob:.2f}). Tidak ada faktor pendorong risiko yang menonjol."
    bullets = "; ".join(_humanize(f) for f, _ in reasons)
    return f"Estimasi risiko default {risk} (prob={prob:.2f}). Faktor utama: {bullets}."


def polish_with_ollama(prob, reasons, model_name="qwen2.5:3b", host="http://localhost:11434", timeout=30):
    """
    Perhalus reason codes jadi 1-3 kalimat Bahasa Indonesia natural lewat Ollama lokal.
    Mengembalikan None kalau Ollama tidak tersedia (pemanggil harus fallback ke template).
    """
    try:
        import requests
    except ImportError:
        return None

    faktor = ", ".join(_humanize(f) for f, _ in reasons) if reasons else "tidak ada faktor menonjol"
    prompt = (
        "Anda analis kredit. Tuliskan penjelasan singkat (maksimal 3 kalimat) dalam Bahasa Indonesia "
        "yang sopan dan mudah dipahami nasabah, menjelaskan kenapa pengajuan kreditnya berisiko. "
        f"Probabilitas gagal bayar: {prob:.2f}. "
        f"Faktor utama pendorong risiko: {faktor}. "
        "Jangan menambah faktor di luar yang diberikan. Jangan pakai tanda hubung panjang."
    )
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3}},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except Exception:
        return None


def generate_reason_codes(model, X_sample, prob_sample, top_k=3, use_llm=True):
    """
    Hasilkan reason codes untuk beberapa nasabah.

    Mengembalikan list dict: {index, prob, reasons, text_template, text_llm, text_final}.
    text_final = versi LLM kalau berhasil, kalau tidak jatuh ke template.
    """
    feature_names = list(X_sample.columns)
    _, shap_values = compute_shap_values(model, X_sample)

    out = []
    llm_alive = True  # sekali gagal, hentikan percobaan LLM berikutnya biar tidak lambat
    for i in range(len(X_sample)):
        prob = float(prob_sample[i])
        reasons = top_reason_features(shap_values[i], feature_names, top_k=top_k)
        text_template = template_reason_text(prob, reasons)

        text_llm = None
        if use_llm and llm_alive:
            text_llm = polish_with_ollama(prob, reasons)
            if text_llm is None:
                llm_alive = False  # Ollama mati -> berhenti coba, pakai template untuk sisanya

        out.append({
            "index": int(X_sample.index[i]),
            "prob": round(prob, 4),
            "reasons": [(f, round(v, 4)) for f, v in reasons],
            "text_template": text_template,
            "text_llm": text_llm,
            "text_final": text_llm if text_llm else text_template,
        })
    return out

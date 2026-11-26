"""
train_model.py
Jalankan sekali untuk menghasilkan model/model.pkl dari data CAESAR.
Usage: python train_model.py --data data/caesar.csv
"""
import argparse
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Konfigurasi (sama persis dengan notebook) ───────────────────────────────
SEED_HOLDOUT = 7
SEED_NESTED  = 2024
TEST_SIZE    = 0.20
MERGE_EXTREMES = True

K_LIST        = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51]
W_LIST        = [1.0, 1.5, 2.0, 3.0]
SMOTE_OPTIONS = [False, True]

FEATS  = ['gender', 'height_cm', 'weight_kg', 'chest_cm', 'waist_cm', 'hip_cm', 'bmi']
W_FEAT = 'chest_cm'

EN13402_EDGES = {
    0: [78, 86, 94, 102, 110, 118, 129],   # pria
    1: [74, 82, 90, 98, 107, 119, 131],    # wanita
}
EN13402_LABELS = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL+']
SIZE_ORDER     = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
SIZE_TO_ID     = {s: i for i, s in enumerate(SIZE_ORDER)}
ID_TO_SIZE     = {i: s for s, i in SIZE_TO_ID.items()}

# ── Fungsi bantu ─────────────────────────────────────────────────────────────
def label_en13402(chest_cm, gender):
    chest_cm = np.asarray(chest_cm, dtype=float)
    gender   = np.asarray(gender)
    labels   = np.array(EN13402_LABELS, dtype=object)
    out      = np.empty(len(chest_cm), dtype=object)
    for g, edges in EN13402_EDGES.items():
        m = (gender == g)
        out[m] = labels[np.digitize(chest_cm[m], edges)]
    return out


def remove_outliers_iqr(X_tr, y_tr):
    num_idx = [i for i, c in enumerate(FEATS) if c != 'gender']
    keep    = np.ones(len(y_tr), dtype=bool)
    for c in np.unique(y_tr):
        rows = np.where(y_tr == c)[0]
        ok   = np.ones(len(rows), dtype=bool)
        for j in num_idx:
            v = X_tr[rows, j]
            q1, q3 = np.percentile(v, [25, 75])
            iqr = q3 - q1
            ok &= (v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)
        keep[rows] = ok
    return X_tr[keep], y_tr[keep]


def apply_smote(X, y, seed):
    counts = np.bincount(y)
    counts = counts[counts > 0]
    k_sm = min(5, int(counts.min()) - 1)
    if k_sm < 1:
        return X, y
    return SMOTE(random_state=seed, k_neighbors=k_sm).fit_resample(X, y)


def select_hparams(X, y, inner_cv, seed=SEED_NESTED):
    scores = {(k, w, sm): [] for k in K_LIST for w in W_LIST for sm in SMOTE_OPTIONS}
    j = FEATS.index(W_FEAT)
    for tr, va in inner_cv.split(X, y):
        X_tr, y_tr = remove_outliers_iqr(X[tr], y[tr])
        sc = MinMaxScaler().fit(X_tr)
        Xtr_s, Xva_s = sc.transform(X_tr), sc.transform(X[va])
        for w in W_LIST:
            A, B = Xtr_s.copy(), Xva_s.copy()
            A[:, j] *= w
            B[:, j] *= w
            for sm in SMOTE_OPTIONS:
                A2, y2 = apply_smote(A, y_tr, seed) if sm else (A, y_tr)
                for k in K_LIST:
                    knn = KNeighborsClassifier(n_neighbors=k).fit(A2, y2)
                    scores[(k, w, sm)].append(
                        f1_score(y[va], knn.predict(B), average='macro'))
    return max(scores, key=lambda p: float(np.mean(scores[p])))


# ── Load & preprocess data ───────────────────────────────────────────────────
def load_data(csv_path):
    df_raw = pd.read_csv(csv_path)
    cols = {
        'gender': 'gender',
        'height': 'height_inch',
        'weight': 'weight_lbs',
        'chest_circum': 'chest_inch',
        'waist_circum_preferred': 'waist_inch',
        'hip_circum': 'hip_inch',
    }
    df = df_raw[list(cols.keys())].rename(columns=cols).copy()
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    df = df.dropna().reset_index(drop=True)

    df['height_cm'] = (df['height_inch'] * 2.54).round(2)
    df['weight_kg'] = (df['weight_lbs'] * 0.453592).round(2)
    df['chest_cm']  = (df['chest_inch'] * 2.54).round(2)
    df['waist_cm']  = (df['waist_inch'] * 2.54).round(2)
    df['hip_cm']    = (df['hip_inch'] * 2.54).round(2)
    df['bmi']       = (df['weight_kg'] / (df['height_cm'] / 100) ** 2).round(2)

    df['size_raw']   = label_en13402(df['chest_cm'].values, df['gender'].values)
    df['size_label'] = df['size_raw'].replace({'XXS': 'XS', '3XL+': 'XXL'})
    df['y']          = df['size_label'].map(SIZE_TO_ID).astype(int)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/caesar.csv')
    parser.add_argument('--out',  default='model/model.pkl')
    args = parser.parse_args()

    print(f'Loading data dari {args.data} ...')
    df = load_data(args.data)
    print(f'  Total sampel siap: {len(df)}')

    X = df[FEATS].values.astype(float)
    y = df['y'].values.astype(int)

    # Pisah hold-out (tidak dipakai untuk pelatihan model yang disimpan,
    # hanya untuk verifikasi bahwa hasil sama dengan notebook)
    idx_dev, idx_test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=SEED_HOLDOUT)
    X_dev, y_dev   = X[idx_dev], y[idx_dev]
    X_test, y_test = X[idx_test], y[idx_test]

    # Pilih hyperparameter terbaik dengan 5-fold CV pada data development
    print('Memilih hyperparameter (5-fold CV) ...')
    inner5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_NESTED)
    k_f, w_f, sm_f = select_hparams(X_dev, y_dev, inner5)
    print(f'  Terpilih: k={k_f}, bobot chest={w_f}, SMOTE={sm_f}')

    # Latih pada seluruh data development
    print('Melatih model final pada semua data development ...')
    X_tr, y_tr = remove_outliers_iqr(X_dev, y_dev)
    sc = MinMaxScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    j = FEATS.index(W_FEAT)
    X_tr_s[:, j] *= w_f
    if sm_f:
        X_tr_s, y_tr = apply_smote(X_tr_s, y_tr, SEED_NESTED)
    knn = KNeighborsClassifier(n_neighbors=k_f, metric='euclidean').fit(X_tr_s, y_tr)

    # Verifikasi cepat di hold-out
    X_te_s = sc.transform(X_test).copy()
    X_te_s[:, j] *= w_f
    y_pred = knn.predict(X_te_s)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    print(f'  Hold-out: Acc={acc*100:.2f}%  F1-macro={f1*100:.2f}%')

    # Simpan artefak
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    artefak = dict(
        knn=knn, scaler=sc,
        feats=FEATS, w_feat=W_FEAT, w_val=w_f,
        k=k_f, smote=sm_f,
        size_order=SIZE_ORDER, id_to_size=ID_TO_SIZE,
        holdout_acc=float(acc), holdout_f1=float(f1),
    )
    with open(args.out, 'wb') as f:
        pickle.dump(artefak, f)
    print(f'Model tersimpan di {args.out}')


if __name__ == '__main__':
    main()

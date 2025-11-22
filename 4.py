"""
Uczenie maszynowe w bezpieczeństwie — Projekt 1
Autor: Kamil Kasprzyk, Piotr Nowek

*** ZADANIE 4 — ODDZIELNY PROJEKT ***
------------------------------------
Klasyfikacja binarna wiadomości (spam/ham) z archiwum TREC07p
z wykorzystaniem rozmytego haszowania LSH (MinHash, MinHashLSH — datasketch).

Wymagania:
1. Użyć algorytmów MinHash i MinHashLSH.
2. Wyniki: macierz konfuzji w procentach + accuracy.
3. Przetestować wiele wartości threshold.
4. Brak blacklist i brak stemizacji — to osobne zadania.

Ten skrypt implementuje wyłącznie Zadanie 4 jako osobny projekt.
"""
import os
import re
from email import policy
from email.parser import BytesParser

import nltk
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_OK = True
except ImportError:
    DATASKETCH_OK = False

# --- KONFIGURACJA ---
TREC_DIR = r"C:\\Users\\Piotr\\Desktop\\UMB\\trec07p"
LABELS_REL = os.path.join("full", "index")
DATA_REL = os.path.join("data")

NUM_PERM = 128
THRESHOLDS = [0.3, 0.5, 0.6, 0.7, 0.8]

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# --- FUNKCJE ---

def detect_trec_structure(root):
    idx = os.path.join(root, LABELS_REL)
    data = os.path.join(root, DATA_REL)
    if not os.path.isfile(idx):
        raise FileNotFoundError(idx)
    if not os.path.isdir(data):
        raise FileNotFoundError(data)
    return idx, data


def parse_index(p):
    rows = []
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                lab = 1 if parts[0] == 'spam' else 0
                rows.append((lab, parts[1]))
    return rows


def resolve_mail_path(idx_path, rel, trec_dir):
    rel_norm = rel.replace('/', os.sep)
    base1 = os.path.dirname(idx_path)
    c1 = os.path.normpath(os.path.join(base1, rel_norm))
    if os.path.isfile(c1): return c1
    c2 = os.path.normpath(os.path.join(trec_dir, rel_norm))
    if os.path.isfile(c2): return c2
    c3 = os.path.normpath(os.path.join(trec_dir, 'data', os.path.basename(rel_norm)))
    return c3


def parse_email(path):
    with open(path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    subject = msg.get('subject', '') or ''
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                try:
                    body += part.get_content() or ''
                except:
                    pass
    else:
        try:
            body = msg.get_content() or ''
        except:
            body = ''
    return subject + " " + body


def tokenize(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    return [w for w in text.split() if w not in STOPWORDS]


def tokens_to_minhash(tokens):
    m = MinHash(num_perm=NUM_PERM)
    for t in set(tokens):
        m.update(t.encode('utf-8'))
    return m


def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]).astype(float)
    cm_pct = cm / cm.sum() * 100
    acc = accuracy_score(y_true, y_pred) * 100
    return cm_pct, acc

# --- MAIN ---

def main():
    print("[ZADANIE 4 — ODDZIELNY PROJEKT] LSH / MinHash")
    if not DATASKETCH_OK:
        print("Brak biblioteki datasketch. Zainstaluj: pip install datasketch")
        return

    idx_path, data_dir = detect_trec_structure(TREC_DIR)
    rows = parse_index(idx_path)

    # podział 80/20
    split = int(0.8 * len(rows))
    train_rows = rows[:split]
    test_rows = rows[split:]

    X_train_raw = []
    X_test_raw = []
    y_train = []
    y_test = []

    # --- Wczytanie treści ---
    for lab, rel in train_rows:
        p = resolve_mail_path(idx_path, rel, TREC_DIR)
        if not os.path.isfile(p): continue
        X_train_raw.append(parse_email(p))
        y_train.append(lab)

    for lab, rel in test_rows:
        p = resolve_mail_path(idx_path, rel, TREC_DIR)
        if not os.path.isfile(p): continue
        X_test_raw.append(parse_email(p))
        y_test.append(lab)

    # --- Tokenizacja ---
    X_train_tok = [tokenize(t) for t in X_train_raw]
    X_test_tok = [tokenize(t) for t in X_test_raw]

    # --- MinHash ---
    print("Tworzenie MinHash...")
    mh_train = [tokens_to_minhash(t) for t in X_train_tok]

    # --- Test dla różnych threshold ---
    for thr in THRESHOLDS:
        print(f"[LSH] threshold = {thr}")

        lsh = MinHashLSH(threshold=thr, num_perm=NUM_PERM)
        for i, m in enumerate(mh_train):
            lsh.insert(str(i), m)

        y_pred = []
        for toks in X_test_tok:
            mh = tokens_to_minhash(toks)
            neigh = lsh.query(mh)
            if not neigh:
                y_pred.append(0)
            else:
                labels = [y_train[int(n)] for n in neigh]
                y_pred.append(1 if sum(labels) >= (len(labels)/2) else 0)

        cm, acc = evaluate(y_test, y_pred)
        print("Macierz konfuzji [%]:")
        print(np.array_str(cm, precision=2, suppress_small=True))
        print(f"Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uczenie maszynowe w bezpieczeństwie — Projekt 1
Autor: Kamil Kasprzyk, Piotr Nowek

------------------------------------
Klasyfikacja binarna wiadomości (spam/ham) z archiwum TREC07p
z wykorzystaniem algorytmu Naive Bayes (MultinomialNB — scikit-learn).


"""
import os
import re
from email import policy
from email.parser import BytesParser

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# --- KONFIGURACJA ---
TREC_DIR = r"C:\\Users\\Piotr\\Desktop\\UMB\\trec07p"
LABELS_REL = os.path.join("full", "index")
DATA_REL = os.path.join("data")

# NLTK — stopwords + stemmer (do wariantu B)
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
STEMMER = SnowballStemmer('english')


# --- FUNKCJE POMOCNICZE ---

def detect_trec_structure(root: str):
    idx = os.path.join(root, LABELS_REL)
    data = os.path.join(root, DATA_REL)
    if not os.path.isfile(idx):
        raise FileNotFoundError(idx)
    if not os.path.isdir(data):
        raise FileNotFoundError(data)
    return idx, data


def parse_index(p: str):
    """Czyta plik index i zwraca listę (label, relpath), gdzie label: 1=spam, 0=ham."""
    rows = []
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                lab = 1 if parts[0].lower() == 'spam' else 0
                rows.append((lab, parts[1]))
    return rows


def resolve_mail_path(idx_path: str, rel: str, trec_dir: str) -> str:
    """Buduje absolutną ścieżkę do pliku e‑maila na podstawie ścieżki relatywnej z index."""
    rel_norm = rel.replace('/', os.sep)
    base1 = os.path.dirname(idx_path)
    c1 = os.path.normpath(os.path.join(base1, rel_norm))
    if os.path.isfile(c1):
        return c1
    c2 = os.path.normpath(os.path.join(trec_dir, rel_norm))
    if os.path.isfile(c2):
        return c2
    c3 = os.path.normpath(os.path.join(trec_dir, 'data', os.path.basename(rel_norm)))
    return c3


def parse_email(path: str) -> str:
    """Czyta e‑mail z pliku i zwraca subject + body jako jeden ciąg tekstu."""
    with open(path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    subject = msg.get('subject', '') or ''
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                try:
                    body += part.get_content() or ''
                except Exception:
                    pass
    else:
        try:
            body = msg.get_content() or ''
        except Exception:
            body = ''
    return subject + " " + body


def preprocess_tokens(text: str) -> list:
    """Usuwa znaki nieliterowe, zamienia na małe litery, usuwa stopwords i stemizuje (Snowball)."""
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = []
    for w in text.split():
        if w in STOPWORDS:
            continue
        tokens.append(STEMMER.stem(w))
    return tokens


def confusion_percent(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    cm_pct = cm / cm.sum() * 100.0
    acc = accuracy_score(y_true, y_pred) * 100.0
    return cm_pct, acc


# Main

def main():
    print("[ZADANIE 5 — Naive Bayes jako oddzielny projekt]")
    idx_path, data_dir = detect_trec_structure(TREC_DIR)
    rows = parse_index(idx_path)

    # chronologiczny podział 80/20
    split = int(0.8 * len(rows))
    train_rows = rows[:split]
    test_rows = rows[split:]

    X_train_raw, y_train = [], []
    X_test_raw, y_test = [], []

    print("[INFO] Wczytywanie e‑maili...")
    for lab, rel in train_rows:
        p = resolve_mail_path(idx_path, rel, TREC_DIR)
        if not os.path.isfile(p):
            continue
        X_train_raw.append(parse_email(p))
        y_train.append(lab)

    for lab, rel in test_rows:
        p = resolve_mail_path(idx_path, rel, TREC_DIR)
        if not os.path.isfile(p):
            continue
        X_test_raw.append(parse_email(p))
        y_test.append(lab)

    # --- WARIANT A: SUROWY TEKST (bez stopwords/stemizacji) ---
    print("[WARIANT A] Naive Bayes — surowy tekst (subject+body)")
    vect_raw = CountVectorizer(lowercase=True)  # standardowe przetwarzanie, bez NLTK
    Xtr_raw = vect_raw.fit_transform(X_train_raw)
    Xte_raw = vect_raw.transform(X_test_raw)

    nb_raw = MultinomialNB()
    nb_raw.fit(Xtr_raw, y_train)
    y_pred_raw = nb_raw.predict(Xte_raw)

    cm_raw, acc_raw = confusion_percent(y_test, y_pred_raw)
    print("Macierz konfuzji [%] — surowy tekst:")
    print(np.array_str(cm_raw, precision=2, suppress_small=True))
    print(f"Accuracy (surowy tekst): {acc_raw:.2f}%")

    # --- WARIANT B: TEKST PO PRE‑PROCESSINGU (stopwords + stemizacja) ---
    print("[WARIANT B] Naive Bayes — po usunięciu stopwords + stemizacji (NLTK)")
    # Tokenizujemy i stemizujemy ręcznie, potem łączymy tokeny z powrotem w stringi
    X_train_proc_tokens = [preprocess_tokens(t) for t in X_train_raw]
    X_test_proc_tokens = [preprocess_tokens(t) for t in X_test_raw]

    X_train_proc = [" ".join(toks) for toks in X_train_proc_tokens]
    X_test_proc = [" ".join(toks) for toks in X_test_proc_tokens]

    # CountVectorizer pracuje teraz na gotowych tokenach (oddzielonych spacjami)
    vect_proc = CountVectorizer(lowercase=False, token_pattern=r"[^ ]+")
    Xtr_proc = vect_proc.fit_transform(X_train_proc)
    Xte_proc = vect_proc.transform(X_test_proc)

    nb_proc = MultinomialNB()
    nb_proc.fit(Xtr_proc, y_train)
    y_pred_proc = nb_proc.predict(Xte_proc)

    cm_proc, acc_proc = confusion_percent(y_test, y_pred_proc)
    print("Macierz konfuzji [%] — po pre‑processingu:")
    print(np.array_str(cm_proc, precision=2, suppress_small=True))
    print(f"Accuracy (po pre‑processingu): {acc_proc:.2f}%")

    # --- PODSUMOWANIE PORÓWNANIA ---
    print("[PORÓWNANIE — ZADANIE 5]")
    print(f"Accuracy — surowy tekst        : {acc_raw:.2f}%")
    print(f"Accuracy — po stopwords+stemmer: {acc_proc:.2f}%")



if __name__ == "__main__":
    main()

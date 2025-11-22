#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uczenie maszynowe w bezpieczeństwie — Projekt 1
Autor: Kamil Kasprzyk, Piotr Nowek
Zadanie 2
---------
Wykorzystując technikę zakazanych słów kluczowych (blacklist), dokonać klasyfikacji binarnej wiadomości
z podziałem na spam i ham. Usuwamy stopwords, dokonujemy stemizacji, tokenizacji, a następnie klasyfikujemy.
"""
import os
import re
import email
from email import policy
from email.parser import BytesParser
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

TREC_DIR = r"C:\\Users\\Piotr\\Desktop\\UMB\\trec07p"
LABELS_REL = os.path.join("full", "index")
DATA_REL = os.path.join("data")

def detect_trec_structure(root: str):
    idx_path = os.path.join(root, LABELS_REL)
    data_dir = os.path.join(root, DATA_REL)
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"Nie znaleziono pliku etykiet: {idx_path}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Nie znaleziono katalogu danych: {data_dir}")
    return idx_path, data_dir

def parse_index(index_path: str):
    rows = []
    with open(index_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                label = 1 if parts[0].lower() == 'spam' else 0
                relpath = parts[1]
                rows.append((label, relpath))
    return rows

def resolve_mail_path(index_path: str, relpath: str, trec_dir: str) -> str:
    """Bezpiecznie zbuduj absolutną ścieżkę do pliku e‑maila z relatywnej ścieżki w index."""
    rel_norm = relpath.replace('/', os.sep).replace('\\\\', os.sep)
    base1 = os.path.dirname(index_path)
    cand1 = os.path.normpath(os.path.join(base1, rel_norm))
    if os.path.isfile(cand1):
        return cand1
    cand2 = os.path.normpath(os.path.join(trec_dir, rel_norm))
    if os.path.isfile(cand2):
        return cand2
    data_dir = os.path.join(trec_dir, 'data')
    cand3 = os.path.normpath(os.path.join(data_dir, os.path.basename(rel_norm)))
    return cand3

def parse_email(path: str):
    with open(path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    subject = msg.get('subject', '')
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == 'text/plain':
                try:
                    body += part.get_content()
                except Exception:
                    continue
    else:
        try:
            body = msg.get_content()
        except Exception:
            body = ''
    return (subject + ' ' + body)

def preprocess(text):
    nltk.download('stopwords', quiet=True)
    stop = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop]
    return tokens

def build_blacklist(train_tokens, y_train, min_count=20, min_spam_ratio=0.9):
    from collections import Counter
    freq_spam, freq_all = Counter(), Counter()
    for tokens, lab in zip(train_tokens, y_train):
        uniq = set(tokens)
        freq_all.update(uniq)
        if lab == 1:
            freq_spam.update(uniq)
    blacklist = set()
    for tok, c in freq_all.items():
        if c >= min_count:
            ratio = freq_spam[tok] / float(c)
            if ratio >= min_spam_ratio:
                blacklist.add(tok)
    return blacklist

def predict_blacklist(blacklist, test_tokens):
    y_pred = []
    for tokens in test_tokens:
        y_pred.append(1 if any(t in blacklist for t in tokens) else 0)
    return y_pred

def confusion_percent(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]).astype(float)
    cm_pct = cm / cm.sum() * 100
    acc = accuracy_score(y_true, y_pred) * 100
    return cm_pct, acc

def main():
    print("[ZADANIE 2] Klasyfikacja blacklist — spam vs ham")
    idx_path, data_dir = detect_trec_structure(TREC_DIR)
    rows = parse_index(idx_path)

    split_point = int(0.8 * len(rows))
    train_rows, test_rows = rows[:split_point], rows[split_point:]

    X_train, y_train, X_test, y_test = [], [], [], []

    missing_train = 0
    for lab, rel in train_rows:
        mail_path = resolve_mail_path(idx_path, rel, TREC_DIR)
        if not os.path.isfile(mail_path):
            print(f"[WARN] Brak pliku e‑maila (train): {mail_path} (rel='{rel}')")
            data_dir = os.path.join(TREC_DIR, 'data')
            try:
                sample = sorted(os.listdir(data_dir))[:5]
                print(f"[INFO] Przykładowe pliki w {data_dir}: {sample}")
            except Exception as e:
                print(f"[INFO] Nie mogę odczytać listy z {data_dir}: {e}")
            missing_train += 1
            continue
        text = parse_email(mail_path)
        X_train.append(preprocess(text))
        y_train.append(lab)
    if missing_train:
        print(f"[INFO] Pomięto {missing_train} wiadomości treningowych z powodu braku plików.")

    missing_test = 0
    for lab, rel in test_rows:
        mail_path = resolve_mail_path(idx_path, rel, TREC_DIR)
        if not os.path.isfile(mail_path):
            print(f"[WARN] Brak pliku e‑maila (test): {mail_path} (rel='{rel}')")
            missing_test += 1
            continue
        text = parse_email(mail_path)
        X_test.append(preprocess(text))
        y_test.append(lab)
    if missing_test:
        print(f"[INFO] Pomięto {missing_test} wiadomości testowych z powodu braku plików.")

    blacklist = build_blacklist(X_train, y_train)
    y_pred = predict_blacklist(blacklist, X_test)

    cm, acc = confusion_percent(y_test, y_pred)
    print("Macierz konfuzji [%]:")
    print(np.array_str(cm, precision=2, suppress_small=True))
    print(f"Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    main()

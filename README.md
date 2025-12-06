# Uczenie maszynowe w bezpieczeństwie — Projekt 1  
Autor: Kamil Kasprzyk, Piotr Nowek

---

## Zadanie 2 — Klasyfikacja z użyciem blacklisty

Wykorzystując technikę zakazanych słów kluczowych (*blacklist*), dokonano klasyfikacji binarnej wiadomości z podziałem na **spam** i **ham**. Przetwarzanie obejmuje:
- usunięcie stopwords,
- stemizację,
- tokenizację,
- klasyfikację na podstawie obecności słów z czarnej listy.

---

### Kod

```python
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

TREC_DIR = r"C:\\Users\\kamil\\Desktop\\UMB\\trec07p"
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
    # Bezpiecznie zbuduj absolutną ścieżkę do pliku e-maila z relatywnej ścieżki w index.
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
            print(f"[WARN] Brak pliku e-maila (train): {mail_path} (rel='{rel}')")
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
            print(f"[WARN] Brak pliku e-maila (test): {mail_path} (rel='{rel}')")
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
```

---

### Wyniki

Macierz konfuzji [%]:

```text
[[19.65 12.16]
 [ 7.57 60.62]]
```

**Accuracy:** 80.27%

### Interpretacja

Osiągnięta dokładność modelu wyniosła **80,27%**, a macierz konfuzji wskazuje, że:

- klasyfikator dobrze wykrywa spam (wysoki udział poprawnych wykryć — **60,62%**),
- jednak popełnia zauważalne błędy na klasie *ham* — około **12,16%** wiadomości nie będących spamem zostało błędnie sklasyfikowanych jako spam.

---

## Zadanie 4 — LSH / MinHash (oddzielny projekt)

Klasyfikacja binarna wiadomości (spam/ham) z archiwum **TREC07p** z wykorzystaniem rozmytego haszowania **LSH (MinHash, MinHashLSH — datasketch)**.

### Wymagania

1. Użyć algorytmów **MinHash** i **MinHashLSH**.
2. Wyniki: macierz konfuzji w procentach + accuracy.
3. Przetestować wiele wartości **threshold**.
4. Brak blacklist i brak stemizacji — to osobne zadania.

---

### Kod

```python
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
TREC_DIR = r"C:\\Users\\kamil\\Desktop\\UMB\\trec07p"
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
    if os.path.isfile(c1):
        return c1
    c2 = os.path.normpath(os.path.join(trec_dir, rel_norm))
    if os.path.isfile(c2):
        return c2
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
```

---

### Wyniki

#### Threshold = **0.3**

```text
[[30.1   1.71]
 [12.14 56.05]]
```

Accuracy: **86.15%**

#### Threshold = **0.5**

```text
[[31.64  0.17]
 [33.47 34.72]]
```

Accuracy: **66.36%**

#### Threshold = **0.6**

```text
[[31.75  0.06]
 [41.05 27.14]]
```

Accuracy: **58.89%**

#### Threshold = **0.7**

```text
[[31.77  0.04]
 [45.68 22.51]]
```

Accuracy: **54.28%**

#### Threshold = **0.8**

```text
[[31.78  0.03]
 [50.5  17.69]]
```

Accuracy: **49.47%**

---

### Wnioski

Metoda **MinHash + LSH** wykazuje silną zależność od parametru *threshold*.

* Najlepsze wyniki przy **threshold = 0.3** — accuracy **86%**.
* Przy wyższych wartościach threshold rośnie liczba fałszywych negatywów.
* Dla progów 0.5–0.8 skuteczność spada nawet poniżej **50%**.
* LSH działa dobrze jedynie przy niskich progach, gdzie tolerowana jest umiarkowana różnorodność treści.

**Podsumowanie:** LSH jest dobre do znajdowania podobnych dokumentów, ale nie jako generalny klasyfikator tekstu.

---

## Zadanie 5 — Naive Bayes

Klasyfikacja binarna wiadomości (spam/ham) z archiwum **TREC07p** przy użyciu algorytmu **Multinomial Naive Bayes**.

---

### Kod

```python
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
TREC_DIR = r"C:\\Users\\kamil\\Desktop\\UMB\\trec07p"
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
    # Czyta plik index i zwraca listę (label, relpath), gdzie label: 1=spam, 0=ham.
    rows = []
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                lab = 1 if parts[0].lower() == 'spam' else 0
                rows.append((lab, parts[1]))
    return rows


def resolve_mail_path(idx_path: str, rel: str, trec_dir: str) -> str:
    # Buduje absolutną ścieżkę do pliku e-maila na podstawie ścieżki relatywnej z index.
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
    # Czyta e-mail z pliku i zwraca subject + body jako jeden ciąg tekstu.
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
    # Usuwa znaki nieliterowe, zamienia na małe litery, usuwa stopwords i stemizuje (Snowball).
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

    print("[INFO] Wczytywanie e-maili...")
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

    # --- WARIANT B: TEKST PO PRE-PROCESSINGU (stopwords + stemizacja) ---
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
    print("Macierz konfuzji [%] — po pre-processingu:")
    print(np.array_str(cm_proc, precision=2, suppress_small=True))
    print(f"Accuracy (po pre-processingu): {acc_proc:.2f}%")

    # --- PODSUMOWANIE PORÓWNANIA ---
    print("[PORÓWNANIE — ZADANIE 5]")
    print(f"Accuracy — surowy tekst        : {acc_raw:.2f}%")
    print(f"Accuracy — po stopwords+stemmer: {acc_proc:.2f}%")


if __name__ == "__main__":
    main()
```

---

### Wyniki

#### Wariant A — surowy tekst

```text
[[31.26  0.54]
 [ 1.48 66.71]]
```

Accuracy: **97.98%**

#### Wariant B — po usunięciu stopwords + stemizacji

```text
[[31.25  0.56]
 [ 2.76 65.43]]
```

Accuracy: **96.67%**

---

### Wnioski

* Naive Bayes osiągnął **najwyższą skuteczność (~98%)** spośród wszystkich klasycznych metod.
* Preprocessing minimalnie obniżył skuteczność modelu.
* Klasyfikator dobrze wychwytuje różnice w rozkładach słów między spamem a hamem.

---

## Zadanie 6 — Klasyfikacja z użyciem sieci neuronowych (TF-IDF + Dense)

Klasyfikacja binarna wiadomości (spam/ham) z archiwum **TREC07p** przy użyciu **głębokich sieci neuronowych** w bibliotece **TensorFlow/Keras**.  
Wejściem modeli jest reprezentacja **TF-IDF** (do 20 000 cech), a na wyjściu pojedynczy neuron z aktywacją sigmoid (prawdopodobieństwo spamu).

Przetestowano trzy architektury:

* **Model_A** – prosta sieć: jedna warstwa ukryta 128 neuronów (ReLU),
* **Model_B** – sieć dwuwarstwowa z dropoutem 0.5,
* **Model_C** – najgłębsza sieć: 3 warstwy ukryte (512–256–128) z dwoma dropoutami 0.5.

---

### Kod

```python
import os
import re
import numpy as np
from email import policy
from email.parser import BytesParser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

# --- Ścieżki ---
TREC_DIR = r"C:\\Users\\kamil\\Desktop\\UMB\\trec07p"
LABELS_REL = os.path.join("full", "index")
DATA_REL = os.path.join("data")

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
                lab = 1 if parts[0].lower() == 'spam' else 0
                rows.append((lab, parts[1]))
    return rows

def resolve_mail_path(idx_path, rel, trec_dir):
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
    text = subject + " " + body
    # lekki cleaning
    text = re.sub(r'\s+', ' ', text)
    return text

# --- Wczytanie danych ---
idx_path, data_dir = detect_trec_structure(TREC_DIR)
rows = parse_index(idx_path)

texts = []
labels = []

for lab, rel in rows:
    p = resolve_mail_path(idx_path, rel, TREC_DIR)
    if not os.path.isfile(p):
        continue
    txt = parse_email(p)
    texts.append(txt)
    labels.append(lab)

labels = np.array(labels)

# --- Train / Test split (np. 80/20) ---
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- TF-IDF ---
vectorizer = TfidfVectorizer(max_features=20000)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

X_train = X_train.toarray().astype("float32")
X_test = X_test.toarray().astype("float32")

# --- Funkcje do budowy modelu ---
def build_model_A(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_model_B(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_model_C(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

input_dim = X_train.shape[1]

models_dict = {
    "Model_A": build_model_A(input_dim),
    "Model_B": build_model_B(input_dim),
    "Model_C": build_model_C(input_dim),
}

for name, model in models_dict.items():
    print(f"\n=== Trening {name} ===")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=64,
        verbose=1
    )

    # Predykcja
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).astype(float)
    cm_pct = cm / cm.sum() * 100
    acc = accuracy_score(y_test, y_pred) * 100

    print(f"[{name}] Macierz konfuzji [%]:")
    print(cm_pct)
    print(f"[{name}] Accuracy: {acc:.2f}%")
```

---

### Wyniki

#### Model_A (1 warstwa ukryta, 128 neuronów)

```text
Epoch 1/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - accuracy: 0.9872 - loss: 0.0704 - val_accuracy: 0.9959 - val_loss: 0.0193
Epoch 2/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - accuracy: 0.9975 - loss: 0.0101 - val_accuracy: 0.9960 - val_loss: 0.0187
Epoch 3/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.9988 - loss: 0.0053 - val_accuracy: 0.9962 - val_loss: 0.0188
Epoch 4/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.9991 - loss: 0.0040 - val_accuracy: 0.9954 - val_loss: 0.0204
Epoch 5/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.9991 - loss: 0.0034 - val_accuracy: 0.9960 - val_loss: 0.0211
472/472 ━━━━━━━━━━━━━━━━━━━━ 0s 621us/step
[Model_A] Macierz konfuzji [%]:
[[33.1344471   0.3049589 ]
 [ 0.1922567  66.36833731]]
[Model_A] Accuracy: 99.50%
```

#### Model_B (2 warstwy ukryte + dropout 0.5)

```text
Epoch 1/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 9s 9ms/step - accuracy: 0.9830 - loss: 0.0482 - val_accuracy: 0.9937 - val_loss: 0.0202
Epoch 2/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 11s 13ms/step - accuracy: 0.9975 - loss: 0.0090 - val_accuracy: 0.9957 - val_loss: 0.0175
Epoch 3/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 10s 11ms/step - accuracy: 0.9986 - loss: 0.0053 - val_accuracy: 0.9954 - val_loss: 0.0198
Epoch 4/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 10s 12ms/step - accuracy: 0.9989 - loss: 0.0042 - val_accuracy: 0.9945 - val_loss: 0.0213
Epoch 5/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 11s 13ms/step - accuracy: 0.9990 - loss: 0.0037 - val_accuracy: 0.9952 - val_loss: 0.0230
472/472 ━━━━━━━━━━━━━━━━━━━━ 0s 797us/step
[Model_B] Macierz konfuzji [%]:
[[33.12118801  0.31821798]
 [ 0.17236807 66.38822593]]
[Model_B] Accuracy: 99.51%
```

#### Model_C (3 warstwy ukryte, 2× dropout 0.5)

```text
Epoch 1/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 26s 30ms/step - accuracy: 0.9819 - loss: 0.0476 - val_accuracy: 0.9947 - val_loss: 0.0236
Epoch 2/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 24s 28ms/step - accuracy: 0.9972 - loss: 0.0108 - val_accuracy: 0.9955 - val_loss: 0.0218
Epoch 3/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 24s 28ms/step - accuracy: 0.9985 - loss: 0.0059 - val_accuracy: 0.9955 - val_loss: 0.0260
Epoch 4/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 24s 28ms/step - accuracy: 0.9989 - loss: 0.0047 - val_accuracy: 0.9952 - val_loss: 0.0282
Epoch 5/5
849/849 ━━━━━━━━━━━━━━━━━━━━ 24s 28ms/step - accuracy: 0.9991 - loss: 0.0042 - val_accuracy: 0.9957 - val_loss: 0.0248
472/472 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step
[Model_C] Macierz konfuzji [%]:
[[33.08804031  0.35136569]
 [ 0.13259082 66.42800318]]
[Model_C] Accuracy: 99.52%
```

---

### Wnioski

* Wszystkie trzy sieci neuronowe osiągnęły **bardzo wysoką skuteczność (~99.5%)**, zauważalnie wyższą niż Naive Bayes (~98%).
* Macierze konfuzji pokazują **bardzo mały odsetek błędów**:
  * fałszywie dodatnie (ham → spam): ok. **0.3%**,
  * fałszywie ujemne (spam → ham): ok. **0.13–0.2%**.
* Różnice między Model_A, Model_B i Model_C są **minimalne** (rzędu 0.01–0.02 pp), co sugeruje, że:
  * już **prosta architektura (Model_A)** + dobra reprezentacja (TF-IDF) wystarcza do osiągnięcia wyniku bliskiego maksimum,
  * dalsze zwiększanie głębokości i liczby parametrów nie daje istotnych korzyści na tym zbiorze, a zwiększa czas trenowania.

---

## Ogólne wnioski z projektu

* **Blacklist** (reguły oparte na słowach kluczowych) daje najsłabsze wyniki i jest mało elastyczna.
* **MinHash + LSH** jest przydatne do wyszukiwania podobnych wiadomości, ale słabo sprawdza się jako samodzielny klasyfikator.
* **Naive Bayes** jest bardzo skutecznym i lekkim baseline’em dla detekcji spamu.
* **Sieci neuronowe z TF-IDF** osiągają najlepsze rezultaty (**~99.5% accuracy**) i praktycznie minimalizują liczbę błędów klasyfikacji na zbiorze TREC07p.

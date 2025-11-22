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
TREC_DIR = r"C:\Users\kamil\Desktop\UMB\trec07p"
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

# --- Funkcja do budowy modelu ---
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

# ============================================================
#  Student Performance Prediction using LSTM & ANN
#  Authors: Phani Charan | Yashwant Pavan Kumar | Mohit
# ============================================================

# ── 1. Imports ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  Student Performance Prediction — LSTM vs ANN")
print("=" * 55)

# ── 2. Dataset Creation ─────────────────────────────────────
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'study_hours':      np.random.randint(1, 10, n),
    'attendance':       np.random.randint(50, 100, n),
    'assignment_score': np.random.randint(40, 100, n),
    'gpa':              np.round(np.random.uniform(5, 10, n), 2),
    'participation':    np.random.randint(1, 10, n),
    'test_score':       np.random.randint(40, 100, n),
    'sleep_hours':      np.random.randint(4, 10, n)
})

def label(row):
    score = (row['study_hours']
             + row['attendance'] / 10
             + row['assignment_score'] / 10
             + row['gpa'])
    if score < 20:
        return 'Fail'
    elif score < 25:
        return 'Pass'
    elif score < 30:
        return 'Average'
    elif score < 35:
        return 'Above Average'
    else:
        return 'Good'

data['performance'] = data.apply(label, axis=1)

print("\n📊 Dataset Sample (first 5 rows):")
print(data.head())
print(f"\n📦 Dataset Shape : {data.shape}")
print(f"🏷️  Class Distribution:\n{data['performance'].value_counts()}")

# ── 3. Preprocessing ─────────────────────────────────────────
X = data.drop('performance', axis=1)
y = data['performance']

scaler  = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded    = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)

# Reshape for LSTM  →  (samples, timesteps=1, features)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

num_classes = y_categorical.shape[1]
print(f"\n✅ Preprocessing done — Train: {X_train.shape}, Test: {X_test.shape}")

# ── 4. LSTM Model ────────────────────────────────────────────
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
         return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
lstm_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

print("\n🧠 LSTM Model Summary:")
lstm_model.summary()

# ── 5. ANN Model ─────────────────────────────────────────────
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
ann_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

print("\n🧠 ANN Model Summary:")
ann_model.summary()

# ── 6. Training ──────────────────────────────────────────────
print("\n🚀 Training LSTM …")
history_lstm = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=20, batch_size=16,
    validation_data=(X_test_lstm, y_test),
    verbose=1
)

print("\n🚀 Training ANN …")
history_ann = ann_model.fit(
    X_train, y_train,
    epochs=20, batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# ── 7. Evaluation ────────────────────────────────────────────
lstm_pred = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
ann_pred  = np.argmax(ann_model.predict(X_test),       axis=1)
y_true    = np.argmax(y_test, axis=1)

print("\n" + "=" * 55)
print("  LSTM Classification Report")
print("=" * 55)
print(classification_report(y_true, lstm_pred,
                             target_names=le.classes_))

print("=" * 55)
print("  ANN Classification Report")
print("=" * 55)
print(classification_report(y_true, ann_pred,
                             target_names=le.classes_))

# Accuracy comparison
lstm_acc = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)[1]
ann_acc  = ann_model.evaluate(X_test,       y_test, verbose=0)[1]

print("\n" + "=" * 55)
print("  Model Performance")
print("=" * 55)
print(f"  LSTM Accuracy : {lstm_acc * 100:.2f}%")
print(f"  ANN  Accuracy : {ann_acc  * 100:.2f}%")
if lstm_acc > ann_acc:
    print("  ✅ LSTM performs better than ANN")
elif ann_acc > lstm_acc:
    print("  ✅ ANN performs better than LSTM")
else:
    print("  🤝 Both models perform equally")
print("=" * 55)

# ── 8. Visualizations ────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {'train': '#4A90D9', 'val': '#E87040'}

def plot_history(history, model_name, save_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — Training History',
                 fontsize=15, fontweight='bold', y=1.02)

    # Accuracy
    axes[0].plot(history.history['accuracy'],
                 color=COLORS['train'], linewidth=2, label='Train')
    axes[0].plot(history.history['val_accuracy'],
                 color=COLORS['val'],   linewidth=2, label='Validation')
    axes[0].set_title('Accuracy', fontsize=13)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].set_ylim([0, 1])

    # Loss
    axes[1].plot(history.history['loss'],
                 color=COLORS['train'], linewidth=2, label='Train')
    axes[1].plot(history.history['val_loss'],
                 color=COLORS['val'],   linewidth=2, label='Validation')
    axes[1].set_title('Loss', fontsize=13)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'outputs/{save_prefix}_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved outputs/{save_prefix}_history.png")

def plot_confusion(y_t, y_p, model_name, save_prefix):
    cm = confusion_matrix(y_t, y_p)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                linewidths=0.5, ax=ax)
    ax.set_title(f'{model_name} — Confusion Matrix',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label',      fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'outputs/{save_prefix}_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved outputs/{save_prefix}_confusion.png")

def plot_comparison(lstm_a, ann_a):
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(['LSTM', 'ANN'],
                  [lstm_a * 100, ann_a * 100],
                  color=['#4A90D9', '#E87040'],
                  width=0.45, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, [lstm_a * 100, ann_a * 100]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.set_title('Model Comparison — Accuracy',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Models',       fontsize=12)
    ax.axhline(y=80, color='gray', linestyle='--',
               linewidth=1, alpha=0.6, label='80% baseline')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  💾 Saved outputs/model_comparison.png")

def plot_class_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = data['performance'].value_counts()
    palette = sns.color_palette('pastel', len(counts))
    bars = ax.bar(counts.index, counts.values, color=palette,
                  edgecolor='grey', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                str(val), ha='center', va='bottom', fontsize=11)
    ax.set_title('Class Distribution in Dataset',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Performance Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  💾 Saved outputs/class_distribution.png")

print("\n📊 Generating visualizations …")
plot_history(history_lstm, 'LSTM', 'lstm')
plot_history(history_ann,  'ANN',  'ann')
plot_confusion(y_true, lstm_pred, 'LSTM', 'lstm')
plot_confusion(y_true, ann_pred,  'ANN',  'ann')
plot_comparison(lstm_acc, ann_acc)
plot_class_distribution()

print("\n✅ All outputs saved to outputs/ folder")
print("\n🎉 Project complete!\n")

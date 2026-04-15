# =============================================================================
# HealthAssistAI — EDA Post-entrenament
# Objectiu: avaluar el model ja entrenat sobre el conjunt de test.
# Cobreix: classification report, matriu de confusió i importància de features.
# Prerequisit: executar primer 'python model/train_model.py'
# =============================================================================

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Carpeta on es guarden els gràfics
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images_post")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Carpeta on es guarda el classification report
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification_report")
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "pkl", "triage_model.pkl")

if not os.path.exists(MODEL_PATH):
    print("ERROR: El model no s'ha trobat a model/pkl/triage_model.pkl")
    print("Executa primer: python model/train_model.py")
    sys.exit(1)

# =============================================================================
# Carregar dataset i reconstruir el mateix conjunt de test que train_model.py
# (mateixa transformació, mateixa llavor aleatòria → X_test / y_test idèntics)
# =============================================================================

ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "csv", "dataset_triage.csv")
df = pd.read_csv(ruta)

ta = df['Tensió arterial'].str.split('/', expand=True).astype(float)
df['TA_sistolica']  = ta[0]
df['TA_diastolica'] = ta[1]
df = df.drop(columns=['Tensió arterial'])

NUM_COLS = ['Edat',
            'TA_sistolica', 'TA_diastolica',
            'Freqüència cardíaca', 'Temperatura',
            'Saturació_oxigen', 'Freqüència_respiratoria']
CAT_COLS = ['Gènere', 'Simptomes principals']
FEATURE_COLS = NUM_COLS + CAT_COLS

X = df[FEATURE_COLS]
y = df['Nivell de triatge']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =============================================================================
# Carregar model i avaluar
# =============================================================================

model = joblib.load(MODEL_PATH)
best_name = joblib.load(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "pkl", "best_model_name.pkl")
)

print(f"Model carregat: {best_name}")
print(f"Conjunt de test: {X_test.shape[0]} registres\n")

y_pred = model.predict(X_test)

TARGET_NAMES = ["N1 Urgència", "N2 Preferent", "N3 Normal", "N4 Lleu"]

print("=== Classification Report ===")
report_str = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
print(report_str)

# Guardar com a text
with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Model: {best_name}\n")
    f.write(f"Conjunt de test: {X_test.shape[0]} registres\n\n")
    f.write(report_str)

# Guardar com a CSV
report_dict = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)
pd.DataFrame(report_dict).T.to_csv(os.path.join(REPORT_DIR, "classification_report.csv"), float_format="%.4f")

print(f"Report guardat a eda/classification_report/")

# --- Matriu de confusió ---
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["N1\nUrgència", "N2\nPreferent", "N3\nNormal", "N4\nLleu"],
    cmap="Blues", ax=ax, colorbar=False
)
plt.title(f"Matriu de confusió — {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "confusio.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Importància de les features (si el model ho suporta) ---
clf = model.named_steps["clf"]
importancia_path = os.path.join(IMAGES_DIR, "importancia_features.png")

if hasattr(clf, "feature_importances_"):
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values()
    colors = ['#c0392b' if imp > 0.15 else 'steelblue' for imp in importances]

    importances.plot(kind='barh', figsize=(9, 5), color=colors, edgecolor='white')
    plt.title(f"Importància de les variables — {best_name} (Gini)")
    plt.xlabel("Importància")
    plt.axvline(x=importances.mean(), color='gray', linestyle='--',
                label=f'Mitjana ({importances.mean():.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(importancia_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gràfic d'importància guardat.")
else:
    # Eliminar imatge antiga si existeix per evitar mostrar dades d'un model diferent
    if os.path.exists(importancia_path):
        os.remove(importancia_path)
    print(f"[Info] {best_name} no exposa feature_importances_; gràfic d'importància no disponible.")

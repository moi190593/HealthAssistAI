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
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

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

NUM_COLS = ['Edat', 'Pes', 'Altura', 'IMC', 'Gravetat_simptoma',
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

TARGET_NAMES = ["N1 Urgència", "N2 Preferent", "N3 Normal", "N4 Programat"]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

# --- Matriu de confusió ---
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["N1\nUrgència", "N2\nPreferent", "N3\nNormal", "N4\nProgramat"],
    cmap="Blues", ax=ax, colorbar=False
)
plt.title(f"Matriu de confusió — {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "confusio.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Importància de les features (si el model ho suporta) ---
clf = model.named_steps["clf"]
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
    plt.savefig(os.path.join(IMAGES_DIR, "importancia_features.png"), dpi=150, bbox_inches='tight')
    plt.close()
else:
    print(f"[Info] {best_name} no exposa feature_importances_; gràfic d'importància omès.")

"""
Compara múltiples models de classificació per predir la prioritat de visita
en un centre d'atenció primària (CAP).
Models avaluats:
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Regressió Logística
  - Arbre de Decisió

Criteris de selecció: F1-macro (RandomizedSearchCV, CV 3-fold) + temps.
Els hiperparàmetres òptims es troben automàticament per cada model.
El millor model es guarda a model/pkl/triage_model.pkl.
Els resultats de la comparació es guarden a model/pkl/comparison_results.pkl.
"""
import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# 1. Carregar i preparar dades
# ---------------------------------------------------------------------------
df = pd.read_csv("data/csv/dataset_triage.csv")

ta = df["Tensió arterial"].str.split("/", expand=True).astype(float)
df["TA_sistolica"]  = ta[0]
df["TA_diastolica"] = ta[1]

FEATURE_COLS = [
    "Edat", "Gènere", "Pes", "Altura", "IMC",
    "Simptomes principals", "Gravetat_simptoma",
    "TA_sistolica", "TA_diastolica",
    "Freqüència cardíaca", "Temperatura",
    "Saturació_oxigen", "Freqüència_respiratoria",
]
TARGET_COL = "Nivell de triatge"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

NUM_COLS = ["Edat", "Pes", "Altura", "IMC", "Gravetat_simptoma",
            "TA_sistolica", "TA_diastolica",
            "Freqüència cardíaca", "Temperatura",
            "Saturació_oxigen", "Freqüència_respiratoria"]
CAT_COLS = ["Gènere", "Simptomes principals"]

LABEL_NAMES = {
    1: "Urgència (Nivell 1)",
    2: "Preferent (Nivell 2)",
    3: "Normal (Nivell 3)",
    4: "Programat (Nivell 4)",
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------------------
# 2. Preprocessador compartit
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
])

# ---------------------------------------------------------------------------
# 3. Definició dels models candidats
# ---------------------------------------------------------------------------
# Classificadors base — els hiperparàmetres s'optimitzen a la secció 4
CANDIDATES = {
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "SVM": SVC(
        class_weight="balanced", probability=True, random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_jobs=-1
    ),
    "Regressió Logística": LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Arbre de Decisió": DecisionTreeClassifier(
        class_weight="balanced", random_state=42
    ),
}

# Espai de cerca d'hiperparàmetres per cada model
PARAM_GRIDS = {
    "Random Forest": {
        "clf__n_estimators":      [100, 200, 300],
        "clf__max_depth":         [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf":  [1, 2],
        "clf__max_features":      ["sqrt", "log2"],
    },
    "SVM": {
        "clf__C":      [0.1, 1, 10],
        "clf__gamma":  ["scale", "auto"],
        "clf__kernel": ["rbf"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
        "clf__weights":     ["uniform", "distance"],
        "clf__metric":      ["euclidean", "manhattan"],
    },
    "Regressió Logística": {
        "clf__C":       [0.01, 0.1, 1, 10, 100],
        "clf__solver":  ["lbfgs", "saga"],
        "clf__penalty": ["l2"],
    },
    "Arbre de Decisió": {
        "clf__max_depth":         [5, 10, 15, 20, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__criterion":         ["gini", "entropy"],
    },
}

# Nombre d'iteracions de cerca per model (equilibri rendiment/temps)
N_ITER = {
    "Random Forest":       10,
    "SVM":                  6,
    "KNN":                 10,
    "Regressió Logística": 10,
    "Arbre de Decisió":    10,
}

# Justificació de cada model (per mostrar a la UI)
MODEL_JUSTIFICATION = {
    "Random Forest": (
        "Ensemble de molts arbres de decisió amb votació majoritària. "
        "Robust al soroll, gestiona bé valors atípics i no requereix escala. "
        "Ofereix importància de features interpretable."
    ),
    "SVM": (
        "Troba l'hiperplà de màxim marge entre classes. "
        "Excel·lent en espais d'alta dimensió. "
        "Requereix escala de dades i pot ser lent en datasets grans."
    ),
    "KNN": (
        "Classifica per similitud amb els k veïns més propers. "
        "Simple i intuïtiu, sense fase d'entrenament real. "
        "Sensible a l'escala i lent en predicció per a datasets grans."
    ),
    "Regressió Logística": (
        "Model lineal probabilístic. Ràpid, interpretable i robust. "
        "Útil com a baseline. Pot no capturar relacions no lineals "
        "presents en dades clíniques."
    ),
    "Arbre de Decisió": (
        "Model basat en regles if/else. Molt interpretable i visualitzable. "
        "Tendeix a l'overfitting sense poda. Útil per explicar decisions "
        "clíniques de forma transparent."
    ),
}

# ---------------------------------------------------------------------------
# 4. Optimització d'hiperparàmetres i comparació (RandomizedSearchCV, CV 3-fold)
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"{'Model':<25} {'Acc CV':>8} {'F1 macro':>10} {'Temps (s)':>10}")
print("=" * 70)

results = []
trained_pipelines = {}

for name, clf in CANDIDATES.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=PARAM_GRIDS[name],
        n_iter=N_ITER[name],
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
        cv=3,
        refit="f1_macro",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_idx = search.best_index_
    f1      = search.cv_results_["mean_test_f1_macro"][best_idx]
    acc     = search.cv_results_["mean_test_accuracy"][best_idx]
    f1_std  = search.cv_results_["std_test_f1_macro"][best_idx]
    acc_std = search.cv_results_["std_test_accuracy"][best_idx]

    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    params_str  = ", ".join(f"{k}={v}" for k, v in best_params.items())

    print(f"{name:<25} {acc:.4f}±{acc_std:.3f}  {f1:.4f}±{f1_std:.3f}  {elapsed:>9.1f}s")
    print(f"  └─ {params_str}")

    results.append({
        "Model": name,
        "Accuracy CV": round(acc, 4),
        "Accuracy std": round(acc_std, 4),
        "F1 macro CV": round(f1, 4),
        "F1 std": round(f1_std, 4),
        "Temps (s)": round(elapsed, 1),
        "Hiperparàmetres": params_str,
    })

    # best_estimator_ ja està ajustat sobre X_train (refit=True)
    trained_pipelines[name] = search.best_estimator_

print("=" * 70)

# ---------------------------------------------------------------------------
# 5. Selecció del millor model (per F1 macro)
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("F1 macro CV", ascending=False)
best_name   = results_df.iloc[0]["Model"]
best_pipe   = trained_pipelines[best_name]

print(f"\n✓ Millor model: {best_name}")
print(f"  F1 macro CV : {results_df.iloc[0]['F1 macro CV']:.4f}")
print(f"  Accuracy CV : {results_df.iloc[0]['Accuracy CV']:.4f}")

# Informe detallat del millor model
y_pred = best_pipe.predict(X_test)
print(f"\n=== Informe de classificació — {best_name} ===")
print(classification_report(
    y_test, y_pred,
    target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
))

# Importància de features (si el model ho suporta)
clf_obj = best_pipe.named_steps["clf"]
if hasattr(clf_obj, "feature_importances_"):
    feat_names = NUM_COLS + CAT_COLS
    importances = clf_obj.feature_importances_
    print("Importància de les variables:")
    for fname, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        print(f"  {fname:<30} {imp:.4f}")

# ---------------------------------------------------------------------------
# 6. Guardar artefactes
# ---------------------------------------------------------------------------
os.makedirs("model/pkl", exist_ok=True)
joblib.dump(best_pipe,    "model/pkl/triage_model.pkl")
joblib.dump(LABEL_NAMES,  "model/pkl/label_names.pkl")
joblib.dump(FEATURE_COLS, "model/pkl/feature_cols.pkl")
joblib.dump(results_df,   "model/pkl/comparison_results.pkl")
joblib.dump(MODEL_JUSTIFICATION, "model/pkl/model_justification.pkl")
joblib.dump(best_name,    "model/pkl/best_model_name.pkl")

# Hiperparàmetres òptims de cada model
best_params_dict = {
    row["Model"]: row["Hiperparàmetres"] for _, row in results_df.iterrows()
}
joblib.dump(best_params_dict, "model/pkl/best_params.pkl")

# Metadades per a la UI
meta = {
    "genere_options":   sorted(df["Gènere"].unique().tolist()),
    "simptomes_options": sorted(df["Simptomes principals"].unique().tolist()),
    "simptomes_nivell1": sorted(df[df["Nivell de triatge"] == 1]["Simptomes principals"].unique().tolist()),
    "simptoma_a_nivell": df.set_index("Simptomes principals")["Gravetat_simptoma"].to_dict(),
    "edat_range":  (int(df["Edat"].min()),   int(df["Edat"].max())),
    "pes_range":   (float(df["Pes"].min()),  float(df["Pes"].max())),
    "altura_range":(float(df["Altura"].min()),float(df["Altura"].max())),
    "fc_range":    (int(df["Freqüència cardíaca"].min()), int(df["Freqüència cardíaca"].max())),
    "temp_range":  (float(df["Temperatura"].min()), float(df["Temperatura"].max())),
    "ta_sis_range":(float(df["TA_sistolica"].min()), float(df["TA_sistolica"].max())),
    "ta_dia_range":(float(df["TA_diastolica"].min()), float(df["TA_diastolica"].max())),
}
joblib.dump(meta, "model/pkl/ui_meta.pkl")
print(f"\nModel guardat a model/pkl/triage_model.pkl  ({best_name})")
print("Comparació guardada a model/pkl/comparison_results.pkl")

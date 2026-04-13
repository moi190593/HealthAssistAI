# HealthAssist AI — Sistema de Triatge Hospitalari amb ML

Projecte de curs d'IA per predir el nivell de triatge hospitalari (Sistema de Manchester, 5 nivells) a partir de símptomes i constants vitals del pacient.

## Estructura del projecte

```
HealthAssistAI/
├── data/
│   ├── generate_dataset.py      # Genera el dataset sintètic d'entrenament
│   └── csv/
│       └── dataset_triage.csv   # Dataset generat (10.000 registres, 91 símptomes)
├── model/
│   ├── train_model.py           # Entrena i compara 6 models, guarda el millor
│   └── pkl/
│       ├── triage_model.pkl     # Pipeline del millor model (Random Forest)
│       ├── label_names.pkl      # Noms dels nivells de triatge
│       ├── feature_cols.pkl     # Llista de features del model
│       ├── comparison_results.pkl
│       ├── model_justification.pkl
│       ├── best_model_name.pkl        ├── best_params.pkl          # Hiperparàmetres òptims de cada model│       └── ui_meta.pkl          # Metadades per a la UI (rangs, opcions)
├── eda/
│   ├── eda_pre_train.py         # EDA del dataset (abans d'entrenar)
│   ├── eda_post_train.py        # Avaluació del model (després d'entrenar)
│   └── images/                  # Gràfics generats per l'EDA
├── app/
│   └── app.py                   # Interfície Streamlit (2 pestanyes)
├── requirements.txt
└── README.md
```

## Instal·lació

```bash
pip install -r requirements.txt
```

## Ús (en ordre)

### 1. Generar el dataset sintètic
```bash
cd HealthAssistAI
python data/generate_dataset.py
```

### 2. Explorar les dades (EDA pre-entrenament)
Analitza el dataset: nuls, duplicats, distribucions, correlacions, outliers i vectorització.
```bash
python eda/eda_pre_train.py
```

### 3. Entrenar els models
```bash
python model/train_model.py
```

### 4. Avaluar el model (EDA post-entrenament)
Genera el classification report, la matriu de confusió i la importància de features.
```bash
python eda/eda_post_train.py
```

### 5. Llançar la interfície
```bash
python -m streamlit run app/app.py
```

## Dataset

Dataset sintètic de 10.000 registres amb lògica clínica coherent:

- **111 símptomes** classificats en 5 nivells de gravetat clínica
- **10 features**: Edat, Gènere, Pes, Altura, IMC, Símptoma principal, TA sistòlica, TA diastòlica, Freqüència cardíaca, Temperatura
- **Distribució**: 5% N1 / 15% N2 / 30% N3 / 35% N4 / 15% N5

## Models

S'avaluen **5 models** amb `RandomizedSearchCV` (CV 3-fold, F1-macro) per trobar els hiperparàmetres òptims de cada un:

| Model                  | Descripció                                      |
|------------------------|-------------------------------------------------|
| Random Forest          | Millor model (F1 ≈ 0.9985). Ensemble d'arbres   |

| SVM                    | Màxim marge, bo en alta dimensió                |
| KNN                    | Classificació per veïns més propers             |
| Regressió Logística    | Model lineal probabilístic, bon baseline        |
| Arbre de Decisió       | Molt interpretable, tendeix a overfitting       |

**Preprocessament**: `StandardScaler` per a numèriques + `OrdinalEncoder` per a categòriques (`ColumnTransformer` + `Pipeline`).

## Nivells del Sistema de Triatge de Manchester

| Nivell | Color  | Nom              | Temps màxim   |
|--------|--------|------------------|---------------|
| 1      | 🔴     | Reanimació       | Immediat      |
| 2      | 🟠     | Emergència       | < 10 min      |
| 3      | 🟡     | Urgent           | < 60 min      |
| 4      | 🟢     | Menys urgent     | < 120 min     |
| 5      | 🔵     | No urgent        | < 240 min     |

> **Avís**: Prototip educatiu. No substitueix el criteri clínic d'un professional sanitari.

# HealthAssist AI — Sistema de Suport al Triatge d'Atenció Primària

Projecte de curs d'IA per predir el nivell de triatge en un centre d'atenció primària (CAP) a partir de símptomes i constants vitals del pacient. Orientat a l'ús per part de professionals sanitaris.

## Estructura del projecte

```
HealthAssistAI/
├── data/
│   ├── generate_dataset.py      # Genera el dataset sintètic d'entrenament
│   └── csv/
│       └── dataset_triage.csv   # Dataset generat (10.000 registres, 74 símptomes)
├── model/
│   ├── train_model.py           # Entrena i compara 5 models, guarda el millor
│   └── pkl/
│       ├── triage_model.pkl     # Pipeline del millor model
│       ├── label_names.pkl      # Noms dels nivells de triatge
│       ├── feature_cols.pkl     # Llista de features del model
│       ├── comparison_results.pkl
│       ├── model_justification.pkl
│       ├── best_model_name.pkl
│       ├── best_params.pkl      # Hiperparàmetres òptims de cada model
│       └── ui_meta.pkl          # Metadades per a la UI (rangs, opcions)
├── eda/
│   ├── eda_pre_train.py         # EDA del dataset (abans d'entrenar)
│   ├── eda_post_train.py        # Avaluació del model (després d'entrenar)
│   ├── images_pre/              # Gràfics generats per l'EDA pre-entrenament
│   ├── images_post/             # Gràfics generats per l'EDA post-entrenament
│   └── classification_report/   # Report en format .txt i .csv
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
Analitza el dataset: nuls, duplicats, distribucions, correlacions, outliers i preprocessament de features.
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

Dataset sintètic de 10.000 registres amb lògica clínica coherent (rangs basats en NEWS2):

- **74 símptomes** classificats en 4 nivells de gravetat clínica
- **9 features**: Edat, Gènere, Símptoma principal, TA sistòlica, TA diastòlica, Freqüència cardíaca, Temperatura, SpO2, Freqüència respiratòria
- **Distribució**: 10% N1 / 20% N2 / 30% N3 / 40% N4

## Models

S'avaluen **5 models** amb `RandomizedSearchCV` (CV 3-fold, F1-macro) per trobar els hiperparàmetres òptims de cada un:

| Model               | Descripció                                          |
|---------------------|-----------------------------------------------------|
| Random Forest       | Ensemble d'arbres, robust al soroll                 |
| SVM                 | Màxim marge, bo en alta dimensió                    |
| KNN                 | Classificació per veïns més propers                 |
| Regressió Logística | Model lineal probabilístic, bon baseline            |
| Arbre de Decisió    | Molt interpretable, tendeix a overfitting           |

**Criteri de selecció**: F1-macro (penalitza per igual els errors en totes les classes, incloses les minoritàries com el Nivell 1).

**Preprocessament**: `StandardScaler` per a numèriques + `OrdinalEncoder` per a categòriques (`ColumnTransformer` + `Pipeline`).

## Nivells de triatge

| Nivell | Color | Nom       | Temps d'atenció       |
|--------|-------|-----------|-----------------------|
| 1      | 🔴    | Urgència  | Immediata (< 15 min)  |
| 2      | 🟠    | Preferent | Urgent (< 30 min)     |
| 3      | 🔵    | Normal    | Preferent (< 60 min)  |
| 4      | 🟢    | Lleu      | No urgent (> 60 min)  |

> **Avís**: Prototip educatiu. No substitueix el criteri clínic d'un professional sanitari.

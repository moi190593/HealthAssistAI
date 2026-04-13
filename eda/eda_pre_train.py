# =============================================================================
# HealthAssistAI — EDA Pre-entrenament
# Objectiu: explorar i validar el dataset ABANS d'entrenar el model.
# Cobreix: càrrega, inspecció, nuls, únics, distribucions, correlacions,
#          scatter plots, equilibri de categories, outliers i vectorització.
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Carpeta on es guarden els gràfics
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# =============================================================================
# Fase 1. Recopilació de les dades
# =============================================================================

ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "csv", "dataset_triage.csv")
df = pd.read_csv(ruta)

# =============================================================================
# Fase 2. Exploració i preparació de les dades
# =============================================================================

# Comprovació que les dades s'han carregat correctament
print(df)

# Informació del dataset
df.info()

# --- Selecció de característiques ---
df = df.drop('ID', axis=1)
df.info()

# --- Nuls ---
print(df.isnull().sum())

# --- Únics ---
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()}")

# Separar tensió arterial en dues columnes numèriques
ta = df['Tensió arterial'].str.split('/', expand=True).astype(float)
df['TA_sistolica']  = ta[0]
df['TA_diastolica'] = ta[1]
df = df.drop(columns=['Tensió arterial'])
print(df.head())

# Mapejar la descripció de triatge a valor numèric per a l'EDA
map_triage = {
    'Nivell 1 – Urgència':   1,
    'Nivell 2 – Preferent':  2,
    'Nivell 3 – Normal':     3,
    'Nivell 4 – Programat':  4,
}
df['Nivell de triatge'] = df['Descripció triatge'].map(map_triage)
df = df.drop(columns=['Descripció triatge'])
print(df.dtypes)

# Resum del dataset net
print(f"Shape: {df.shape}")
print(df.describe().round(2))

# Distribució de la variable objectiu
ax = df['Nivell de triatge'].value_counts().sort_index().plot(
    kind='bar', color=['#c0392b', '#e67e22', '#2980b9', '#27ae60'],
    figsize=(8, 4), edgecolor='white'
)
ax.set_title('Distribució de les prioritats de visita (AP)')
ax.set_xlabel('Nivell de triatge')
ax.set_ylabel('Nombre de pacients')
ax.set_xticklabels(
    ['N1 Urgència', 'N2 Preferent', 'N3 Normal', 'N4 Programat'],
    rotation=20, ha='right'
)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "distribucio_nivells.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Duplicats ---
print("Files duplicades:", df.duplicated().sum())

# --- Distribucions ---
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols_per_row = 3
num_rows = (len(num_cols) + num_cols_per_row - 1) // num_cols_per_row

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols_per_row, figsize=(15, 5 * num_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribució de {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Freqüència')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "distribucions_histogrames.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Correlacions ---
df_numericas = df.select_dtypes(include=["float64", "int64"])
correlation_matrix = df_numericas.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriu de correlació")
plt.savefig(os.path.join(IMAGES_DIR, "correlacio.png"), dpi=150, bbox_inches='tight')
plt.close()

# Variables numèriques vs variable objectiu
numeric_features = ['Edat', 'Pes', 'Altura', 'IMC', 'TA_sistolica', 'TA_diastolica',
                    'Freqüència cardíaca', 'Temperatura']
target = 'Nivell de triatge'

plt.figure(figsize=(20, 18))
for i, feature in enumerate(numeric_features):
    plt.subplot(4, 2, i + 1)
    sns.scatterplot(data=df, x=feature, y=target, alpha=0.3,
                    hue=target, palette='coolwarm', legend=False)
    sns.regplot(data=df, x=feature, y=target, scatter=False, color='red')
    plt.title(f'{feature} vs {target}')
    plt.xlabel(feature)
    plt.ylabel(target)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "scatter_numeriques.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Equilibri de categories ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

df['Gènere'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Distribució de Gènere')
axes[0].set_xlabel('Gènere')
axes[0].set_ylabel('Freqüència')
axes[0].tick_params(axis='x', rotation=0)

df['Simptomes principals'].value_counts().sort_values().plot(
    kind='barh', ax=axes[1], color='salmon', edgecolor='white'
)
axes[1].set_title('Distribució de Símptomes principals')
axes[1].set_xlabel('Freqüència')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "equilibri_categories.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Outliers ---
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols_per_row = 3
num_rows = (len(num_cols) + num_cols_per_row - 1) // num_cols_per_row

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols_per_row, figsize=(15, 5 * num_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot de {col}')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "outliers_boxplots.png"), dpi=150, bbox_inches='tight')
plt.close()

Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

outliers = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).sum()
print("Nombre d'outliers per columna:")
print(outliers)
# Nota: els outliers NO s'eliminen. En context clínic poden ser valors reals i rellevants.

# =============================================================================
# Vectorització
# (s'usa el dataset complet sense eliminar outliers, igual que train_model.py)
# =============================================================================

NUM_COLS = ['Edat', 'Pes', 'Altura', 'IMC', 'Gravetat_simptoma',
            'TA_sistolica', 'TA_diastolica',
            'Freqüència cardíaca', 'Temperatura',
            'Saturació_oxigen', 'Freqüència_respiratoria']
CAT_COLS = ['Gènere', 'Simptomes principals']

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
])

# Reload del dataset original per al split (sense outliers eliminats, = train_model.py)
df_full = pd.read_csv(ruta)
ta_full = df_full['Tensió arterial'].str.split('/', expand=True).astype(float)
df_full['TA_sistolica']  = ta_full[0]
df_full['TA_diastolica'] = ta_full[1]
df_full = df_full.drop(columns=['Tensió arterial', 'ID', 'Descripció triatge'])

X = df_full[NUM_COLS + CAT_COLS]
y = df_full['Nivell de triatge']

X_transformed = preprocessor.fit_transform(X)
print("Shape després del preprocessament:", X_transformed.shape)
print("Features:", NUM_COLS + CAT_COLS)

# Visualització dels valors escalats de les variables numèriques
X_scaled_df = pd.DataFrame(X_transformed[:, :len(NUM_COLS)], columns=NUM_COLS)

n_cols_plot = 3
n_rows_plot = (len(NUM_COLS) + n_cols_plot - 1) // n_cols_plot
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(16, 5 * n_rows_plot))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    sns.histplot(X_scaled_df[col], bins=30, kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'{col} (escalat)')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.suptitle('Distribució de variables numèriques després de StandardScaler', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "features_escalades.png"), dpi=150, bbox_inches='tight')
plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape[0]} registres  |  Test: {X_test.shape[0]} registres")
print(f"Distribució train:\n{y_train.value_counts().sort_index()}")

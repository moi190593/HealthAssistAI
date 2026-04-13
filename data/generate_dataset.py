"""
Genera un dataset sintètic amb lògica clínica per predir la prioritat de visita
en un centre d'atenció primària (CAP).
Les etiquetes s'assignen seguint criteris de triatge d'AP:
combinació de símptoma + constants vitals + edat → nivell (1-4).
"""
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

LABEL_NAMES = {
    1: "Nivell 1 – Urgència",
    2: "Nivell 2 – Preferent",
    3: "Nivell 3 – Normal",
    4: "Nivell 4 – Programat",
}

GENERES = ["Home", "Dona", "Altres"]

# ---------------------------------------------------------------------------
# Símptomes per nivell de triatge d'atenció primària
# ---------------------------------------------------------------------------
SIMPTOMES_PER_NIVELL = {
    1: [  # Urgència — atenció < 2 hores
        "Febre alta amb mal estat general",
        "Dificultat respiratòria lleu",
        "Crisi asmàtica lleu",
        "Dolor abdominal intens",
        "Cefalea intensa sobtada",
        "Dolor toràcic atípic",
        "Vertigen intens incapacitant",
        "Palpitacions sostingudes",
        "Reacció al·lèrgica moderada",
        "Epistaxi persistent",
        "Sospita de pielonefritis",
        "Dolor lumbar agut incapacitant",
        "Hipoglucèmia moderada conscient",
        "Vòmits incoercibles",
        "Dolor ocular agut",
        "Crisi d'ansietat intensa",
    ],
    2: [  # Preferent — atenció < 24 hores
        "Faringitis o amigdalitis aguda",
        "Otitis aguda",
        "Sinusitis aguda",
        "Infecció urinària baixa sense febre",
        "Tos amb expectoració purulenta",
        "Febre moderada sense complicacions",
        "Conjuntivitis aguda",
        "Diarrea aguda amb febre lleu",
        "Erupció cutània aguda",
        "Dolor muscular o articular moderat",
        "Ferida amb risc d'infecció",
        "Dolor de queixal moderat",
        "Lumbàlgia subaguda",
        "Cefalea tensional freqüent",
        "Ansietat o estrès agut",
        "Dolor menstrual intens",
        "Molèsties urinàries lleus",
        "Picada d'insecte amb reacció local",
    ],
    3: [  # Normal — atenció en 48–72 hores
        "Refredat comú",
        "Rinitis al·lèrgica",
        "Tos seca sense febre",
        "Mal de gola lleu",
        "Mal de cap tensional ocasional",
        "Mareig lleu ocasional",
        "Insomni",
        "Pruïja sense causa clara",
        "Molèstia digestiva lleu",
        "Nàusees lleus sense vòmits",
        "Diarrea lleu sense febre",
        "Tendinitis o cervicalgia lleu",
        "Contusió menor",
        "Peu d'atleta",
        "Úlcera bucal",
        "Herpes labial",
        "Molèstia ocular lleu",
        "Cansament general persistent",
        "Ronquera lleu",
        "Dolor intermenstrual lleu",
    ],
    4: [  # Programat — visita planificada (> 72 hores)
        "Renovació de recepta crònica",
        "Control de diabetis",
        "Control d'hipertensió arterial",
        "Seguiment de malaltia cardíaca estable",
        "Revisió analítica rutinària",
        "Sol·licitud de baixa laboral",
        "Vacunació",
        "Revisió ginecològica rutinària",
        "Control de pes",
        "Seguiment de dislipèmia",
        "Seguiment d'hipotiroidisme",
        "Revisió de medicació",
        "Consell anticonceptiu",
        "Sol·licitud de derivació a especialista",
        "Certificat mèdic",
        "Consell nutricional",
        "Revisió pediàtrica rutinària",
        "Control de coagulació crònica",
        "Seguiment de depressió estable",
        "Revisió de ferida cicatritzada",
    ],
}

# Mapa símptoma → nivell de gravetat clínica (per a la feature Gravetat_simptoma)
SIMPTOMA_A_NIVELL = {
    simptoma: nivell
    for nivell, simptomes in SIMPTOMES_PER_NIVELL.items()
    for simptoma in simptomes
}

# Distribució objectiu: 10%, 20%, 30%, 40%
TARGET_DIST = {1: 1000, 2: 2000, 3: 3000, 4: 4000}


def generar_pacient(target_level):
    """Genera les constants vitals i el símptoma coherents amb el nivell objectiu."""
    edat   = int(rng.integers(18, 86))  # Atenció primària: adults (18–85 anys)
    genere = rng.choice(GENERES)
    pes    = round(float(rng.normal(72, 14)), 1)
    pes    = max(40.0, min(180.0, pes))
    altura = round(float(rng.normal(168, 10)), 1)
    altura = max(150.0, min(200.0, altura))
    imc    = round(pes / ((altura / 100) ** 2), 2)

    simptoma = rng.choice(SIMPTOMES_PER_NIVELL[target_level])

    if target_level == 1:  # Urgència — constants alterades però no crítiques
        fc   = int(rng.integers(100, 126))
        temp = round(float(rng.uniform(39.5, 40.5)), 1)
        ta_s = int(rng.choice([rng.integers(90, 101), rng.integers(155, 181)]))
        ta_d = int(rng.choice([rng.integers(55, 66), rng.integers(100, 116)]))
        spo2 = int(rng.integers(93, 97))
        fr   = int(rng.integers(20, 27))

    elif target_level == 2:  # Preferent — constants lleugerament alterades
        fc   = int(rng.integers(85, 106))
        temp = round(float(rng.uniform(38.0, 39.5)), 1)
        ta_s = int(rng.integers(110, 156))
        ta_d = int(rng.integers(65, 101))
        spo2 = int(rng.integers(95, 99))
        fr   = int(rng.integers(16, 21))

    elif target_level == 3:  # Normal — constants dins rang normal
        fc   = int(rng.integers(65, 91))
        temp = round(float(rng.uniform(37.0, 38.1)), 1)
        ta_s = int(rng.integers(105, 141))
        ta_d = int(rng.integers(60, 91))
        spo2 = int(rng.integers(96, 100))
        fr   = int(rng.integers(14, 19))

    else:  # 4 — Programat — constants completament normals
        fc   = int(rng.integers(60, 86))
        temp = round(float(rng.uniform(36.0, 37.5)), 1)
        ta_s = int(rng.integers(100, 141))
        ta_d = int(rng.integers(60, 91))
        spo2 = int(rng.integers(97, 100))
        fr   = int(rng.integers(12, 17))

    return {
        "Edat": edat,
        "Gènere": genere,
        "Pes": pes,
        "Altura": altura,
        "IMC": imc,
        "Simptomes principals": simptoma,
        "Gravetat_simptoma": SIMPTOMA_A_NIVELL[simptoma],
        "Tensió arterial": f"{ta_s}/{ta_d}",
        "Freqüència cardíaca": fc,
        "Temperatura": temp,
        "Saturació_oxigen": spo2,
        "Freqüència_respiratoria": fr,
        "Nivell de triatge": int(target_level),
        "Descripció triatge": LABEL_NAMES[int(target_level)],
    }


records = []
for level, count in TARGET_DIST.items():
    for _ in range(count):
        records.append(generar_pacient(level))

df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "ID", range(1, len(df) + 1))

total_simptomes = sum(len(v) for v in SIMPTOMES_PER_NIVELL.values())
print(f"Dataset generat: {len(df)} registres amb {total_simptomes} símptomes únics")
print(df["Descripció triatge"].value_counts())
print("\nEstadístiques per nivell (temperatura, FC, SpO2 i FR):")
print(df.groupby("Nivell de triatge")[["Temperatura", "Freqüència cardíaca", "Saturació_oxigen", "Freqüència_respiratoria"]].mean().round(2))

df.to_csv("data/csv/dataset_triage.csv", index=False)


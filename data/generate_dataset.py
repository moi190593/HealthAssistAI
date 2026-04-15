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
    4: "Nivell 4 – Lleu",
}

GENERES = ["Home", "Dona", "Altres"]

# ---------------------------------------------------------------------------
# Símptomes per nivell de triatge d'atenció primària
# ---------------------------------------------------------------------------
SIMPTOMES_PER_NIVELL = {
    1: [ 
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
    2: [ 
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
    3: [ 
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
    4: [ 
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

# Distribució objectiu: 10%, 20%, 30%, 40%
TARGET_DIST = {1: 1000, 2: 2000, 3: 3000, 4: 4000}


def generar_pacient(target_level):
    """Genera les constants vitals i el símptoma coherents amb el nivell objectiu."""
    edat   = int(rng.integers(15, 86))
    genere = rng.choice(GENERES)

    simptoma = rng.choice(SIMPTOMES_PER_NIVELL[target_level])

    if target_level == 1:  # Urgència — constants crítiques (NEWS2 alt risc)
        fc   = int(rng.choice([rng.integers(30, 55), rng.integers(121, 161)])) 
        temp = round(float(rng.uniform(38.5, 41.0)), 1)
        ta_s = int(rng.choice([rng.integers(70, 96), rng.integers(171, 211)])) 
        ta_d = int(rng.choice([rng.integers(40, 62), rng.integers(110, 131)]))
        spo2 = int(rng.integers(84, 94))                                   
        fr   = int(rng.integers(24, 36))                                    

    elif target_level == 2:  # Preferent — constants alterades (NEWS2 risc moderat)
        fc   = int(rng.integers(96, 131))                                 
        temp = round(float(rng.uniform(37.2, 39.0)), 1)                      
        ta_s = int(rng.choice([rng.integers(88, 105), rng.integers(151, 181)]))
        ta_d = int(rng.choice([rng.integers(53, 73), rng.integers(98, 115)]))
        spo2 = int(rng.integers(91, 96))                            
        fr   = int(rng.integers(19, 27))                                

    elif target_level == 3:  # Normal — constants lleugerament alterades
        fc   = int(rng.integers(74, 102))                                
        temp = round(float(rng.uniform(36.3, 37.8)), 1)                   
        ta_s = int(rng.integers(98, 158))
        ta_d = int(rng.integers(58, 98))
        spo2 = int(rng.integers(93, 98))                                     
        fr   = int(rng.integers(15, 22))                                     

    else:  # 4 — Lleu — constants normals
        fc   = int(rng.integers(53, 78))                                        
        temp = round(float(rng.uniform(35.5, 36.8)), 1)                        
        ta_s = int(rng.integers(98, 143))
        ta_d = int(rng.integers(58, 93))
        spo2 = int(rng.integers(96, 100))                                       
        fr   = int(rng.integers(11, 17))                                     

    # Soroll gaussià — variabilitat biològica i d'instrument
    fc   = int(np.clip(fc   + rng.normal(0, 4),   25,  200))
    temp = round(float(np.clip(temp + rng.normal(0, 0.3), 34.0, 42.5)), 1)
    spo2 = int(np.clip(spo2 + rng.normal(0, 1.5), 70,  100))
    fr   = int(np.clip(fr   + rng.normal(0, 2),    8,   50))
    ta_s = int(np.clip(ta_s + rng.normal(0, 6),   50,  240))
    ta_d = int(np.clip(ta_d + rng.normal(0, 4),   30,  140))

    return {
        "Edat": edat,
        "Gènere": genere,
        "Simptomes principals": simptoma,
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


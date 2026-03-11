# -*- coding: utf-8 -*-
"""
Configurazione centralizzata dell'analisi.
Modifica questo file per adattare i percorsi e i parametri al tuo caso.
"""

import numpy as np
import os

# ============================================================
# COSTANTI FISICHE
# ============================================================
G = 9.81  # m/s²

# ============================================================
# FILE DI INPUT
# ============================================================
# Accelerogramma di input (3 colonne in g, spazio-separato, senza header)
INPUT_COLB = "COLB.txt"
INPUT_DT   = 0.01        # passo temporale del COLB [s]

# Excel master (unico file con tutti i fogli)
# Fogli attesi: Riepilogo, TimeHistory, IDR_Riepilogo, IDR_TimeHistory
EXCEL_MASTER = "RISULTATI_COMPLETI.xlsx"

# Se hai un file per numero di piani, usa questo dict (modalità multi-file):
# EXCEL_BY_FLOORS = {
#     3:  "RISULTATI_3P.xlsx",
#     5:  "RISULTATI_5P.xlsx",
#     10: "RISULTATI_10P.xlsx",
#     15: "RISULTATI_15P.xlsx",
# }

# Piani da analizzare
FLOORS = [3, 5, 10, 15]

# ============================================================
# PARAMETRI DI ANALISI
# ============================================================
TMAX           = 30.0                         # finestra temporale [s]
PERIODS        = np.linspace(0.01, 4.0, 1000) # periodi spettro [s]
OSC_FREQS      = 1.0 / PERIODS
OSC_DAMPING    = 0.05                          # smorzamento spettrale (5%)
SA_XMAX        = 3.0                           # max periodo nei grafici SA [s]
PSD_FMAX       = 50.0                          # max frequenza PSD [Hz]
WELCH_NPERSEG  = 2048
HUSID_P1       = 0.05                          # percentili Husid
HUSID_P2       = 0.95

# Soglie IDR prestazionali (NTC2018)
IDR_SLO_FRAGILI      = 0.0033
IDR_SLO_DEFORMABILI  = 0.0050

# ============================================================
# CARTELLE DI OUTPUT
# ============================================================
OUTDIR_RIGIDA      = "OUTPUT_BASE_FISSA"
OUTDIR_ISOLAMENTO  = "OUTPUT_ISOLAMENTO"

# Sottocartelle – BASE FISSA
DIR_R_BASE      = os.path.join(OUTDIR_RIGIDA, "01_ACC_BASE")
DIR_R_IDR       = os.path.join(OUTDIR_RIGIDA, "02_IDR")
DIR_R_ACC_TOP   = os.path.join(OUTDIR_RIGIDA, "03_ACC_TESTA")
DIR_R_SA        = os.path.join(OUTDIR_RIGIDA, "04_SPETTRI_PIANO")
DIR_R_ARIAS     = os.path.join(OUTDIR_RIGIDA, "05_ARIAS")
DIR_R_TABLES    = os.path.join(OUTDIR_RIGIDA, "06_TABELLE")

# Sottocartelle – ISOLAMENTO
DIR_I_IDR       = os.path.join(OUTDIR_ISOLAMENTO, "01_IDR")
DIR_I_SA        = os.path.join(OUTDIR_ISOLAMENTO, "02_SPETTRI_PIANO")
DIR_I_CONFRONTO = os.path.join(OUTDIR_ISOLAMENTO, "03_CONFRONTO_PFA_SA")
DIR_I_TABLES    = os.path.join(OUTDIR_ISOLAMENTO, "04_TABELLE")

ALL_OUTPUT_DIRS = [
    DIR_R_BASE, DIR_R_IDR, DIR_R_ACC_TOP, DIR_R_SA, DIR_R_ARIAS, DIR_R_TABLES,
    DIR_I_IDR, DIR_I_SA, DIR_I_CONFRONTO, DIR_I_TABLES,
]

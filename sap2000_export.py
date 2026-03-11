# -*- coding: utf-8 -*-
"""
sap2000_export.py – Estrazione dati SAP2000 → Excel master per SpettriPiano.

Esecuzione (Windows, SAP2000 installato):
    python sap2000_export.py

Output:
    RISULTATI_COMPLETI.xlsx  (formato compatibile con app.py / analisi.py)

Fogli generati:
    Riepilogo      – PGA, SA_max, metadati per ogni nodo/asse/modello
    TimeHistory    – storie di accelerazione (campionate a max 5000 punti)
    IDR_Riepilogo  – IDR max per interpiano (se disponibile nell'analisi)
    IDR_TimeHistory– IDR(t) per interpiano critico

Dipendenze Windows-only:
    comtypes  (pip install comtypes)
    SAP2000   (v20+ con API COM attivata)

Tutte le altre funzioni di calcolo riutilizzano src/compute.py e src/loader.py
per garantire coerenza con il post-processing in app.py.
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rendi importabile il package src/ sia da questa cartella sia dalla radice
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.compute import compute_sa, compute_psd, arias_curve
from src.loader  import axis_standardize, classify_strategy, extract_tis
from src.config  import SPECTRA_PERIODS as PERIODS, DAMPING as OSC_DAMPING


# ============================================================
# CONFIGURAZIONE  (modifica questi valori)
# ============================================================

FILES_LIST = [
    r"C:\Users\giuli\Desktop\Dataset\Modelli\PIANI_3_BASE_RIGIDA.sdb",
    r"C:\Users\giuli\Desktop\Dataset\Modelli\PIANI_5_BASE_RIGIDA.sdb",
    r"C:\Users\giuli\Desktop\Dataset\Modelli\PIANI_10_BASE_RIGIDA.sdb",
    r"C:\Users\giuli\Desktop\Dataset\Modelli\PIANI_15_BASE_RIGIDA.sdb",
]

# Nodi di controllo per numero di piani: {node_id: "etichetta"}
NODES_CONFIG_MAP = {
    3:  {75:    "Spigolo", 3013:  "Centro_Trave", 3918:  "Centro_Solaio"},
    5:  {6165:  "Spigolo", 5407:  "Centro_Trave", 5387:  "Centro_Solaio"},
    10: {12270: "Spigolo", 12264: "Centro_Trave", 12068: "Centro_Solaio"},
    15: {12270: "Spigolo", 12264: "Centro_Trave", 12068: "Centro_Solaio"},
}

CASE_NAME      = "TH"
OUTPUT_DIR     = r"C:\Users\giuli\Desktop\Dataset\OUTPUT_ANALISI_COMPLETA_FISSI"
DT_DEFAULT     = 0.001          # [s] usato se SAP non restituisce il dt
TH_MAX_POINTS  = 5000           # campionamento TH per Excel (non influisce sui calcoli)

# ============================================================
# SETUP CARTELLE E LOG
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
EXCEL_OUT = os.path.join(OUTPUT_DIR, "RISULTATI_COMPLETI.xlsx")
LOG_FILE  = os.path.join(OUTPUT_DIR,
                         f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
# MAPPATURA ASSI SAP2000 → standard HNE/HNN/HNZ
# ============================================================
#   SAP: U1=X  U2=Y  U3=Z  →  HNE  HNN  HNZ
_SAP_AXES = {0: "HNE", 1: "HNN", 2: "HNZ"}


# ============================================================
# INIT EXCEL (crea il file con i fogli vuoti + header)
# ============================================================

_SHEETS_SCHEMA = {
    "Riepilogo": [
        "Modello", "N_Piani", "Tipo_Base", "Nodo_ID", "Nodo_Tag", "Asse",
        "DT_s", "Num_Steps", "Durata_s", "Convergenza",
        "PGA_m/s2", "SA_Max_m/s2", "Periodo_Max_s", "Ia_Total",
    ],
    "TimeHistory": ["Modello", "N_Piani", "Nodo_ID", "Asse", "Tempo_s", "Acc_m/s2"],
    "IDR_Riepilogo":   ["N_Piani", "Modello", "Tipo_Base", "Asse",
                        "Story_i", "IDR_Max", "Nodo_Basso", "Nodo_Alto"],
    "IDR_TimeHistory": ["N_Piani", "Modello", "Tipo_Base", "Asse",
                        "Story_i", "Tempo_s", "IDR"],
}


def init_excel() -> None:
    """Ricrea il file Excel con i fogli vuoti e gli header corretti."""
    if os.path.exists(EXCEL_OUT):
        os.remove(EXCEL_OUT)
    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
        for sheet, cols in _SHEETS_SCHEMA.items():
            pd.DataFrame(columns=cols).to_excel(writer, sheet_name=sheet, index=False)
    _log(f"Excel inizializzato: {EXCEL_OUT}")


def _append_rows(sheet: str, df: pd.DataFrame) -> None:
    """Appende df al foglio sheet senza riscrivere l'header."""
    if df.empty:
        return
    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl",
                        mode="a", if_sheet_exists="overlay") as writer:
        existing = pd.read_excel(EXCEL_OUT, sheet_name=sheet)
        start    = len(existing) + 1      # +1 perché la riga 0 è l'header
        df.to_excel(writer, sheet_name=sheet,
                    index=False, header=False, startrow=start)


# ============================================================
# ESTRAZIONE DATI DA UN NODO
# ============================================================

def _extract_node(model_name: str, n_piani: int, tipo_base: str,
                  node_id: int, node_tag: str,
                  sap_model, dt: float) -> dict:
    """
    Estrae accelerazioni assolute dal nodo, calcola SA/PSD/Arias e
    restituisce dict con i dati pronti per Excel.

    Returns None se SAP non ha dati per questo nodo.
    """
    res = sap_model.Results.JointAccAbs(str(node_id), 0)
    # API SAP2000: res = (ret, Obj, Elm, Case, StepType, StepNum, U1, U2, U3, ...)
    if not res or len(res) < 9 or res[0] != 0:
        return None

    u_raw = {ax: np.array(res[6 + i], dtype=float)
             for i, ax in _SAP_AXES.items()}
    n_steps = len(next(iter(u_raw.values())))
    if n_steps < 50:
        return None

    t = np.arange(n_steps, dtype=float) * dt

    rows_sum, rows_th = [], []

    for ax_label, acc in u_raw.items():
        acc = np.nan_to_num(acc)

        # --- calcoli (riusa src/compute.py) ---
        sa     = compute_sa(t, acc)
        ia     = arias_curve(acc, t)
        pga    = float(np.max(np.abs(acc)))
        sa_max = float(np.max(sa))
        t_max  = float(PERIODS[int(np.argmax(sa))])

        rows_sum.append({
            "Modello": model_name, "N_Piani": n_piani, "Tipo_Base": tipo_base,
            "Nodo_ID": node_id, "Nodo_Tag": node_tag, "Asse": ax_label,
            "DT_s": dt, "Num_Steps": n_steps, "Durata_s": round(t[-1], 3),
            "Convergenza": "OK",
            "PGA_m/s2": round(pga, 5),
            "SA_Max_m/s2": round(sa_max, 5),
            "Periodo_Max_s": round(t_max, 4),
            "Ia_Total": round(float(ia[-1]), 6),
        })

        # campiona TH per Excel (risparmia spazio, non influisce sui calcoli)
        step = max(1, n_steps // TH_MAX_POINTS)
        rows_th.append(pd.DataFrame({
            "Modello": model_name, "N_Piani": n_piani,
            "Nodo_ID": node_id,   "Asse": ax_label,
            "Tempo_s": t[::step], "Acc_m/s2": acc[::step],
        }))

    return {
        "summary": rows_sum,
        "th":      pd.concat(rows_th, ignore_index=True),
    }


# ============================================================
# AVVIO SAP2000 VIA COM
# ============================================================

def _start_sap2000():
    """Avvia SAP2000 e restituisce (mySapObject, SapModel)."""
    try:
        import comtypes.client
        import comtypes.gen.SAP2000v1 as sap_api
    except ImportError as e:
        raise RuntimeError("comtypes non trovato. Installa con: pip install comtypes") from e

    # Chiude eventuali istanze precedenti
    os.system("taskkill /F /IM SAP2000.exe >nul 2>&1")
    time.sleep(2)

    helper = comtypes.client.CreateObject("SAP2000v1.Helper")
    helper = helper.QueryInterface(sap_api.cHelper)
    sap    = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
    sap.ApplicationStart()
    time.sleep(5)
    return sap, sap.SapModel


def _get_dt(sap_model, case_name: str) -> float:
    """Tenta di leggere il dt del caso time-history. Ritorna DT_DEFAULT se fallisce."""
    try:
        ret = sap_model.LoadCases.ModHistLinear.GetTimeStep(case_name)
        if ret[0] == 0:
            return float(ret[2])
    except Exception:
        pass
    return DT_DEFAULT


def _n_piani_from_name(model_name: str) -> int:
    for n in (15, 10, 5, 3):
        if f"PIANI_{n}" in model_name.upper():
            return n
    return 0


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    _log("=== AVVIO ESTRAZIONE SAP2000 ===")
    init_excel()

    sap, sap_model = _start_sap2000()

    ok, failed = 0, []

    for file_path in FILES_LIST:
        model_name = os.path.splitext(os.path.basename(file_path))[0]

        if not os.path.exists(file_path):
            _log(f"⚠️  File mancante: {file_path}")
            failed.append(model_name)
            continue

        _log(f"\n--- {model_name} ---")
        n_piani   = _n_piani_from_name(model_name)
        tipo_base = classify_strategy("", model_name)

        try:
            sap_model.File.OpenFile(file_path)
            sap_model.Results.Setup.DeselectAllCasesAndCombosForOutput()
            sap_model.Results.Setup.SetCaseSelectedForOutput(CASE_NAME)
            sap_model.Results.Setup.SetOptionModalHist(2)

            dt = _get_dt(sap_model, CASE_NAME)
            _log(f"   dt={dt*1000:.2f} ms | {n_piani} piani | {tipo_base}")

            nodes = NODES_CONFIG_MAP.get(n_piani, {})
            buf_sum, buf_th = [], []

            for node_id, node_tag in nodes.items():
                _log(f"   Nodo {node_tag} ({node_id})...", )
                result = _extract_node(model_name, n_piani, tipo_base,
                                       node_id, node_tag, sap_model, dt)
                if result is None:
                    _log("     NO DATA")
                    continue
                buf_sum.extend(result["summary"])
                buf_th.append(result["th"])
                _log(f"     OK ({len(result['th'])} righe TH)")

            if buf_sum:
                _append_rows("Riepilogo",   pd.DataFrame(buf_sum))
                _append_rows("TimeHistory", pd.concat(buf_th, ignore_index=True))
                ok += 1

        except Exception as e:
            _log(f"   ❌ Errore: {e}")
            failed.append(model_name)

    _log(f"\n=== FINE: OK={ok} | Falliti={failed} ===")
    _log(f"Excel salvato in: {EXCEL_OUT}")

    try:
        sap.ApplicationExit(False)
    except Exception:
        pass


if __name__ == "__main__":
    main()

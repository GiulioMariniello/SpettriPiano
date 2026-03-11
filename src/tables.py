# -*- coding: utf-8 -*-
"""
Generazione tabelle di risultati (Excel editabile).

Funzioni esportate:
    build_results_table   → PFA_Max, SA_Max per ogni modello/nodo/asse
    build_idr_table       → IDR_Max per ogni modello/asse/piano
    compute_reductions    → riduzioni % rispetto a base fissa
    save_tables           → salva tutto in un unico Excel multi-foglio
"""

import numpy as np
import pandas as pd

from .config import IDR_SLO_FRAGILI, IDR_SLO_DEFORMABILI, TMAX
from .loader import get_th, classify_strategy, extract_tis
from .compute import compute_sa, arias_curve, husid_metrics, PERIODS


# ============================================================
# 1. TABELLA PFA / SA
# ============================================================

def build_results_table(df_r, df_th) -> pd.DataFrame:
    """
    Per ogni combinazione (N_Piani, Nodo_ID, Nodo_Tag, Asse, Modello):
      - PFA_Max  : max |acc(t)| su 0-TMAX [m/s²]
      - SA_Max   : max SA spettro 5% (solo HNE/HNN) [m/s²]
      - Ia_Total : intensità di Arias totale [m/s]
      - D5_95    : durata significativa Husid [s]
    """
    sa_cache = {}
    records  = []

    cols_key = ["N_Piani", "Nodo_ID", "Nodo_Tag", "Asse", "Modello", "Categoria", "Tis"]
    groups   = df_r[[c for c in cols_key if c in df_r.columns]].drop_duplicates()

    for _, g in groups.iterrows():
        nP       = int(g.get("N_Piani", 0))
        nodo_id  = g["Nodo_ID"]
        nodo_tag = g.get("Nodo_Tag", "")
        asse     = g["Asse"]
        modello  = g["Modello"]
        cat      = g.get("Categoria", "")
        tis      = float(g.get("Tis", 0.0))

        t, a = get_th(df_th, modello, nodo_id, asse)
        if t is None:
            continue

        pfa_max = float(np.max(np.abs(a))) if len(a) else 0.0

        sa_max = 0.0
        if asse != "HNZ" and len(a) >= 2:
            sa  = compute_sa(t, a, sa_cache, (modello, str(nodo_id), asse))
            sa_max = float(np.max(sa)) if len(sa) else 0.0

        Ia_arr = arias_curve(a, t)
        m      = husid_metrics(Ia_arr, t)

        records.append({
            "N_Piani":  nP,
            "Nodo_ID":  nodo_id,
            "Nodo_Tag": nodo_tag,
            "Asse":     asse,
            "Modello":  modello,
            "Categoria": cat,
            "Tis":      tis,
            "PFA_Max":  round(pfa_max, 5),
            "SA_Max":   round(sa_max,  5) if asse != "HNZ" else np.nan,
            "Ia_Total": round(float(m["Ia_total"]), 6),
            "D5_95_s":  round(float(m["D5_95"]),    3),
        })

    return pd.DataFrame(records)


# ============================================================
# 2. TABELLA IDR
# ============================================================

def build_idr_table(df_idr_sum) -> pd.DataFrame:
    """
    Tabella IDR massimo con confronto soglie NTC2018.

    Colonne di input attese in df_idr_sum:
        N_Piani, Modello, Asse, Story_i, IDR_Max

    Colonne output:
        N_Piani, Modello, Asse, Story_critico, IDR_Max,
        Margine_SLO_Fragili, Margine_SLO_Deformabili,
        OK_Fragili, OK_Deformabili
    """
    if df_idr_sum.empty:
        return pd.DataFrame()

    records = []
    cols    = ["N_Piani", "Modello", "Asse"]
    groups  = df_idr_sum[[c for c in cols if c in df_idr_sum.columns]].drop_duplicates()

    for _, g in groups.iterrows():
        sub = df_idr_sum.copy()
        for col in cols:
            if col in g.index and col in sub.columns:
                sub = sub[sub[col] == g[col]]

        if sub.empty:
            continue

        crit = sub.loc[sub["IDR_Max"].astype(float).idxmax()]
        idr  = float(crit["IDR_Max"])

        records.append({
            "N_Piani":              int(g.get("N_Piani", 0)),
            "Modello":              g.get("Modello", ""),
            "Asse":                 g.get("Asse", ""),
            "Story_critico":        int(crit.get("Story_i", 0)),
            "IDR_Max":              round(idr, 6),
            "Soglia_Fragili":       IDR_SLO_FRAGILI,
            "Soglia_Deformabili":   IDR_SLO_DEFORMABILI,
            "Margine_Fragili":      round(IDR_SLO_FRAGILI - idr, 6),
            "Margine_Deformabili":  round(IDR_SLO_DEFORMABILI - idr, 6),
            "OK_Fragili":           "SI" if idr <= IDR_SLO_FRAGILI else "NO",
            "OK_Deformabili":       "SI" if idr <= IDR_SLO_DEFORMABILI else "NO",
        })

    return pd.DataFrame(records)


# ============================================================
# 3. RIDUZIONI % rispetto a BASE FISSA
# ============================================================

def compute_reductions(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola riduzioni percentuali PFA e SA rispetto al modello RIGIDA
    per ogni (N_Piani, Nodo_ID, Asse).

    Aggiunge una colonna 'Strategia' con etichetta leggibile.

    Parameters
    ----------
    df_res : output di build_results_table()
    """
    records = []

    for (nP, nodo_id, asse), group in df_res.groupby(["N_Piani", "Nodo_ID", "Asse"]):
        ref = group[group["Categoria"] == "RIGIDA"]
        if ref.empty:
            continue

        ref_pfa = float(ref.iloc[0]["PFA_Max"])
        ref_sa  = float(ref.iloc[0]["SA_Max"]) if asse != "HNZ" else np.nan

        for _, row in group.iterrows():
            if row["Categoria"] == "RIGIDA":
                rid_pfa = 0.0
                rid_sa  = 0.0
            else:
                rid_pfa = ((ref_pfa - float(row["PFA_Max"])) / ref_pfa * 100.0
                           if ref_pfa > 0 else np.nan)
                rid_sa  = ((ref_sa - float(row["SA_Max"])) / ref_sa * 100.0
                           if (not np.isnan(ref_sa) and ref_sa > 0) else np.nan)

            tis_label = f"{row['Tis']:.1f}s" if not np.isnan(row["Tis"]) else ""
            if row["Categoria"] == "RIGIDA":
                strategia = "Base Fissa"
            elif row["Categoria"] == "SLITTE":
                strategia = f"Slitte T={tis_label}"
            elif row["Categoria"] == "ISOLATA":
                strategia = f"Isolato T={tis_label}"
            else:
                strategia = row["Modello"]

            records.append({
                "N_Piani":       int(nP),
                "Nodo_ID":       nodo_id,
                "Nodo_Tag":      row.get("Nodo_Tag", ""),
                "Asse":          asse,
                "Categoria":     row["Categoria"],
                "Strategia":     strategia,
                "Tis":           float(row["Tis"]),
                "Modello":       row["Modello"],
                "PFA_Max":       float(row["PFA_Max"]),
                "SA_Max":        float(row["SA_Max"]) if asse != "HNZ" else np.nan,
                "Ia_Total":      float(row.get("Ia_Total", np.nan)),
                "D5_95_s":       float(row.get("D5_95_s", np.nan)),
                "Rid_PFA_pct":   round(float(rid_pfa), 2) if not np.isnan(rid_pfa) else np.nan,
                "Rid_SA_pct":    round(float(rid_sa),  2) if not np.isnan(rid_sa)  else np.nan,
            })

    return pd.DataFrame(records)


# ============================================================
# 4. SALVATAGGIO EXCEL MULTI-FOGLIO
# ============================================================

def save_tables(path: str,
                df_risultati: pd.DataFrame,
                df_idr: pd.DataFrame,
                df_riduzioni: pd.DataFrame) -> None:
    """
    Salva le tabelle in un unico file Excel (.xlsx) con più fogli:
        - Risultati    : PFA, SA, Arias per ogni nodo/asse/modello
        - IDR          : IDR max con verifica NTC2018
        - Riduzioni    : % riduzione PFA e SA rispetto a base fissa

    Il file è direttamente editabile in Excel/LibreOffice.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_risultati.to_excel(writer,  sheet_name="Risultati",  index=False)
        df_idr.to_excel(writer,        sheet_name="IDR",         index=False)
        df_riduzioni.to_excel(writer,  sheet_name="Riduzioni",   index=False)

        # formattazione colonne larghezza auto
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = max(
                    (len(str(c.value)) for c in col_cells if c.value is not None),
                    default=10
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)

    print(f"  → Tabelle salvate in: {path}")

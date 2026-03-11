# -*- coding: utf-8 -*-
"""
Caricamento dati: COLB.txt e fogli Excel.
"""

import re
import numpy as np
import pandas as pd

from .config import G, TMAX, INPUT_DT


# ============================================================
# UTILS GENERALI
# ============================================================

def _strip_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def axis_standardize(series: pd.Series) -> pd.Series:
    """Converte U1→HNE, U2→HNN, U3→HNZ."""
    return series.replace({"U1": "HNE", "U2": "HNN", "U3": "HNZ"})


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s))


def truncate_by_time(t: np.ndarray, x: np.ndarray, tmax: float):
    if len(t) == 0:
        return t, x
    mask = t <= tmax
    return t[mask], x[mask]


def extract_tis(model_name: str) -> float:
    """Estrae il periodo target dall'etichetta del modello (es. 'ISO_2.0s' → 2.0)."""
    m = re.search(r"(\d+\.\d+)", str(model_name))
    return float(m.group(1)) if m else 0.0


def classify_strategy(tipo_base: str, modello: str) -> str:
    """Classifica il modello in RIGIDA / SLITTE / ISOLATA / ALTRO."""
    tb  = (tipo_base or "").upper()
    mn  = (modello   or "").upper()
    txt = tb if tb not in ("", "NAN", "NONE") else mn

    if any(k in txt for k in ("RIGIDA", "FISSA", "FIXED")):
        return "RIGIDA"
    if any(k in txt for k in ("SLITT", "FRICTION")):
        return "SLITTE"
    if any(k in txt for k in ("ISOL", "ELAST")):
        return "ISOLATA"
    return "ALTRO"


# Colore per categoria – usato sia per stili plot che per scatter/bar chart
CAT_COLOR = {"RIGIDA": "black", "SLITTE": "#D55E00", "ISOLATA": "#0072B2", "ALTRO": "gray"}


def style_for_strategy(cat: str, tis: float) -> dict:
    """Restituisce stile matplotlib in base alla categoria."""
    color = CAT_COLOR.get(cat, "gray")
    if cat == "RIGIDA":
        return dict(color=color, lw=2.8, zorder=10, label="Base Fissa")
    if cat == "SLITTE":
        return dict(color=color, lw=1.8, zorder=5,  label=f"Slitte T={tis:.1f}s")
    if cat == "ISOLATA":
        return dict(color=color, lw=1.8, zorder=5,  label=f"Isolato T={tis:.1f}s")
    return dict(color=color, lw=1.2, zorder=1, label="Altro")


def format_strategia(categoria: str, tis: float) -> str:
    """Etichetta leggibile della strategia (usata in tabelle e plot)."""
    tis_s = f"T={tis:.1f}s" if not (tis != tis) else ""   # NaN-safe
    if categoria == "RIGIDA":
        return "Base Fissa"
    if categoria == "SLITTE":
        return f"Slitte {tis_s}"
    if categoria == "ISOLATA":
        return f"Isolato {tis_s}"
    return categoria


def filter_idr_rigida(df_idr_sum: pd.DataFrame,
                       df_idr_th:  pd.DataFrame) -> tuple:
    """
    Filtra i DataFrame IDR tenendo solo i record base fissa (RIGIDA).
    Ritorna nuove copie filtrate senza modificare gli originali.
    """
    def _filt(df):
        if df.empty or "Tipo_Base" not in df.columns:
            return df.copy()
        mask = df["Tipo_Base"].astype(str).str.upper().str.contains("RIG", na=False)
        return df.loc[mask].copy()

    return _filt(df_idr_sum), _filt(df_idr_th)


def fmt_ax(ax, xlabel: str, ylabel: str,
           xlim=None, ylim=None, title: str = None,
           legend: bool = True) -> None:
    """Formattazione assi matplotlib condivisa tra tutti i moduli plot."""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(True, ls=":", alpha=0.55)
    if legend:
        ax.legend(fontsize=9, loc="best", framealpha=0.9)


# ============================================================
# INPUT GROUND MOTION  (COLB.txt)
# ============================================================

def read_colb(path: str, dt: float = INPUT_DT) -> dict:
    """
    Legge COLB.txt (3 colonne in g, no header).
    Restituisce dict con chiavi HNE/HNN/HNZ, ciascuno con:
        {'t': ndarray, 'acc': ndarray [m/s²], 'pga': float}
    """
    import os
    if not os.path.exists(path):
        print(f"⚠️  COLB.txt non trovato: {path}")
        return {}

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 3:
        print("⚠️  COLB.txt: meno di 3 colonne.")
        return {}

    n  = len(df)
    t  = np.arange(n) * dt
    t, _ = truncate_by_time(t, t, TMAX)

    result = {}
    for i, ax in enumerate(("HNE", "HNN", "HNZ")):
        a = df.iloc[:len(t), i].to_numpy(dtype=float) * G
        result[ax] = {
            "t":   t,
            "acc": a,
            "pga": float(np.max(np.abs(a))) if len(a) else 0.0,
        }
    return result


# ============================================================
# EXCEL MASTER
# ============================================================

_REQUIRED_SHEETS  = {"Riepilogo", "TimeHistory"}
_OPTIONAL_SHEETS  = {"IDR_Riepilogo", "IDR_TimeHistory"}


def load_excel(path: str) -> tuple:
    """
    Carica il file Excel master e restituisce la tupla:
        (df_riepilogo, df_th, df_idr_sum, df_idr_th)

    Colonne attese in Riepilogo:
        N_Piani, Modello, Tipo_Base, Nodo_ID, Nodo_Tag, Asse, PGA_m/s2

    Colonne attese in TimeHistory:
        Modello, Nodo_ID, Asse, Tempo_s, Acc_m/s2

    Colonne attese in IDR_Riepilogo:
        N_Piani, Modello, Tipo_Base, Asse, Story_i, IDR_Max

    Colonne attese in IDR_TimeHistory:
        N_Piani, Modello, Tipo_Base, Asse, Story_i, Tempo_s, IDR
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel non trovato: {path}")

    xls     = pd.ExcelFile(path)
    missing = _REQUIRED_SHEETS - set(xls.sheet_names)
    if missing:
        raise ValueError(f"Fogli mancanti nell'Excel: {missing}")

    df_r       = pd.read_excel(xls, sheet_name="Riepilogo")
    df_th      = pd.read_excel(xls, sheet_name="TimeHistory")
    df_idr_sum = (pd.read_excel(xls, sheet_name="IDR_Riepilogo")
                  if "IDR_Riepilogo"   in xls.sheet_names else pd.DataFrame())
    df_idr_th  = (pd.read_excel(xls, sheet_name="IDR_TimeHistory")
                  if "IDR_TimeHistory" in xls.sheet_names else pd.DataFrame())

    # strip spazi
    df_r       = _strip_cols(df_r,       ["Modello", "Asse", "Tipo_Base", "Nodo_Tag", "Nodo_ID"])
    df_th      = _strip_cols(df_th,      ["Modello", "Asse", "Nodo_ID"])
    df_idr_sum = _strip_cols(df_idr_sum, ["Modello", "Tipo_Base", "Asse"])
    df_idr_th  = _strip_cols(df_idr_th,  ["Modello", "Tipo_Base", "Asse"])

    # standardizza assi
    for df in (df_r, df_th, df_idr_sum, df_idr_th):
        if "Asse" in df.columns:
            df["Asse"] = axis_standardize(df["Asse"])

    # colonne derivate su df_r
    if "Tipo_Base" not in df_r.columns:
        df_r["Tipo_Base"] = ""
    df_r["Categoria"] = df_r.apply(
        lambda r: classify_strategy(r.get("Tipo_Base", ""), r.get("Modello", "")), axis=1
    )
    df_r["Tis"] = df_r["Modello"].apply(extract_tis)

    # propaga N_Piani a df_th se mancante
    if "N_Piani" not in df_th.columns and "N_Piani" in df_r.columns:
        map_np = df_r[["Modello", "N_Piani"]].drop_duplicates()
        df_th  = df_th.merge(map_np, on="Modello", how="left")

    return df_r, df_th, df_idr_sum, df_idr_th


def get_th(df_th: pd.DataFrame, modello: str, nodo_id, asse: str):
    """
    Estrae time history (t, a) da df_th per un dato modello/nodo/asse.
    Restituisce (None, None) se non trovata o troppo corta.
    """
    sub = df_th[
        (df_th["Modello"] == modello) &
        (df_th["Nodo_ID"] == str(nodo_id)) &
        (df_th["Asse"]    == asse)
    ]
    # fallback senza cast a str
    if sub.empty:
        sub = df_th[
            (df_th["Modello"] == modello) &
            (df_th["Nodo_ID"] == nodo_id) &
            (df_th["Asse"]    == asse)
        ]
    if sub.empty or len(sub) < 2:
        return None, None

    t = sub["Tempo_s"].to_numpy(dtype=float)
    a = sub["Acc_m/s2"].to_numpy(dtype=float)
    t, a = truncate_by_time(t, a, TMAX)
    return t, a


def get_idr_th(df_idr_th: pd.DataFrame, nP: int, modello: str, asse: str, story_i: int):
    """
    Estrae IDR(t) per un dato interpiano critico.
    """
    sub = df_idr_th[
        (df_idr_th["N_Piani"] == nP) &
        (df_idr_th["Modello"] == modello) &
        (df_idr_th["Asse"]    == asse) &
        (df_idr_th["Story_i"] == story_i)
    ]
    if sub.empty or len(sub) < 2:
        return None, None

    t   = sub["Tempo_s"].to_numpy(dtype=float)
    idr = sub["IDR"].to_numpy(dtype=float)
    t, idr = truncate_by_time(t, idr, TMAX)
    return t, idr

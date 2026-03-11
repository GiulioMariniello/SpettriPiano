# -*- coding: utf-8 -*-
"""
Funzioni di calcolo: spettri SA, PSD Welch, Arias, Husid.
"""

import numpy as np
from scipy.signal import welch
from scipy.integrate import cumulative_trapezoid

from .config import G, PERIODS, OSC_FREQS, OSC_DAMPING, WELCH_NPERSEG, PSD_FMAX, HUSID_P1, HUSID_P2

try:
    import pyrotd
except ImportError as e:
    raise ImportError("pyrotd non trovato. Installa con: pip install pyrotd") from e


# ============================================================
# SPETTRO DI RISPOSTA SA
# ============================================================

def compute_sa(t: np.ndarray, a: np.ndarray,
               cache: dict | None = None, key=None) -> np.ndarray:
    """
    Spettro di accelerazione (SA) con smorzamento OSC_DAMPING.

    Parameters
    ----------
    t, a : ndarray  – tempo [s] e accelerazione [m/s²]
    cache : dict    – cache opzionale per evitare ricalcoli (key deve essere fornita)
    key           – chiave hashable per la cache

    Returns
    -------
    sa : ndarray shape (len(PERIODS),) in m/s²
    """
    if cache is not None and key is not None and key in cache:
        return cache[key]

    if len(t) < 2:
        sa = np.zeros_like(PERIODS)
    else:
        dt   = float(np.mean(np.diff(t)))
        spec = pyrotd.calc_spec_accels(dt, a, OSC_FREQS, OSC_DAMPING)
        sa   = np.asarray(spec.spec_accel, dtype=float)

    if cache is not None and key is not None:
        cache[key] = sa
    return sa


# ============================================================
# PSD WELCH
# ============================================================

def compute_psd(t: np.ndarray, a: np.ndarray) -> tuple:
    """
    Densità spettrale di potenza con metodo di Welch.

    Returns
    -------
    f : ndarray  – frequenze [Hz]  (solo f <= PSD_FMAX)
    Pxx : ndarray – PSD [(m/s²)²/Hz]
    """
    if len(t) < 4:
        return np.array([]), np.array([])

    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt if dt > 0 else 0.0
    if fs <= 0:
        return np.array([]), np.array([])

    f, Pxx = welch(a, fs=fs, nperseg=min(WELCH_NPERSEG, len(a)))
    mask   = f <= PSD_FMAX
    return f[mask], Pxx[mask]


# ============================================================
# INTENSITÀ DI ARIAS
# ============================================================

def arias_curve(acc: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Ia(t) = π/(2g) ∫₀ᵗ a²(τ) dτ   [m/s]
    """
    if len(acc) < 2:
        return np.zeros_like(t)
    integ = cumulative_trapezoid(acc**2, t, initial=0.0)
    return (np.pi / (2.0 * G)) * integ


# ============================================================
# HUSID
# ============================================================

def husid_metrics(ia_curve: np.ndarray, t: np.ndarray,
                  p1: float = HUSID_P1, p2: float = HUSID_P2) -> dict:
    """
    Calcola curva di Husid normalizzata e metriche temporali.

    Returns
    -------
    dict con chiavi:
        husid      – ndarray Ia(t)/Ia_tot
        Ia_total   – intensità di Arias totale [m/s]
        t5, t95    – istanti al p1% e p2% dell'energia [s]
        D5_95      – durata significativa [s]
    """
    Ia_tot = float(ia_curve[-1]) if len(ia_curve) else 0.0
    if Ia_tot <= 0:
        return dict(husid=None, Ia_total=Ia_tot, t5=np.nan, t95=np.nan, D5_95=np.nan)

    h = ia_curve / Ia_tot

    def _time_at(p):
        idx = np.searchsorted(h, p, side="left")
        if idx <= 0:
            return float(t[0])
        if idx >= len(t):
            return float(t[-1])
        t0, t1v = float(t[idx - 1]), float(t[idx])
        h0, h1  = float(h[idx - 1]), float(h[idx])
        if h1 == h0:
            return t1v
        return t0 + (p - h0) * (t1v - t0) / (h1 - h0)

    tp1 = _time_at(p1)
    tp2 = _time_at(p2)
    return dict(husid=h, Ia_total=Ia_tot, t5=tp1, t95=tp2, D5_95=tp2 - tp1)


# ============================================================
# UTILITY: max globale SA per scala Y uniforme
# ============================================================

def global_sa_ymax(df_r, df_th, get_th_fn, axes=("HNE", "HNN")) -> float:
    """
    Calcola il massimo globale di SA su tutti i modelli/nodi/assi indicati,
    da usare come ylim coerente su tutti i grafici.
    """
    sa_max = 0.0
    cache  = {}
    cols   = ["Modello", "Nodo_ID", "Asse"]
    groups = df_r[df_r["Asse"].isin(axes)][cols].drop_duplicates()

    for _, g in groups.iterrows():
        t, a = get_th_fn(df_th, g["Modello"], g["Nodo_ID"], g["Asse"])
        if t is None:
            continue
        sa     = compute_sa(t, a, cache, (g["Modello"], str(g["Nodo_ID"]), g["Asse"]))
        sa_max = max(sa_max, float(np.max(sa)))

    return 1.15 * sa_max if sa_max > 0 else 1.0


def global_acc_ymax(df_r, df_th, get_th_fn, pga_input: dict) -> float:
    """
    Massimo globale delle accelerazioni (output + PGA input) per scala Y uniforme TH.
    """
    acc_max = max((v["pga"] for v in pga_input.values()), default=0.0)
    cols    = ["Modello", "Nodo_ID", "Asse"]

    for _, g in df_r[cols].drop_duplicates().iterrows():
        t, a = get_th_fn(df_th, g["Modello"], g["Nodo_ID"], g["Asse"])
        if t is None:
            continue
        acc_max = max(acc_max, float(np.max(np.abs(a))))

    return 1.15 * acc_max if acc_max > 0 else 1.0

# -*- coding: utf-8 -*-
"""
Grafici – MODALITÀ BASE FISSA (RIGIDA).

Funzioni esportate:
    plot_acc_base          → storie di accelerazione alla base (3 assi)
    plot_idr_crit          → IDR(t) all'interpiano massimo per modello e asse
    plot_acc_top           → storie di acc ai 3 punti di controllo in testa
    plot_spettri_piano     → spettri di piano in testa edificio
    plot_arias             → intensità di Arias (+ Husid)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import TMAX, SA_XMAX, PERIODS, G
from .loader import safe_name, get_th, get_idr_th, fmt_ax
from .compute import compute_sa, compute_psd, arias_curve, husid_metrics

# palette cross-piani
_FLOOR_STYLE = {
    3:  dict(color="#1b7837", ls="-",  lw=2.8),
    5:  dict(color="#762a83", ls="--", lw=2.3),
    10: dict(color="#e08214", ls="-.", lw=2.1),
    15: dict(color="#2166ac", ls=":",  lw=2.1),
}


def _floor_style(nP: int) -> dict:
    return _FLOOR_STYLE.get(nP, dict(color="gray", ls="-", lw=1.8))


# ============================================================
# 1. ACCELERAZIONE ALLA BASE – per ciascuna delle 3 direzioni
# ============================================================

def plot_acc_base(colb: dict) -> list:
    """
    Genera 3 figure (HNE, HNN, HNZ) con la storia di accelerazione
    dell'accelerogramma di input (COLB.txt).

    Parameters
    ----------
    colb : dict  – output di loader.read_colb()

    Returns
    -------
    figs : list[Figure]  – lista di 3 figure matplotlib
    """
    figs = []
    for ax_code in ("HNE", "HNN", "HNZ"):
        if ax_code not in colb:
            continue
        d = colb[ax_code]
        t, a = d["t"], d["acc"]
        pga  = d["pga"]

        fig, ax = plt.subplots(figsize=(14, 4))
        fig.suptitle(f"Accelerogramma di input – {ax_code}  (PGA = {pga:.3f} m/s²)",
                     fontsize=14, fontweight="bold")
        ax.plot(t, a, color="steelblue", lw=1.2, label=f"{ax_code}")
        ax.axhline(+pga, color="red", ls="--", lw=1.4, label=f"+PGA = {pga:.3f} m/s²")
        ax.axhline(-pga, color="red", ls="--", lw=1.4)
        fmt_ax(ax, "Tempo [s]", "Acc [m/s²]", xlim=(0, TMAX),
                ylim=(-1.15 * pga, 1.15 * pga) if pga > 0 else None)
        fig.tight_layout()
        figs.append((f"ACC_BASE_{ax_code}", fig))
    return figs


# ============================================================
# 2. IDR(t) ALL'INTERPIANO CRITICO – per ciascun modello e asse
# ============================================================

def plot_idr_crit(nP: int, df_idr_sum, df_idr_th) -> list:
    """
    Per ciascun (modello, asse) del dataset base-fissa:
      - Individua l'interpiano critico (max IDR_Max)
      - Plotta IDR(t) a quell'interpiano

    Returns
    -------
    figs : list[tuple(name, Figure)]
    """
    figs = []
    axes_avail = sorted(df_idr_sum["Asse"].unique())

    for asse in axes_avail:
        sub_sum = df_idr_sum[
            (df_idr_sum["N_Piani"] == nP) &
            (df_idr_sum["Asse"]    == asse)
        ].copy()
        if sub_sum.empty:
            continue

        modelli = sub_sum["Modello"].unique() if "Modello" in sub_sum.columns else [None]

        for modello in modelli:
            if modello is not None:
                sub_m = sub_sum[sub_sum["Modello"] == modello]
            else:
                sub_m = sub_sum

            if sub_m.empty:
                continue

            # interpiano con IDR_Max maggiore
            crit_row   = sub_m.loc[sub_m["IDR_Max"].astype(float).idxmax()]
            crit_story = int(crit_row["Story_i"])
            idr_max    = float(crit_row["IDR_Max"])

            mod_label = modello if modello else "RIGIDA"

            # time history IDR
            t, idr = get_idr_th(df_idr_th, nP,
                                 modello if modello else crit_row.get("Modello", ""),
                                 asse, crit_story)

            fig, axs = plt.subplots(1, 2, figsize=(16, 5))
            fig.suptitle(
                f"IDR – {nP} Piani | {mod_label} | {asse}\n"
                f"Interpiano critico: i={crit_story}  (IDR_max={idr_max:.4f})",
                fontsize=13, fontweight="bold"
            )

            # profilo IDR max
            sub_m_sorted = sub_m.sort_values("Story_i")
            axs[0].plot(
                sub_m_sorted["IDR_Max"].astype(float),
                sub_m_sorted["Story_i"].astype(int),
                marker="o", color="navy", lw=2.0
            )
            axs[0].axvline(0, color="gray", lw=0.8, ls="--")
            fmt_ax(axs[0], "IDR max [-]", "Interpiano i",
                    title="Profilo IDR max lungo l'altezza")
            axs[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # IDR(t)
            if t is not None:
                axs[1].plot(t, idr, lw=1.3, color="navy")
                axs[1].axhline(idr_max, color="red", ls="--", lw=1.2,
                               label=f"IDR_max={idr_max:.4f}")
            else:
                axs[1].text(0.5, 0.5, "Dati IDR(t) non disponibili",
                            ha="center", va="center", transform=axs[1].transAxes)
            fmt_ax(axs[1], "Tempo [s]", "IDR [-]",
                    xlim=(0, TMAX), title="IDR(t) interpiano critico")

            fig.tight_layout()
            figs.append((f"IDR_{nP}P_{safe_name(mod_label)}_{asse}", fig))

    return figs


# ============================================================
# 3. ACC IN TESTA – 3 punti di controllo, per piano e direzione
# ============================================================

def plot_acc_top(nP: int, asse: str, df_r, df_th,
                 colb: dict, acc_ymax: float,
                 top_tags: list | None = None) -> list:
    """
    Per ciascun Nodo_Tag in 'top_tags' (i 3 punti di controllo in testa):
    plotta la storia di accelerazione con la PGA dell'input come riferimento.

    Se top_tags è None, usa tutti i Nodo_Tag presenti nel dataset.

    Returns
    -------
    figs : list[tuple(name, Figure)]
    """
    sub = df_r[(df_r["N_Piani"] == nP) &
               (df_r["Asse"]    == asse) &
               (df_r["Categoria"] == "RIGIDA")].copy()

    if sub.empty:
        return []

    tags = top_tags if top_tags else sorted(sub["Nodo_Tag"].unique())
    pga_in = colb.get(asse, {}).get("pga", 0.0)

    figs = []
    for tag in tags:
        sub_t = sub[sub["Nodo_Tag"] == tag]
        if sub_t.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(
            f"Acc. in testa – {nP} Piani | Nodo: {tag} | {asse}\n"
            f"Riferimento: ±PGA input = {pga_in:.3f} m/s²",
            fontsize=13, fontweight="bold"
        )

        if pga_in > 0:
            ax.axhline(+pga_in, color="gray", ls="--", lw=1.5,
                       alpha=0.8, label=f"±PGA input = {pga_in:.3f} m/s²")
            ax.axhline(-pga_in, color="gray", ls="--", lw=1.5, alpha=0.8)

        for _, row in sub_t.iterrows():
            t, a = get_th(df_th, row["Modello"], row["Nodo_ID"], asse)
            if t is None:
                continue
            ax.plot(t, a, lw=1.4, label=row["Modello"], alpha=0.85)

        fmt_ax(ax, "Tempo [s]", "Acc [m/s²]",
                xlim=(0, TMAX),
                ylim=(-acc_ymax, acc_ymax))
        fig.tight_layout()
        figs.append((f"ACC_TOP_{nP}P_{safe_name(str(tag))}_{asse}", fig))

    return figs


# ============================================================
# 4. SPETTRI DI PIANO (SA) IN TESTA EDIFICIO
# ============================================================

def plot_spettri_piano(nP: int, asse: str, df_r, df_th,
                       sa_ymax: float,
                       top_tags: list | None = None) -> list:
    """
    Spettri di risposta in accelerazione (5% smorzamento) ai nodi in testa edificio.
    Un diagramma per ciascun Nodo_Tag.

    Returns
    -------
    figs : list[tuple(name, Figure)]
    """
    if asse == "HNZ":
        return []   # solo orizzontale per spettri di piano

    sub = df_r[(df_r["N_Piani"] == nP) &
               (df_r["Asse"]    == asse) &
               (df_r["Categoria"] == "RIGIDA")].copy()
    if sub.empty:
        return []

    tags   = top_tags if top_tags else sorted(sub["Nodo_Tag"].unique())
    cache  = {}
    figs   = []

    for tag in tags:
        sub_t = sub[sub["Nodo_Tag"] == tag]
        if sub_t.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(
            f"Spettro di piano (SA 5%) – {nP} Piani | Nodo: {tag} | {asse}",
            fontsize=13, fontweight="bold"
        )

        for _, row in sub_t.iterrows():
            t, a = get_th(df_th, row["Modello"], row["Nodo_ID"], asse)
            if t is None:
                continue
            sa = compute_sa(t, a, cache, (row["Modello"], str(row["Nodo_ID"]), asse))
            ax.plot(PERIODS, sa, lw=1.8, label=row["Modello"], alpha=0.9)

        fmt_ax(ax, "Periodo [s]", "SA [m/s²]",
                xlim=(0, SA_XMAX), ylim=(0, sa_ymax),
                title="Spettro di risposta in accelerazione (ξ=5%)")
        fig.tight_layout()
        figs.append((f"SA_{nP}P_{safe_name(str(tag))}_{asse}", fig))

    return figs


# ============================================================
# 5. INTENSITÀ DI ARIAS (+ HUSID)
# ============================================================

def plot_arias(nP: int, asse: str, df_r, df_th,
               colb: dict,
               top_tags: list | None = None) -> list:
    """
    Per ciascun Nodo_Tag plotta:
      - fig_arias : Ia(t) di output + Ia(t) di input (linea grigia tratteggiata)
      - fig_husid : curva di Husid normalizzata

    Returns
    -------
    figs : list[tuple(name, Figure)]
    """
    sub = df_r[(df_r["N_Piani"] == nP) &
               (df_r["Asse"]    == asse) &
               (df_r["Categoria"] == "RIGIDA")].copy()
    if sub.empty:
        return []

    tags = top_tags if top_tags else sorted(sub["Nodo_Tag"].unique())

    # input Arias
    colb_ax  = colb.get(asse, {})
    t_in     = colb_ax.get("t", np.array([]))
    a_in     = colb_ax.get("acc", np.array([]))
    Ia_in    = arias_curve(a_in, t_in) if len(a_in) >= 2 else np.array([])

    figs = []
    for tag in tags:
        sub_t = sub[sub["Nodo_Tag"] == tag]
        if sub_t.empty:
            continue

        fig_a, ax_a  = plt.subplots(figsize=(12, 5))
        fig_h, ax_h  = plt.subplots(figsize=(12, 5))

        for fig, ax, ttl in (
            (fig_a, ax_a, f"Intensità di Arias – {nP}P | {tag} | {asse}"),
            (fig_h, ax_h, f"Husid normalizzato  – {nP}P | {tag} | {asse}"),
        ):
            fig.suptitle(ttl, fontsize=13, fontweight="bold")

        if len(Ia_in):
            ax_a.plot(t_in, Ia_in, color="gray", ls="--", lw=1.5,
                      alpha=0.8, label="Input (COLB)")
            m_in = husid_metrics(Ia_in, t_in)
            if m_in["husid"] is not None:
                ax_h.plot(t_in, m_in["husid"], color="gray", ls="--", lw=1.5,
                          alpha=0.8, label="Input (COLB)")

        for _, row in sub_t.iterrows():
            t, a = get_th(df_th, row["Modello"], row["Nodo_ID"], asse)
            if t is None:
                continue
            Ia = arias_curve(a, t)
            m  = husid_metrics(Ia, t)

            ax_a.plot(t, Ia, lw=1.6, label=row["Modello"], alpha=0.9)
            if m["husid"] is not None:
                ax_h.plot(t, m["husid"], lw=1.6,
                          label=f"{row['Modello']} (D5-95={m['D5_95']:.1f}s)",
                          alpha=0.9)

        fmt_ax(ax_a, "Tempo [s]", "Ia [m/s]",  xlim=(0, TMAX))
        fmt_ax(ax_h, "Tempo [s]", "Ia(t)/Ia_tot [-]",
                xlim=(0, TMAX), ylim=(0, 1.05))
        fig_a.tight_layout()
        fig_h.tight_layout()

        base = f"ARIAS_{nP}P_{safe_name(str(tag))}_{asse}"
        figs.append((base,           fig_a))
        figs.append((base + "_HUSID", fig_h))

    return figs

# -*- coding: utf-8 -*-
"""
Grafici – MODALITÀ ISOLAMENTO.

Funzioni esportate:
    plot_idr_isolamento      → IDR per tutti i modelli (rigida + isolati)
    plot_spettri_piano_iso   → Spettri di piano in testa per tutti i modelli
    plot_confronto_pfa_sa    → Confronto PFA e SA_max: rigida vs slitte vs isolato
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .config import TMAX, SA_XMAX, PERIODS
from .loader import safe_name, get_th, get_idr_th, style_for_strategy, extract_tis
from .compute import compute_sa, arias_curve, husid_metrics


# ============================================================
# HELPERS
# ============================================================

def _sorted_models(df_r):
    """
    Restituisce df_r ordinato: RIGIDA prima, poi SLITTE per Tis crescente,
    poi ISOLATA per Tis crescente.
    """
    order_map = {"RIGIDA": 0, "SLITTE": 1, "ISOLATA": 2, "ALTRO": 9}
    tmp = df_r.copy()
    tmp["__ord"] = tmp["Categoria"].map(order_map).fillna(9).astype(int)
    return tmp.sort_values(["__ord", "Tis", "Modello"]).drop(columns="__ord")


def _fmt_ax(ax, xlabel, ylabel, xlim=None, ylim=None, title=None, legend=True):
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
        ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)


# ============================================================
# 1. IDR – tutti i modelli (RIGIDA + ISOLATI)
# ============================================================

def plot_idr_isolamento(nP: int, asse: str,
                        df_idr_sum, df_idr_th,
                        df_r) -> list:
    """
    Per ciascun modello presenti in df_idr_sum plotta:
      - Profilo IDR_max lungo l'altezza
      - IDR(t) all'interpiano critico

    I modelli vengono colorati per categoria (RIGIDA / SLITTE / ISOLATA).

    Returns
    -------
    figs : list[tuple(name, Figure)]
    """
    sub_sum = df_idr_sum[
        (df_idr_sum["N_Piani"] == nP) &
        (df_idr_sum["Asse"]    == asse)
    ].copy()
    if sub_sum.empty:
        return []

    # lista modelli ordinata
    meta = df_r[["Modello", "Categoria", "Tis"]].drop_duplicates()

    # figura unica con profili sovrapposti
    fig_prof, ax_prof = plt.subplots(figsize=(8, 8))
    fig_prof.suptitle(
        f"IDR – Profilo altezza – {nP} Piani | {asse}",
        fontsize=14, fontweight="bold"
    )

    modelli = sub_sum["Modello"].dropna().unique() if "Modello" in sub_sum.columns else []

    figs_th = []
    for modello in modelli:
        sub_m = sub_sum[sub_sum["Modello"] == modello].sort_values("Story_i")
        if sub_m.empty:
            continue

        row_meta = meta[meta["Modello"] == modello]
        cat  = row_meta["Categoria"].iloc[0] if not row_meta.empty else "ALTRO"
        tis  = float(row_meta["Tis"].iloc[0]) if not row_meta.empty else 0.0
        st   = style_for_strategy(cat, tis)

        # profilo
        ax_prof.plot(
            sub_m["IDR_Max"].astype(float),
            sub_m["Story_i"].astype(int),
            marker="o", color=st["color"], lw=st["lw"],
            label=st["label"], zorder=st["zorder"]
        )

        # IDR(t) interpiano critico → figura separata
        crit_row   = sub_m.loc[sub_m["IDR_Max"].astype(float).idxmax()]
        crit_story = int(crit_row["Story_i"])
        idr_max    = float(crit_row["IDR_Max"])

        t, idr = get_idr_th(df_idr_th, nP, modello, asse, crit_story)
        if t is not None:
            fig_t, ax_t = plt.subplots(figsize=(12, 4))
            fig_t.suptitle(
                f"IDR(t) – {nP}P | {modello} | {asse} | interpiano {crit_story}"
                f"  (IDR_max={idr_max:.4f})",
                fontsize=12, fontweight="bold"
            )
            ax_t.plot(t, idr, color=st["color"], lw=1.4)
            ax_t.axhline(idr_max, color="red", ls="--", lw=1.1,
                         label=f"IDR_max={idr_max:.4f}")
            _fmt_ax(ax_t, "Tempo [s]", "IDR [-]", xlim=(0, TMAX))
            fig_t.tight_layout()
            figs_th.append((f"IDR_TH_{nP}P_{safe_name(modello)}_{asse}", fig_t))

    ax_prof.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _fmt_ax(ax_prof, "IDR max [-]", "Interpiano i")
    fig_prof.tight_layout()

    return [(f"IDR_PROFILO_{nP}P_{asse}", fig_prof)] + figs_th


# ============================================================
# 2. SPETTRI DI PIANO IN TESTA – tutti i modelli
# ============================================================

def plot_spettri_piano_iso(nP: int, asse: str, df_r, df_th,
                           sa_ymax: float,
                           top_tags: list | None = None) -> list:
    """
    Spettri SA in testa edificio per tutti i modelli (rigida + isolati).
    Un diagramma per Nodo_Tag.

    Returns
    -------
    figs : list[tuple(name, Figure)]
    """
    if asse == "HNZ":
        return []

    sub = df_r[(df_r["N_Piani"] == nP) & (df_r["Asse"] == asse)].copy()
    if sub.empty:
        return []

    sub    = _sorted_models(sub)
    tags   = top_tags if top_tags else sorted(sub["Nodo_Tag"].unique())
    cache  = {}
    figs   = []

    for tag in tags:
        sub_t = sub[sub["Nodo_Tag"] == tag]
        if sub_t.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.suptitle(
            f"Spettri di piano (SA 5%) – {nP} Piani | Nodo: {tag} | {asse}\n"
            "Confronto: Base Fissa / Isolato / Slitte",
            fontsize=13, fontweight="bold"
        )

        for _, row in sub_t.iterrows():
            t, a = get_th(df_th, row["Modello"], row["Nodo_ID"], asse)
            if t is None:
                continue
            st = style_for_strategy(row["Categoria"], float(row["Tis"]))
            sa = compute_sa(t, a, cache, (row["Modello"], str(row["Nodo_ID"]), asse))
            ax.plot(PERIODS, sa, color=st["color"], lw=st["lw"],
                    alpha=0.9, label=st["label"], zorder=st["zorder"])

        _fmt_ax(ax, "Periodo [s]", "SA [m/s²]",
                xlim=(0, SA_XMAX), ylim=(0, sa_ymax))
        fig.tight_layout()
        figs.append((f"SA_ISO_{nP}P_{safe_name(str(tag))}_{asse}", fig))

    return figs


# ============================================================
# 3. CONFRONTO PFA e SA_max – isolato vs non isolato
# ============================================================

def plot_confronto_pfa_sa(df_risultati: "pd.DataFrame") -> list:
    """
    Produce 4 figure di confronto basate sulla tabella dei risultati
    (output di tables.build_results_table):

      A) Bar chart PFA_Max per asse e numero di piani, raggruppato per strategia
      B) Bar chart SA_Max per asse e numero di piani, raggruppato per strategia
      C) Scatter PFA_Max vs SA_Max colorato per categoria (tutti i piani)
      D) Riduzione % PFA e SA per piano (linee: Slitte vs Isolato per T diversi)

    Parameters
    ----------
    df_risultati : DataFrame
        Deve contenere: N_Piani, Nodo_Tag, Asse, Modello, Categoria, Tis,
                        PFA_Max, SA_Max, Rid_PFA_pct, Rid_SA_pct

    Returns
    -------
    figs : list[tuple(name, Figure)]
    """
    import pandas as pd

    figs = []

    # ----------- palette --------------
    _cat_color = {"RIGIDA": "black", "SLITTE": "#D55E00", "ISOLATA": "#0072B2", "ALTRO": "gray"}

    # ----------- A: bar PFA per piano --------------
    for asse in ("HNE", "HNN", "HNZ"):
        sub = df_risultati[df_risultati["Asse"] == asse].copy()
        if sub.empty:
            continue

        strats   = sub.groupby(["Categoria", "Tis"])["Strategia"].first().reset_index()
        strats   = strats.sort_values(["Categoria", "Tis"])
        piani    = sorted(sub["N_Piani"].unique())
        x        = np.arange(len(strats))
        width    = 0.18
        n_piani  = len(piani)

        fig, ax = plt.subplots(figsize=(max(10, 2 * len(strats)), 6))
        fig.suptitle(f"PFA Max – confronto strategie | {asse}", fontsize=14, fontweight="bold")

        for i, nP in enumerate(piani):
            vals = []
            for _, s in strats.iterrows():
                mean_pfa = sub[(sub["N_Piani"] == nP) &
                               (sub["Categoria"] == s["Categoria"]) &
                               (sub["Tis"] == s["Tis"])]["PFA_Max"].mean()
                vals.append(mean_pfa if not np.isnan(mean_pfa) else 0.0)

            offset = (i - n_piani / 2.0 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width, label=f"{nP} Piani",
                            alpha=0.85, edgecolor="white")
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(strats["Strategia"].tolist(), rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("PFA Max [m/s²]", fontsize=11)
        ax.grid(axis="y", ls="--", alpha=0.5)
        ax.legend(fontsize=9)
        fig.tight_layout()
        figs.append((f"BAR_PFA_{asse}", fig))

    # ----------- B: bar SA per piano (solo HNE/HNN) --------------
    for asse in ("HNE", "HNN"):
        sub = df_risultati[df_risultati["Asse"] == asse].copy()
        if sub.empty:
            continue

        strats  = sub.groupby(["Categoria", "Tis"])["Strategia"].first().reset_index()
        strats  = strats.sort_values(["Categoria", "Tis"])
        piani   = sorted(sub["N_Piani"].unique())
        x       = np.arange(len(strats))
        width   = 0.18
        n_piani = len(piani)

        fig, ax = plt.subplots(figsize=(max(10, 2 * len(strats)), 6))
        fig.suptitle(f"SA Max – confronto strategie | {asse}", fontsize=14, fontweight="bold")

        for i, nP in enumerate(piani):
            vals = []
            for _, s in strats.iterrows():
                mean_sa = sub[(sub["N_Piani"] == nP) &
                              (sub["Categoria"] == s["Categoria"]) &
                              (sub["Tis"] == s["Tis"])]["SA_Max"].mean()
                vals.append(mean_sa if not np.isnan(mean_sa) else 0.0)

            offset = (i - n_piani / 2.0 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width, label=f"{nP} Piani",
                            alpha=0.85, edgecolor="white")
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(strats["Strategia"].tolist(), rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("SA Max [m/s²]", fontsize=11)
        ax.grid(axis="y", ls="--", alpha=0.5)
        ax.legend(fontsize=9)
        fig.tight_layout()
        figs.append((f"BAR_SA_{asse}", fig))

    # ----------- C: Scatter PFA vs SA (HNE+HNN) --------------
    sub_horiz = df_risultati[df_risultati["Asse"].isin(["HNE", "HNN"])].copy()
    if not sub_horiz.empty:
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.suptitle("Scatter PFA_Max vs SA_Max – tutti i piani e assi orizzontali",
                     fontsize=13, fontweight="bold")

        for cat, grp in sub_horiz.groupby("Categoria"):
            ax.scatter(grp["PFA_Max"], grp["SA_Max"],
                       color=_cat_color.get(cat, "gray"),
                       alpha=0.75, s=60, label=cat, zorder=3)

        ax.set_xlabel("PFA Max [m/s²]", fontsize=11)
        ax.set_ylabel("SA Max [m/s²]", fontsize=11)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=10)
        fig.tight_layout()
        figs.append(("SCATTER_PFA_vs_SA", fig))

    # ----------- D: Riduzione % per numero di piani (curva per T_target) --------------
    if "Rid_PFA_pct" in df_risultati.columns:
        sub_red = df_risultati[
            df_risultati["Categoria"].isin(["SLITTE", "ISOLATA"])
        ].copy()

        for asse in ("HNE", "HNN"):
            sub_a = sub_red[sub_red["Asse"] == asse]
            if sub_a.empty:
                continue

            fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
            fig.suptitle(f"Riduzione PFA e SA vs N. Piani | {asse}", fontsize=13, fontweight="bold")

            for col, ax, ylabel in (
                ("Rid_PFA_pct", axs[0], "Riduzione PFA [%]"),
                ("Rid_SA_pct",  axs[1], "Riduzione SA  [%]"),
            ):
                for (cat, tis), grp in sub_a.groupby(["Categoria", "Tis"]):
                    by_piani = grp.groupby("N_Piani")[col].mean().reset_index()
                    ls  = "--" if cat == "SLITTE" else "-"
                    col_c = _cat_color.get(cat, "gray")
                    ax.plot(by_piani["N_Piani"], by_piani[col],
                            marker="o", ls=ls, lw=2.0, color=col_c,
                            label=f"{cat} T={tis:.1f}s", alpha=0.9)

                ax.set_xlabel("N. Piani", fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_ylim(0, 100)
                ax.grid(True, ls=":", alpha=0.55)
                ax.legend(fontsize=8, ncol=2)

            fig.tight_layout()
            figs.append((f"RIDUZIONE_vs_PIANI_{asse}", fig))

    return figs

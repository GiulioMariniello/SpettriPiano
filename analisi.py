#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisi.py – Script principale di post-processing sismico.

Uso:
    python analisi.py --mode rigida
    python analisi.py --mode isolamento
    python analisi.py --mode entrambi          # default

Parametri (tutti opzionali, sovrascrivono src/config.py):
    --excel   PATH   file Excel master  (default: RISULTATI_COMPLETI.xlsx)
    --colb    PATH   accelerogramma     (default: COLB.txt)
    --dt      FLOAT  passo COLB [s]     (default: 0.01)
    --piani   INT [INT ...]  piani da analizzare (default: 3 5 10 15)
    --top     STR [STR ...]  Nodo_Tag dei punti di controllo in testa
                             (default: tutti quelli presenti nei dati)
    --outdir  PATH   cartella radice di output  (default: da config.py)
"""

import argparse
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")  # headless (niente GUI)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import package locale
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src import config as cfg
from src.loader import read_colb, load_excel, get_th, filter_idr_rigida
from src.compute import (
    compute_sa, arias_curve, husid_metrics,
    global_sa_ymax, global_acc_ymax,
)
from src.plots_rigida import (
    plot_acc_base,
    plot_idr_crit,
    plot_acc_top,
    plot_spettri_piano,
    plot_arias,
)
from src.plots_isolamento import (
    plot_idr_isolamento,
    plot_spettri_piano_iso,
    plot_confronto_pfa_sa,
)
from src.tables import (
    build_results_table,
    build_idr_table,
    compute_reductions,
    save_tables,
)


# ============================================================
# SALVATAGGIO FIGURE
# ============================================================

def _save(fig, path: str, dpi: int = 150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_all(figs: list, directory: str, dpi: int = 150):
    """Salva una lista di (nome, Figure) nella directory indicata."""
    os.makedirs(directory, exist_ok=True)
    for name, fig in figs:
        _save(fig, os.path.join(directory, f"{name}.png"), dpi=dpi)


# ============================================================
# MODALITÀ BASE FISSA
# ============================================================

def run_rigida(df_r, df_th, df_idr_sum, df_idr_th,
               colb: dict, floors: list, top_tags: list | None):
    print("\n" + "=" * 60)
    print("  MODALITÀ BASE FISSA")
    print("=" * 60)

    # filtra solo RIGIDA
    df_r_rig = df_r[df_r["Categoria"] == "RIGIDA"].copy()
    if df_r_rig.empty:
        print("⚠️  Nessun modello RIGIDA trovato nel dataset.")
        return

    df_idr_sum_rig, df_idr_th_rig = filter_idr_rigida(df_idr_sum, df_idr_th)

    # limiti globali per scale uniformi
    print("  Calcolo limiti globali per scale uniformi…")
    acc_ymax = global_acc_ymax(df_r_rig, df_th, get_th, colb)
    sa_ymax  = global_sa_ymax(df_r_rig,  df_th, get_th)

    # ── 1. Acc alla base ────────────────────────────────────
    print("  [1/5] Storie di accelerazione alla base…")
    figs_base = plot_acc_base(colb)
    _save_all(figs_base, cfg.DIR_R_BASE)

    # ── 2. IDR(t) interpiano critico ────────────────────────
    print("  [2/5] IDR(t) interpiano critico…")
    for nP in floors:
        sub_sum = df_idr_sum_rig[df_idr_sum_rig["N_Piani"] == nP]
        if sub_sum.empty:
            continue
        figs_idr = plot_idr_crit(nP, sub_sum, df_idr_th_rig)
        _save_all(figs_idr, cfg.DIR_R_IDR)

    # ── 3. Acc in testa – 3 punti di controllo ─────────────
    print("  [3/5] Acc in testa edificio…")
    for nP in floors:
        for asse in ("HNE", "HNN", "HNZ"):
            figs_top = plot_acc_top(nP, asse, df_r_rig, df_th,
                                    colb, acc_ymax, top_tags)
            _save_all(figs_top, cfg.DIR_R_ACC_TOP)

    # ── 4. Spettri di piano ──────────────────────────────────
    print("  [4/5] Spettri di piano (SA 5%)…")
    for nP in floors:
        for asse in ("HNE", "HNN"):
            figs_sa = plot_spettri_piano(nP, asse, df_r_rig, df_th,
                                         sa_ymax, top_tags)
            _save_all(figs_sa, cfg.DIR_R_SA)

    # ── 5. Intensità di Arias + Husid ────────────────────────
    print("  [5/5] Intensità di Arias e Husid…")
    for nP in floors:
        for asse in ("HNE", "HNN", "HNZ"):
            figs_ar = plot_arias(nP, asse, df_r_rig, df_th, colb, top_tags)
            _save_all(figs_ar, cfg.DIR_R_ARIAS)

    # ── Tabelle ──────────────────────────────────────────────
    print("  Generazione tabelle…")
    df_res = build_results_table(df_r_rig, df_th)
    df_idr = build_idr_table(df_idr_sum_rig)

    # tabella IDR max editabile
    os.makedirs(cfg.DIR_R_TABLES, exist_ok=True)
    df_idr.to_excel(os.path.join(cfg.DIR_R_TABLES, "IDR_MAX.xlsx"), index=False)
    df_res.to_excel(os.path.join(cfg.DIR_R_TABLES, "PFA_SA_ARIAS.xlsx"), index=False)

    print(f"\n  Output base fissa → {cfg.OUTDIR_RIGIDA}/")


# ============================================================
# MODALITÀ ISOLAMENTO
# ============================================================

def run_isolamento(df_r, df_th, df_idr_sum, df_idr_th,
                   colb: dict, floors: list, top_tags: list | None):
    print("\n" + "=" * 60)
    print("  MODALITÀ ISOLAMENTO")
    print("=" * 60)

    if df_r.empty:
        print("⚠️  Dataset vuoto.")
        return

    # limiti globali (su tutti i modelli: rigida + isolati)
    print("  Calcolo limiti globali per scale uniformi…")
    sa_ymax = global_sa_ymax(
        df_r[df_r["Asse"].isin(["HNE", "HNN"])], df_th, get_th
    )

    # ── 1. IDR – tutti i modelli ────────────────────────────
    print("  [1/3] IDR per tutti i modelli…")
    for nP in floors:
        for asse in ("HNE", "HNN"):
            sub_sum = df_idr_sum[df_idr_sum["N_Piani"] == nP]
            if sub_sum.empty:
                continue
            figs_idr = plot_idr_isolamento(nP, asse, sub_sum, df_idr_th, df_r)
            _save_all(figs_idr, cfg.DIR_I_IDR)

    # ── 2. Spettri di piano in testa ────────────────────────
    print("  [2/3] Spettri di piano (tutti i modelli)…")
    for nP in floors:
        for asse in ("HNE", "HNN"):
            figs_sa = plot_spettri_piano_iso(nP, asse, df_r, df_th,
                                             sa_ymax, top_tags)
            _save_all(figs_sa, cfg.DIR_I_SA)

    # ── 3. Confronto PFA / SA_max ────────────────────────────
    print("  [3/3] Confronto PFA e SA_max (rigida vs isolato vs slitte)…")
    df_res  = build_results_table(df_r, df_th)
    df_rid  = compute_reductions(df_res)
    figs_cf = plot_confronto_pfa_sa(df_rid)
    _save_all(figs_cf, cfg.DIR_I_CONFRONTO)

    # ── Tabelle ──────────────────────────────────────────────
    print("  Generazione tabelle…")
    df_idr  = build_idr_table(df_idr_sum)
    os.makedirs(cfg.DIR_I_TABLES, exist_ok=True)
    save_tables(
        os.path.join(cfg.DIR_I_TABLES, "RISULTATI_ISOLAMENTO.xlsx"),
        df_res, df_idr, df_rid
    )

    print(f"\n  Output isolamento → {cfg.OUTDIR_ISOLAMENTO}/")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Post-processing sismico SpettriPiano")
    p.add_argument("--mode",   default="entrambi",
                   choices=["rigida", "isolamento", "entrambi"],
                   help="Modalità di analisi (default: entrambi)")
    p.add_argument("--excel",  default=cfg.EXCEL_MASTER,
                   help=f"Excel master (default: {cfg.EXCEL_MASTER})")
    p.add_argument("--colb",   default=cfg.INPUT_COLB,
                   help=f"COLB.txt (default: {cfg.INPUT_COLB})")
    p.add_argument("--dt",     type=float, default=cfg.INPUT_DT,
                   help=f"passo temporale COLB [s] (default: {cfg.INPUT_DT})")
    p.add_argument("--piani",  type=int,   nargs="+", default=cfg.FLOORS,
                   help=f"piani da analizzare (default: {cfg.FLOORS})")
    p.add_argument("--top",    nargs="+",  default=None,
                   help="Nodo_Tag dei punti di controllo in testa (default: tutti)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Output dirs ──────────────────────────────────────────
    for d in cfg.ALL_OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)

    # ── Caricamento dati ─────────────────────────────────────
    print(f"\nCaricamento COLB: {args.colb}")
    colb = read_colb(args.colb, dt=args.dt)

    print(f"Caricamento Excel: {args.excel}")
    df_r, df_th, df_idr_sum, df_idr_th = load_excel(args.excel)

    # filtra piani richiesti
    if "N_Piani" in df_r.columns:
        df_r       = df_r[df_r["N_Piani"].isin(args.piani)].copy()
    if "N_Piani" in df_idr_sum.columns:
        df_idr_sum = df_idr_sum[df_idr_sum["N_Piani"].isin(args.piani)].copy()
    if "N_Piani" in df_idr_th.columns:
        df_idr_th  = df_idr_th[df_idr_th["N_Piani"].isin(args.piani)].copy()

    print(f"Modelli trovati: {sorted(df_r['Categoria'].unique())}")
    print(f"Piani analizzati: {args.piani}")
    print(f"Punti di controllo: {args.top if args.top else 'tutti'}")

    # ── Esecuzione ───────────────────────────────────────────
    if args.mode in ("rigida", "entrambi"):
        run_rigida(df_r, df_th, df_idr_sum, df_idr_th,
                   colb, args.piani, args.top)

    if args.mode in ("isolamento", "entrambi"):
        run_isolamento(df_r, df_th, df_idr_sum, df_idr_th,
                       colb, args.piani, args.top)

    print("\nAnalisi completata.")


if __name__ == "__main__":
    main()

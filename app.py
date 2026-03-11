# -*- coding: utf-8 -*-
"""
app.py – Interfaccia Streamlit per SpettriPiano.

Esecuzione locale:
    streamlit run app.py

HuggingFace Spaces:
    entry point automatico (vedi README.md)
"""

import io
import os
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# PATH setup (funziona sia locale che su HuggingFace)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

import src.config as cfg
from src.loader import (
    read_colb, load_excel, get_th,
    classify_strategy, extract_tis,
)
from src.compute import (
    compute_sa, arias_curve, husid_metrics,
    global_sa_ymax, global_acc_ymax,
)
from src.plots_rigida import (
    plot_acc_base, plot_idr_crit,
    plot_acc_top, plot_spettri_piano, plot_arias,
)
from src.plots_isolamento import (
    plot_idr_isolamento, plot_spettri_piano_iso, plot_confronto_pfa_sa,
)
from src.tables import (
    build_results_table, build_idr_table,
    compute_reductions, save_tables,
)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SpettriPiano",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { color: #1a3a5c; }
    h2 { color: #2a5a8c; margin-top: 1.2rem; }
    h3 { color: #3a7aac; }
    .stButton>button { width: 100%; }
    .plot-caption { font-size: 0.8rem; color: #666; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _figs_to_zip(figs: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figs:
            zf.writestr(f"{name}.png", _fig_to_bytes(fig))
    buf.seek(0)
    return buf.getvalue()


def _tables_to_excel(df_res, df_idr, df_rid) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_res.to_excel(writer, sheet_name="Risultati", index=False)
        df_idr.to_excel(writer, sheet_name="IDR",       index=False)
        df_rid.to_excel(writer, sheet_name="Riduzioni", index=False)
    buf.seek(0)
    return buf.getvalue()


def _show_figs(figs: list, cols: int = 2):
    """Mostra lista (name, fig) in griglia a 'cols' colonne."""
    if not figs:
        st.info("Nessun grafico disponibile per questa selezione.")
        return
    for i in range(0, len(figs), cols):
        row = figs[i:i + cols]
        columns = st.columns(len(row))
        for col, (name, fig) in zip(columns, row):
            with col:
                st.pyplot(fig, use_container_width=True)
                st.caption(name)


@st.cache_data(show_spinner=False)
def _load_colb(colb_bytes: bytes, dt: float) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(colb_bytes)
        tmp_path = tmp.name
    try:
        result = read_colb(tmp_path, dt=dt)
    finally:
        os.unlink(tmp_path)
    return result


@st.cache_data(show_spinner=False)
def _load_excel(excel_bytes: bytes) -> tuple:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(excel_bytes)
        tmp_path = tmp.name
    try:
        result = load_excel(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🏢 SpettriPiano")
    st.caption("Post-processing sismico | Isolamento sismico")

    st.divider()
    st.subheader("📁 File di input")

    colb_file  = st.file_uploader("Accelerogramma input (COLB.txt)",
                                   type=["txt"], key="colb_upload")
    excel_file = st.file_uploader("Excel master (.xlsx)",
                                   type=["xlsx", "xls"], key="excel_upload")

    st.divider()
    st.subheader("⚙️ Parametri globali")

    dt_val = st.number_input("Passo temporale COLB [s]",
                              value=0.01, min_value=0.0001,
                              step=0.001, format="%.4f")
    tmax_val = st.number_input("Finestra analisi TMAX [s]",
                                value=30.0, min_value=1.0, step=1.0)
    damp_val = st.slider("Smorzamento spettrale ξ [%]",
                          min_value=1, max_value=20, value=5) / 100.0
    sa_xmax_val = st.number_input("Max periodo grafici SA [s]",
                                   value=3.0, min_value=0.5, step=0.5)

    st.divider()
    st.subheader("🔢 Piani da analizzare")
    floors_val = st.multiselect("N. Piani", options=[3, 5, 10, 15],
                                 default=[3, 5, 10, 15])

    st.divider()
    st.subheader("🎯 Punti di controllo (testa)")
    top_input = st.text_input(
        "Nodo_Tag (separati da virgola, vuoto = tutti)",
        value="",
        help="Esempio: TESTA_A, TESTA_B, TESTA_C"
    )
    top_tags = [t.strip() for t in top_input.split(",") if t.strip()] or None

    st.divider()
    load_btn = st.button("📂 Carica dati", type="primary")


# ============================================================
# AGGIORNA CONFIG a runtime
# ============================================================
cfg.TMAX        = tmax_val
cfg.OSC_DAMPING = damp_val
cfg.SA_XMAX     = sa_xmax_val


# ============================================================
# CARICAMENTO DATI (session state)
# ============================================================

if load_btn:
    if not colb_file or not excel_file:
        st.sidebar.error("Carica entrambi i file prima di procedere.")
    else:
        with st.spinner("Caricamento dati in corso…"):
            try:
                colb_data = _load_colb(colb_file.read(), dt_val)
                df_r, df_th, df_idr_sum, df_idr_th = _load_excel(excel_file.read())

                # applica filtro piani
                if "N_Piani" in df_r.columns:
                    df_r       = df_r[df_r["N_Piani"].isin(floors_val)].copy()
                if "N_Piani" in df_idr_sum.columns:
                    df_idr_sum = df_idr_sum[df_idr_sum["N_Piani"].isin(floors_val)].copy()
                if "N_Piani" in df_idr_th.columns:
                    df_idr_th  = df_idr_th[df_idr_th["N_Piani"].isin(floors_val)].copy()

                st.session_state["colb"]      = colb_data
                st.session_state["df_r"]      = df_r
                st.session_state["df_th"]     = df_th
                st.session_state["df_idr_sum"]= df_idr_sum
                st.session_state["df_idr_th"] = df_idr_th
                st.session_state["floors"]    = floors_val
                st.session_state["top_tags"]  = top_tags
                st.session_state["loaded"]    = True
                st.sidebar.success("✅ Dati caricati correttamente!")
            except Exception as e:
                st.sidebar.error(f"Errore nel caricamento: {e}")
                st.session_state["loaded"] = False


# ============================================================
# MAIN AREA
# ============================================================

if not st.session_state.get("loaded"):
    st.title("🏢 SpettriPiano")
    st.markdown("""
    ### Analisi di risposta sismica – Isolamento sismico

    **Istruzioni:**
    1. Carica il file **COLB.txt** (accelerogramma, 3 colonne in *g*, spazio-separato)
    2. Carica il file **Excel master** con i fogli:
       - `Riepilogo` | `TimeHistory` | `IDR_Riepilogo` | `IDR_TimeHistory`
    3. Imposta i parametri nel pannello laterale
    4. Clicca **Carica dati**
    5. Esplora i risultati nelle schede

    ---
    **Modalità disponibili:**
    - 🏗️ **Base Fissa** → acc alla base, IDR, acc testa, spettri di piano, Arias
    - 🔵 **Isolamento** → IDR, spettri di piano, confronto PFA/SA
    """)

    st.info("⬅️ Inizia caricando i file nella barra laterale.")
    st.stop()


# ---------------------------------------------------------------------------
# Dati disponibili
# ---------------------------------------------------------------------------
colb      = st.session_state["colb"]
df_r      = st.session_state["df_r"]
df_th     = st.session_state["df_th"]
df_idr_sum= st.session_state["df_idr_sum"]
df_idr_th = st.session_state["df_idr_th"]
floors    = st.session_state["floors"]
top_tags  = st.session_state["top_tags"]

# Dataset base fissa
df_r_rig = df_r[df_r["Categoria"] == "RIGIDA"].copy()

# Idr solo base fissa
df_idr_sum_rig = df_idr_sum.copy()
df_idr_th_rig  = df_idr_th.copy()
for dfi in (df_idr_sum_rig, df_idr_th_rig):
    if "Tipo_Base" in dfi.columns:
        mask = dfi["Tipo_Base"].astype(str).str.upper().str.contains("RIG", na=False)
        dfi.drop(dfi[~mask].index, inplace=True)

# ============================================================
# TABS PRINCIPALI
# ============================================================
tab_info, tab_rig, tab_iso, tab_tab = st.tabs([
    "📁 Dati",
    "🏗️ Base Fissa",
    "🔵 Isolamento",
    "📊 Tabelle",
])


# ─────────────────────────────────────────────────────────────
# TAB 0 – INFO / DATI
# ─────────────────────────────────────────────────────────────
with tab_info:
    st.header("Riepilogo dati caricati")

    c1, c2, c3, c4 = st.columns(4)
    cats = df_r["Categoria"].value_counts()
    c1.metric("Modelli RIGIDA",  int(cats.get("RIGIDA",  0)))
    c2.metric("Modelli ISOLATA", int(cats.get("ISOLATA", 0)))
    c3.metric("Modelli SLITTE",  int(cats.get("SLITTE",  0)))
    c4.metric("Righe TH",        len(df_th))

    st.subheader("PGA input (COLB.txt)")
    if colb:
        pga_df = pd.DataFrame([
            {"Asse": ax, "PGA [m/s²]": round(d["pga"], 4), "N. campioni": len(d["t"])}
            for ax, d in colb.items()
        ])
        st.dataframe(pga_df, use_container_width=True, hide_index=True)
    else:
        st.warning("COLB.txt non caricato o non valido.")

    st.subheader("Modelli disponibili")
    cols_show = [c for c in ["Modello", "Categoria", "Tis", "N_Piani", "Asse", "Nodo_Tag"]
                 if c in df_r.columns]
    st.dataframe(
        df_r[cols_show].drop_duplicates().sort_values(["Categoria", "Tis", "N_Piani"]),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Fogli Excel – anteprima")
    with st.expander("Riepilogo (prime 50 righe)"):
        st.dataframe(df_r.head(50), use_container_width=True)
    with st.expander("TimeHistory (prime 50 righe)"):
        st.dataframe(df_th.head(50), use_container_width=True)
    with st.expander("IDR_Riepilogo (prime 50 righe)"):
        st.dataframe(df_idr_sum.head(50), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 1 – BASE FISSA
# ─────────────────────────────────────────────────────────────
with tab_rig:
    st.header("🏗️ Modalità Base Fissa")

    if df_r_rig.empty:
        st.warning("Nessun modello RIGIDA trovato nel dataset.")
    else:
        # Controlli comuni
        col_nP, col_ax = st.columns([1, 2])
        with col_nP:
            nP_sel = st.selectbox("N. Piani", sorted(df_r_rig["N_Piani"].unique())
                                  if "N_Piani" in df_r_rig.columns else floors,
                                  key="rig_nP")
        with col_ax:
            ax_sel = st.multiselect("Assi", ["HNE", "HNN", "HNZ"],
                                    default=["HNE", "HNN", "HNZ"], key="rig_ax")

        # Limiti globali (calcolati una volta)
        @st.cache_data(show_spinner="Calcolo limiti globali…")
        def _get_limits(_df_r, _df_th, _colb):
            acc_y = global_acc_ymax(_df_r, _df_th, get_th, _colb)
            sa_y  = global_sa_ymax(_df_r, _df_th, get_th)
            return acc_y, sa_y

        acc_ymax, sa_ymax = _get_limits(df_r_rig, df_th, colb)

        st.divider()

        # ── 1. Acc alla base ─────────────────────────────────
        with st.expander("1. Storie di accelerazione alla BASE (COLB.txt)", expanded=True):
            if st.button("▶ Genera grafici ACC Base", key="btn_acc_base"):
                with st.spinner("Generazione…"):
                    figs = plot_acc_base(colb)
                    figs_f = [(n, f) for n, f in figs if any(ax in n for ax in ax_sel)]
                st.session_state["figs_acc_base"] = figs_f

            if "figs_acc_base" in st.session_state:
                figs_f = st.session_state["figs_acc_base"]
                _show_figs(figs_f, cols=1)
                if figs_f:
                    st.download_button("⬇️ Scarica ZIP acc base",
                                       data=_figs_to_zip(figs_f),
                                       file_name="ACC_BASE.zip",
                                       mime="application/zip",
                                       key="dl_acc_base")

        # ── 2. IDR(t) interpiano critico ─────────────────────
        with st.expander("2. IDR(t) – interpiano critico"):
            if st.button("▶ Genera grafici IDR", key="btn_idr_rig"):
                with st.spinner("Generazione IDR…"):
                    figs = []
                    sub_sum = df_idr_sum_rig[df_idr_sum_rig["N_Piani"] == nP_sel]
                    for ax in ax_sel:
                        sub_ax = sub_sum[sub_sum["Asse"] == ax]
                        figs += plot_idr_crit(nP_sel, sub_ax, df_idr_th_rig)
                st.session_state["figs_idr_rig"] = figs

            if "figs_idr_rig" in st.session_state:
                figs = st.session_state["figs_idr_rig"]
                _show_figs(figs, cols=2)
                if figs:
                    st.download_button("⬇️ Scarica ZIP IDR",
                                       data=_figs_to_zip(figs),
                                       file_name="IDR_BASE_FISSA.zip",
                                       mime="application/zip",
                                       key="dl_idr_rig")

        # ── 3. Acc in testa ──────────────────────────────────
        with st.expander("3. Storie di accelerazione in TESTA (±PGA input)"):
            if st.button("▶ Genera grafici Acc Testa", key="btn_acc_top"):
                with st.spinner("Generazione…"):
                    figs = []
                    for ax in ax_sel:
                        figs += plot_acc_top(nP_sel, ax, df_r_rig, df_th,
                                             colb, acc_ymax, top_tags)
                st.session_state["figs_acc_top"] = figs

            if "figs_acc_top" in st.session_state:
                figs = st.session_state["figs_acc_top"]
                _show_figs(figs, cols=1)
                if figs:
                    st.download_button("⬇️ Scarica ZIP acc testa",
                                       data=_figs_to_zip(figs),
                                       file_name="ACC_TESTA.zip",
                                       mime="application/zip",
                                       key="dl_acc_top")

        # ── 4. Spettri di piano ──────────────────────────────
        with st.expander("4. Spettri di piano (SA 5%)"):
            if st.button("▶ Genera Spettri di Piano", key="btn_sa_rig"):
                with st.spinner("Calcolo spettri SA – potrebbe richiedere qualche secondo…"):
                    figs = []
                    for ax in [a for a in ax_sel if a != "HNZ"]:
                        figs += plot_spettri_piano(nP_sel, ax, df_r_rig, df_th,
                                                   sa_ymax, top_tags)
                st.session_state["figs_sa_rig"] = figs

            if "figs_sa_rig" in st.session_state:
                figs = st.session_state["figs_sa_rig"]
                _show_figs(figs, cols=2)
                if figs:
                    st.download_button("⬇️ Scarica ZIP spettri",
                                       data=_figs_to_zip(figs),
                                       file_name="SPETTRI_PIANO.zip",
                                       mime="application/zip",
                                       key="dl_sa_rig")

        # ── 5. Arias + Husid ─────────────────────────────────
        with st.expander("5. Intensità di Arias e Husid"):
            if st.button("▶ Genera Arias / Husid", key="btn_arias_rig"):
                with st.spinner("Generazione…"):
                    figs = []
                    for ax in ax_sel:
                        figs += plot_arias(nP_sel, ax, df_r_rig, df_th,
                                           colb, top_tags)
                st.session_state["figs_arias_rig"] = figs

            if "figs_arias_rig" in st.session_state:
                figs = st.session_state["figs_arias_rig"]
                _show_figs(figs, cols=2)
                if figs:
                    st.download_button("⬇️ Scarica ZIP Arias",
                                       data=_figs_to_zip(figs),
                                       file_name="ARIAS_HUSID.zip",
                                       mime="application/zip",
                                       key="dl_arias_rig")


# ─────────────────────────────────────────────────────────────
# TAB 2 – ISOLAMENTO
# ─────────────────────────────────────────────────────────────
with tab_iso:
    st.header("🔵 Modalità Isolamento")

    cats_available = df_r["Categoria"].unique()
    if not any(c in cats_available for c in ("ISOLATA", "SLITTE")):
        st.warning("Nessun modello isolato trovato. Carica un dataset con modelli ISOLATA/SLITTE.")
    else:
        col_nP2, col_ax2 = st.columns([1, 2])
        with col_nP2:
            nP_iso = st.selectbox("N. Piani", sorted(df_r["N_Piani"].unique())
                                  if "N_Piani" in df_r.columns else floors,
                                  key="iso_nP")
        with col_ax2:
            ax_iso = st.multiselect("Assi", ["HNE", "HNN"],
                                    default=["HNE", "HNN"], key="iso_ax")

        # Limite SA globale su tutti i modelli
        @st.cache_data(show_spinner="Calcolo SA ymax globale…")
        def _get_sa_ymax_all(_df_r, _df_th):
            return global_sa_ymax(
                _df_r[_df_r["Asse"].isin(["HNE", "HNN"])], _df_th, get_th
            )
        sa_ymax_all = _get_sa_ymax_all(df_r, df_th)

        st.divider()

        # ── 1. IDR tutti i modelli ───────────────────────────
        with st.expander("1. IDR – tutti i modelli (rigida + isolati)", expanded=True):
            if st.button("▶ Genera IDR Isolamento", key="btn_idr_iso"):
                with st.spinner("Generazione…"):
                    figs = []
                    sub_sum = df_idr_sum[df_idr_sum["N_Piani"] == nP_iso]
                    for ax in ax_iso:
                        figs += plot_idr_isolamento(nP_iso, ax, sub_sum,
                                                    df_idr_th, df_r)
                st.session_state["figs_idr_iso"] = figs

            if "figs_idr_iso" in st.session_state:
                figs = st.session_state["figs_idr_iso"]
                _show_figs(figs, cols=2)
                if figs:
                    st.download_button("⬇️ Scarica ZIP IDR isolamento",
                                       data=_figs_to_zip(figs),
                                       file_name="IDR_ISOLAMENTO.zip",
                                       mime="application/zip",
                                       key="dl_idr_iso")

        # ── 2. Spettri di piano (tutti i modelli) ────────────
        with st.expander("2. Spettri di piano in testa – tutti i modelli"):
            if st.button("▶ Genera Spettri di Piano (iso)", key="btn_sa_iso"):
                with st.spinner("Calcolo spettri SA…"):
                    figs = []
                    for ax in ax_iso:
                        figs += plot_spettri_piano_iso(nP_iso, ax, df_r, df_th,
                                                       sa_ymax_all, top_tags)
                st.session_state["figs_sa_iso"] = figs

            if "figs_sa_iso" in st.session_state:
                figs = st.session_state["figs_sa_iso"]
                _show_figs(figs, cols=2)
                if figs:
                    st.download_button("⬇️ Scarica ZIP spettri isolamento",
                                       data=_figs_to_zip(figs),
                                       file_name="SPETTRI_ISOLAMENTO.zip",
                                       mime="application/zip",
                                       key="dl_sa_iso")

        # ── 3. Confronto PFA / SA_max ────────────────────────
        with st.expander("3. Confronto PFA e SA_max – rigida vs slitte vs isolato"):
            if st.button("▶ Genera grafici confronto", key="btn_cfr"):
                with st.spinner("Calcolo risultati e confronto…"):
                    df_res = build_results_table(df_r, df_th)
                    df_rid = compute_reductions(df_res)
                    figs   = plot_confronto_pfa_sa(df_rid)
                st.session_state["figs_cfr"]  = figs
                st.session_state["df_res_iso"]= df_res
                st.session_state["df_rid_iso"]= df_rid

            if "figs_cfr" in st.session_state:
                figs = st.session_state["figs_cfr"]
                _show_figs(figs, cols=2)
                if figs:
                    st.download_button("⬇️ Scarica ZIP confronto",
                                       data=_figs_to_zip(figs),
                                       file_name="CONFRONTO_PFA_SA.zip",
                                       mime="application/zip",
                                       key="dl_cfr")


# ─────────────────────────────────────────────────────────────
# TAB 3 – TABELLE
# ─────────────────────────────────────────────────────────────
with tab_tab:
    st.header("📊 Tabelle dei risultati")

    gen_col, _ = st.columns([1, 3])
    with gen_col:
        gen_tables_btn = st.button("▶ Genera / aggiorna tabelle", type="primary",
                                    key="btn_gen_tables")

    if gen_tables_btn:
        with st.spinner("Calcolo tabelle…"):
            df_res_t = build_results_table(df_r, df_th)
            df_idr_t = build_idr_table(df_idr_sum)
            df_rid_t = compute_reductions(df_res_t)
            st.session_state["df_res_t"] = df_res_t
            st.session_state["df_idr_t"] = df_idr_t
            st.session_state["df_rid_t"] = df_rid_t

    if "df_res_t" not in st.session_state:
        st.info("Clicca 'Genera tabelle' per calcolare i risultati.")
    else:
        df_res_t = st.session_state["df_res_t"]
        df_idr_t = st.session_state["df_idr_t"]
        df_rid_t = st.session_state["df_rid_t"]

        # ── IDR Max ─────────────────────────────────────────
        st.subheader("IDR Massimo – verifica NTC2018")
        st.caption("🟢 OK_Fragili: IDR ≤ 0.0033 | 🟢 OK_Deformabili: IDR ≤ 0.0050")

        def _color_ok(val):
            return "background-color: #d4edda; color: #155724" if val == "SI" \
                   else "background-color: #f8d7da; color: #721c24"

        if not df_idr_t.empty:
            styled_idr = df_idr_t.style.applymap(
                _color_ok, subset=[c for c in ["OK_Fragili", "OK_Deformabili"]
                                   if c in df_idr_t.columns]
            )
            st.dataframe(styled_idr, use_container_width=True, hide_index=True)
        else:
            st.warning("IDR_Riepilogo non disponibile o vuoto.")

        # ── PFA / SA / Arias ─────────────────────────────────
        st.subheader("PFA, SA Max, Arias – tutti i modelli")
        sort_cols = [c for c in ["N_Piani", "Categoria", "Tis", "Asse", "Nodo_Tag"]
                     if c in df_res_t.columns]
        st.dataframe(
            df_res_t.sort_values(sort_cols) if sort_cols else df_res_t,
            use_container_width=True, hide_index=True,
        )

        # ── Riduzioni % ──────────────────────────────────────
        st.subheader("Riduzioni % rispetto a Base Fissa")
        if not df_rid_t.empty:
            # evidenzia colonne riduzione
            def _color_rid(val):
                if not isinstance(val, (int, float)) or np.isnan(val):
                    return ""
                if val > 50:
                    return "background-color: #d4edda; color: #155724"
                if val > 20:
                    return "background-color: #fff3cd; color: #856404"
                return "background-color: #f8d7da; color: #721c24"

            rid_cols = [c for c in ["Rid_PFA_pct", "Rid_SA_pct"] if c in df_rid_t.columns]
            styled_rid = df_rid_t.style.applymap(_color_rid, subset=rid_cols)
            st.dataframe(styled_rid, use_container_width=True, hide_index=True)
        else:
            st.info("Nessun modello isolato trovato per calcolare le riduzioni.")

        # ── Download ─────────────────────────────────────────
        st.divider()
        st.subheader("⬇️ Scarica risultati")
        excel_bytes = _tables_to_excel(df_res_t, df_idr_t, df_rid_t)
        st.download_button(
            label="📥 Scarica Excel (Risultati + IDR + Riduzioni)",
            data=excel_bytes,
            file_name="RISULTATI_SPETTRIPIANO.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

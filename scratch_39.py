# Requisiti: streamlit
# Puoi installarlo con: pip install streamlit

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Funzione parametri da tabella NTC
@st.cache_data
def get_ap_a_b(T1):
    if T1 <= 0.5:
        return 0.8, 1.4, 5.0
    elif T1 <= 1.0:
        return 0.3, 1.2, 4.0
    else:
        return 0.3, 1.0, 2.5

def Sa_excel_style(Ta, T1, alpha, S, zh):
    a, b, ap = get_ap_a_b(T1)
    h_ratio = 1 + zh
    if Ta < a * T1:
        den = 1 + (ap - 1) * (1 - Ta / (a * T1))**2
        return alpha * S * h_ratio * ap / den
    elif Ta > b * T1:
        den = 1 + (ap - 1) * (1 - Ta / (b * T1))**2
        return alpha * S * h_ratio * ap / den
    else:
        return alpha * S * h_ratio * ap

# Titolo e descrizione
st.title("Spettro di Piano NTC2018 - Interfaccia Interattiva")
st.write("""
Questa app permette di calcolare e visualizzare lo spettro di piano semplificato secondo la Circolare NTC2018.
Modifica i parametri a sinistra per vedere come cambia il grafico.
""")

# Sidebar con parametri
alpha = st.sidebar.slider("α (PGA/g)", 0.1, 1.0, 0.35, 0.01)
S = st.sidebar.slider("S (suolo)", 0.5, 2.5, 1.2, 0.1)
zh = st.sidebar.slider("z/H (quota relativa)", 0.0, 1.0, 0.3, 0.05)
T1 = st.sidebar.slider("T₁ (Periodo struttura)", 0.2, 2.5, 1.0, 0.05)

# Calcolo
Ta_vals = np.linspace(0.05, 3.0, 300)
Sa_vals = [Sa_excel_style(Ta, T1, alpha, S, zh) for Ta in Ta_vals]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ta_vals, Sa_vals, label=f"T₁ = {T1:.2f} s")
ax.set_xlabel("Tₐ (s)")
ax.set_ylabel("Sₐ(Tₐ) [g]")
ax.set_title("Spettro di piano semplificato (NTC2018)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Download CSV
df = pd.DataFrame({"Tₐ (s)": Ta_vals, "Sₐ (g)": Sa_vals})
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Scarica CSV", csv, "spettro_di_piano.csv", "text/csv")

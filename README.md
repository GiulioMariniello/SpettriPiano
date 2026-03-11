---
title: SpettriPiano
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
short_description: Post-processing sismico – Isolamento sismico
---

# SpettriPiano

Post-processing di analisi time-history per edifici con isolamento sismico.

## Utilizzo

1. Carica **COLB.txt** (accelerogramma, 3 colonne in *g*, spazio-separato)
2. Carica l'**Excel master** con i fogli: `Riepilogo`, `TimeHistory`, `IDR_Riepilogo`, `IDR_TimeHistory`
3. Imposta parametri (passo temporale, finestra analisi, smorzamento)
4. Genera grafici e scarica i risultati

## Modalità

### 🏗️ Base Fissa
- Storie di accelerazione alla base (3 direzioni)
- IDR(t) all'interpiano critico per modello e asse
- Storie di accelerazione ai punti di controllo in testa
- Spettri di piano (SA 5%)
- Intensità di Arias e Husid

### 🔵 Isolamento
- IDR per tutti i modelli (rigida + isolati)
- Spettri di piano sovrapposti per categoria
- Confronto PFA e SA_max: base fissa vs slitte vs isolato per diversi T_target
- Tabelle Excel editabili con riduzioni %

## Dipendenze

```
numpy / pandas / matplotlib / scipy / openpyxl / pyrotd / streamlit
```

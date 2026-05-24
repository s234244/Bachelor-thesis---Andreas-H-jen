# Gradient Based FullBins - samlet overblik

## Genererede overblik

- `gb_fullbins_all_summary_runs.csv`
  - Samlet maskinlaesbar tabel med alle summary-runs i denne mappe.
- `gb_fullbins_all_summary_runs.md`
  - Samme data som Markdown-tabel plus top GB full-bins runs.
- `gb_fullbins_aep_plot.png`
  - Plot over AEP for alle GB full-bins optimeringsruns.
- `make_gb_fullbins_overview.py`
  - Scriptet der genererer tabellen og plottet.

Samlet den 2026-05-24. Originalfilerne er ikke flyttet; dette er kun kopier.

## Mapper

- `01_SS_GB_FullBins`
  - Fra `Results_CSV_SS_GB_FullBins`
  - Kørsel: `RunSS_GB_FullBins`
  - Summary timestamp: `20260523_031110`
  - Indhold: SS-layout efterfulgt af Gradient Based med full bins.
  - Resultat: `SS--GB`, seed 1, AEP `715.205 GWh`, forbedring `2.605 GWh`.

- `02_BestSS2S_GB_FullBins`
  - Fra `Results_CSV_BestSS2S_GB_FullBins`
  - Kørsel: `RunBestSS2S_GB_FullBins`
  - Summary timestamp: `20260523_043420`
  - Indhold: bedste SS--2S-layout efterfulgt af Gradient Based med full bins.
  - Resultat: `SS--2S--GB-fullbins`, seed 3, AEP `715.430 GWh`, forbedring `0.435 GWh`.

- `03_SourceLayouts_GB_FullBins`
  - Fra `Results_CSV_SourceLayouts_GB_FullBins`
  - Kørsel: `RunSourceLayouts_GB_FullBins`
  - Summary timestamp: `20260523_054538`
  - Indhold: Gradient Based full-bins optimering fra 4 layoutfamilier med 5 seeds hver.
  - Bedste resultater:
    - `SS--GB-fullbins`: seed 2, AEP `715.238 GWh`
    - `SS--2S--GB-fullbins`: seed 2, AEP `715.477 GWh`
    - `RS--GB-fullbins`: seed 3, AEP `715.633 GWh`
    - `RS--2S--GB-fullbins`: seed 3, AEP `715.813 GWh`

## Ikke inkluderet

- `Results_Gradient_CSV/ss_gradient_iterations_all_20260512_135658.csv`
  - Den ser ud til at være en anden/ældre gradient-kørsel, da AEP-skalaen ligger omkring `72 GWh` i stedet for `701-715 GWh`.

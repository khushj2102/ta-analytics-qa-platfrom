# Turbulence Aware: Analytics & QA Master Platform (Streamlit)

Windows-only Streamlit app for:
- Querying TA data (Athena via ODBC DSN)
- Auto-enrichment: Candidate_Key, flight profiling, heartbeat/trigger classification
- TA flight_id segmentation (TA only)
- FR24 overlay (Athena via ODBC DSN) using TA flights as reference
- D-Tale EDA (separate query tab)
- Local audit logging + run folders + parquet caching

---

## Prerequisites

### 1) Python
- Python **3.12.10** recommended

### 2) ODBC DSNs (hardcoded)
You must have these ODBC DSNs configured locally:

- TA DSN: `ATHENA_MET_PROD`
- FR24 DSN: `ATHENA_FLIGHTRADAR_PROD`

### 3) Airport mapping file
Ensure this file exists:
- `data/airport_data_set_v1.csv`

Required columns:
- `CD_LOCATIONICAO`
- `CD_LOCATIONIATA`

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

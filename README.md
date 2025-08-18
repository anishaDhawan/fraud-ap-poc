# LedgerLock: AP Invoice Fraud Detection POC

LedgerLock is a Python proof-of-concept for detecting fraudulent or anomalous AP invoices in logistics & cold-chain shipping. 

## Features
- Duplicate invoice detection
- Vendor mismatch & unknown vendor flagging (with fuzzy matching)
- Bank account change detection
- Unusual amount flagging
- Optional: BOL mismatch, sanctions check

## Quickstart
```bash
pip install -r requirements.txt
python -m src.main --invoices data/invoices_sample.csv --vendors data/vendor_master.csv --bol data/bol_sample.csv --out fraud_report.xlsx
```

## Roadmap
- Anomaly detection models
- Telematics/IoT checks
- FastAPI + React dashboard
- RBAC, audit logging, SOC2
- Cross-client fraud-shadow network

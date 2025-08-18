"""
Fraud rules for LedgerLock AP Invoice POC.
Each function returns a DataFrame of flagged invoices with reasons.
"""
import pandas as pd
from rapidfuzz import process, fuzz

def flag_duplicates(invoices: pd.DataFrame) -> pd.DataFrame:
    """Flag duplicate invoices (same invoice_id + vendor_name)."""
    dups = invoices.duplicated(subset=["invoice_id", "vendor_name"], keep=False)
    flagged = invoices[dups].copy()
    flagged["reason"] = "Duplicate invoice ID and vendor"
    return flagged[["invoice_id", "vendor_name", "reason"]]

def flag_bank_mismatch(invoices: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    """Flag invoices where bank_account does not match vendor master."""
    merged = invoices.merge(vendors[["vendor_name", "bank_account"]], on="vendor_name", how="left", suffixes=("", "_master"))
    mismatch = merged[merged["bank_account"] != merged["bank_account_master"]]
    mismatch = mismatch.copy()
    mismatch["reason"] = "Bank account mismatch with vendor master"
    return mismatch[["invoice_id", "vendor_name", "reason"]]

def flag_unknown_vendor(invoices: pd.DataFrame, vendors: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """Flag invoices with vendor_name not in vendor master. Suggest closest match if fuzzy match > threshold."""
    known_vendors = set(vendors["vendor_name"])
    def suggest(v):
        match, score, _ = process.extractOne(v, known_vendors, scorer=fuzz.token_sort_ratio)
        return match if score >= threshold else None
    unknown = invoices[~invoices["vendor_name"].isin(known_vendors)].copy()
    unknown["reason"] = unknown["vendor_name"].apply(lambda v: f"Unknown vendor. Suggest: {suggest(v) or 'N/A'}")
    return unknown[["invoice_id", "vendor_name", "reason"]]

def flag_unusual_amounts(invoices: pd.DataFrame, vendors: pd.DataFrame, multiplier: float = 2.0) -> pd.DataFrame:
    """Flag invoices where amount > multiplier × vendor's historical average."""
    avg = invoices.groupby("vendor_name")["amount"].mean().rename("avg_amount")
    merged = invoices.merge(avg, on="vendor_name")
    flagged = merged[merged["amount"] > merged["avg_amount"] * multiplier].copy()
    flagged["reason"] = "Invoice amount unusually high vs vendor average"
    return flagged[["invoice_id", "vendor_name", "reason"]]

def flag_bol_mismatch(invoices: pd.DataFrame, bols: pd.DataFrame) -> pd.DataFrame:
    """Flag invoices with BOL not in known BOL list."""
    known_bols = set(bols["bol_id"])
    mismatch = invoices[~invoices["bol_id"].isin(known_bols)].copy()
    mismatch["reason"] = "BOL not found in BOL master list"
    return mismatch[["invoice_id", "vendor_name", "reason"]]

# TODO: Add sanctions check, weekend anomaly, accessorial outlier, etc.

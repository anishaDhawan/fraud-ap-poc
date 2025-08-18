"""
Parser module for LedgerLock.
Functions to load invoice, vendor, and BOL data from CSVs (and stub for PDF parsing).
"""
import pandas as pd

def load_invoices(csv_path: str) -> pd.DataFrame:
    """Load invoices from CSV into DataFrame."""
    return pd.read_csv(csv_path)

def load_vendors(csv_path: str) -> pd.DataFrame:
    """Load vendor master from CSV into DataFrame."""
    return pd.read_csv(csv_path)

def load_bols(csv_path: str) -> pd.DataFrame:
    """Load BOLs from CSV into DataFrame."""
    return pd.read_csv(csv_path)

def parse_invoice_pdf(pdf_path: str) -> pd.DataFrame:
    """Stub: Parse invoice PDF to DataFrame. TODO: Implement PDF parsing with pdfplumber."""
    # TODO: Implement PDF parsing
    raise NotImplementedError("PDF parsing not implemented yet.")

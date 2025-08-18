"""
Unit tests for LedgerLock fraud rules.
"""
import pandas as pd
import pytest
from src import rules

def test_flag_duplicates():
    df = pd.DataFrame({
        'invoice_id': ['A', 'A', 'B'],
        'vendor_name': ['X', 'X', 'Y'],
        'bank_account': ['1', '1', '2'],
        'invoice_date': ['2025-01-01']*3,
        'amount': [100, 100, 200],
        'bol_id': ['B1', 'B1', 'B2']
    })
    flagged = rules.flag_duplicates(df)
    assert len(flagged) == 2
    assert all(flagged['reason'] == 'Duplicate invoice ID and vendor')

def test_flag_unknown_vendor():
    invoices = pd.DataFrame({
        'invoice_id': ['1'],
        'vendor_name': ['Unknown Vendor'],
        'bank_account': ['1'],
        'invoice_date': ['2025-01-01'],
        'amount': [100],
        'bol_id': ['B1']
    })
    vendors = pd.DataFrame({
        'vendor_name': ['Known Vendor'],
        'bank_account': ['1'],
        'tax_id': ['T1'],
        'mc_or_dot': ['MC1']
    })
    flagged = rules.flag_unknown_vendor(invoices, vendors, threshold=0)
    assert len(flagged) == 1
    assert 'Unknown vendor' in flagged.iloc[0]['reason']

def test_flag_unusual_amounts():
    invoices = pd.DataFrame({
        'invoice_id': ['1', '2', '3'],
        'vendor_name': ['A', 'A', 'A'],
        'bank_account': ['1', '1', '1'],
        'invoice_date': ['2025-01-01']*3,
        'amount': [100, 100, 1000],
        'bol_id': ['B1', 'B2', 'B3']
    })
    vendors = pd.DataFrame({
        'vendor_name': ['A'],
        'bank_account': ['1'],
        'tax_id': ['T1'],
        'mc_or_dot': ['MC1']
    })
    flagged = rules.flag_unusual_amounts(invoices, vendors, multiplier=2.0)
    assert len(flagged) == 1
    assert flagged.iloc[0]['amount'] == 1000
    assert 'unusually high' in flagged.iloc[0]['reason']

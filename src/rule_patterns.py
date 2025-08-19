from typing import List, Dict, Set, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class VendorAnalysisRules:
    """Collection of vendor-specific fraud detection rules."""
    
    @staticmethod
    def detect_bank_changes(df: pd.DataFrame) -> pd.DataFrame:
        """Detect sudden changes in vendor bank details."""
        if not all(field in df.columns for field in ['vendor_id', 'bank_account', 'invoice_date']):
            return pd.DataFrame()
            
        suspicious = []
        for vendor_id in df['vendor_id'].unique():
            vendor_df = df[df['vendor_id'] == vendor_id].sort_values('invoice_date')
            if len(vendor_df['bank_account'].unique()) > 1:
                # Find points where bank account changes
                changes = vendor_df[vendor_df['bank_account'] != vendor_df['bank_account'].shift()]
                suspicious.extend(changes.index.tolist())
        
        return df.loc[suspicious]

    @staticmethod
    def detect_new_large_vendors(df: pd.DataFrame, threshold_amount: float = 10000) -> pd.DataFrame:
        """Identify recently created vendors with large transactions."""
        if not all(field in df.columns for field in ['vendor_id', 'registration_date', 'amount_total']):
            return pd.DataFrame()
            
        recent_threshold = pd.Timestamp.now() - pd.Timedelta(days=90)
        return df[
            (pd.to_datetime(df['registration_date']) > recent_threshold) & 
            (df['amount_total'] > threshold_amount)
        ]

    @staticmethod
    def detect_shared_details(df: pd.DataFrame) -> pd.DataFrame:
        """Find vendors sharing contact details."""
        if not all(field in df.columns for field in ['vendor_id', 'address', 'phone_number', 'email']):
            return pd.DataFrame()
            
        suspicious = []
        for field in ['address', 'phone_number', 'email']:
            if field in df.columns:
                value_counts = df[field].value_counts()
                shared_values = value_counts[value_counts > 1].index
                suspicious.extend(df[df[field].isin(shared_values)].index.tolist())
        
        return df.loc[list(set(suspicious))]

class InvoicePatternRules:
    """Collection of invoice pattern analysis rules."""
    
    @staticmethod
    def detect_round_amounts(df: pd.DataFrame, precision: int = 2) -> pd.DataFrame:
        """Detect suspiciously round invoice amounts."""
        if 'amount_total' not in df.columns:
            return pd.DataFrame()
            
        def is_round(amount: float) -> bool:
            # Check if amount is round at different scales (1s, 10s, 100s, 1000s)
            str_amount = f"{amount:.{precision}f}"
            return (
                str_amount.endswith('0' * precision) or
                str_amount.endswith('5' + '0' * (precision - 1)) or
                str_amount.endswith('00') or
                str_amount.endswith('000')
            )
        
        return df[df['amount_total'].apply(is_round)]

    @staticmethod
    def analyze_invoice_sequence(df: pd.DataFrame) -> pd.DataFrame:
        """Detect irregular invoice number sequences."""
        if not all(field in df.columns for field in ['vendor_id', 'invoice_number']):
            return pd.DataFrame()
            
        suspicious = []
        for vendor_id in df['vendor_id'].unique():
            vendor_df = df[df['vendor_id'] == vendor_id].copy()
            
            # Extract numeric parts of invoice numbers
            vendor_df['num_part'] = vendor_df['invoice_number'].str.extract(r'(\d+)').astype(float)
            vendor_df = vendor_df.sort_values('num_part')
            
            # Check for gaps and irregularities in sequence
            vendor_df['diff'] = vendor_df['num_part'].diff()
            median_diff = vendor_df['diff'].median()
            if not pd.isna(median_diff):
                irregular = vendor_df[
                    (vendor_df['diff'] > median_diff * 2) |
                    (vendor_df['diff'] < median_diff * 0.5)
                ]
                suspicious.extend(irregular.index.tolist())
        
        return df.loc[suspicious]

class PaymentPatternRules:
    """Collection of payment pattern analysis rules."""
    
    @staticmethod
    def detect_split_payments(df: pd.DataFrame, threshold: float = 10000,
                            time_window: int = 5) -> pd.DataFrame:
        """Detect payments potentially split to avoid approval thresholds."""
        if not all(field in df.columns for field in ['vendor_id', 'amount_total', 'invoice_date']):
            return pd.DataFrame()
            
        suspicious = []
        for vendor_id in df['vendor_id'].unique():
            vendor_df = df[df['vendor_id'] == vendor_id].sort_values('invoice_date')
            
            for i, row in vendor_df.iterrows():
                end_date = row['invoice_date'] + pd.Timedelta(days=time_window)
                window_invoices = vendor_df[
                    (vendor_df['invoice_date'] >= row['invoice_date']) &
                    (vendor_df['invoice_date'] <= end_date)
                ]
                
                if (len(window_invoices) > 1 and
                    all(amt < threshold for amt in window_invoices['amount_total']) and
                    window_invoices['amount_total'].sum() > threshold):
                    suspicious.extend(window_invoices.index.tolist())
        
        return df.loc[list(set(suspicious))]

    @staticmethod
    def detect_unusual_timing(df: pd.DataFrame) -> pd.DataFrame:
        """Detect unusual payment timing patterns."""
        if not all(field in df.columns for field in ['invoice_date', 'payment_date']):
            return pd.DataFrame()
            
        df['payment_delay'] = (pd.to_datetime(df['payment_date']) - 
                             pd.to_datetime(df['invoice_date'])).dt.days
        
        # Detect unusually quick or delayed payments
        delay_stats = df['payment_delay'].agg(['mean', 'std'])
        suspicious = df[
            (df['payment_delay'] < delay_stats['mean'] - 2 * delay_stats['std']) |
            (df['payment_delay'] > delay_stats['mean'] + 2 * delay_stats['std'])
        ]
        
        return suspicious

class DocumentAnalysisRules:
    """Collection of document analysis rules."""
    
    @staticmethod
    def check_required_fields(df: pd.DataFrame, required_fields: Set[str]) -> pd.DataFrame:
        """Check for missing required fields."""
        missing_any = pd.Series(False, index=df.index)
        for field in required_fields:
            if field in df.columns:
                missing_any |= df[field].isna() | (df[field] == '')
        
        return df[missing_any]

    @staticmethod
    def detect_inconsistent_formatting(df: pd.DataFrame) -> pd.DataFrame:
        """Detect inconsistencies in data formatting."""
        if 'invoice_number' not in df.columns:
            return pd.DataFrame()
            
        suspicious = []
        
        # Group by vendor and check for consistent invoice number format
        for vendor_id in df['vendor_id'].unique():
            vendor_df = df[df['vendor_id'] == vendor_id]
            
            # Extract patterns from invoice numbers
            patterns = vendor_df['invoice_number'].apply(lambda x: re.sub(r'\d+', '#', str(x)))
            
            # If vendor uses multiple patterns, flag as suspicious
            if len(patterns.unique()) > 1:
                suspicious.extend(vendor_df.index.tolist())
        
        return df.loc[suspicious]

"""
Core functionality for AP Invoice Fraud Detection
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List
from data_validation import DataFieldValidator
from advanced_rules import FraudDetectionEngine

def analyze_invoices(invoice_data: pd.DataFrame,
                    vendor_data: pd.DataFrame = None,
                    payment_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Main entry point for fraud detection analysis.
    Returns enriched DataFrame and analysis report.
    """
    # Initialize the fraud detection engine
    engine = FraudDetectionEngine()
    
    # Prepare and validate data
    df = engine.prepare_data(invoice_data, vendor_data, payment_data)
    
    # Run fraud detection
    results_df, report = engine.detect_fraud(df)
    
    # Add risk levels
    results_df['risk_level'] = pd.cut(
        results_df['risk_score'],
        bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    return results_df, report

def check_duplicate_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """Check for duplicate invoice numbers from the same vendor"""
    duplicates = df[df.duplicated(['vendor_id', 'invoice_id'], keep=False)]
    return duplicates

def check_unusual_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Identify unusually high invoice amounts"""
    mean = df['amount_total'].mean()
    std = df['amount_total'].std()
    threshold = mean + (2 * std)
    return df[df['amount_total'] > threshold]

def check_near_duplicates(df: pd.DataFrame, amount_threshold: float=0.05, 
                         days_threshold: int=7) -> pd.DataFrame:
    """Find near-duplicate invoices within time and amount thresholds"""
    suspicious = []
    df = df.copy()
    df['date_issued'] = pd.to_datetime(df['date_issued'], format='mixed')
    
    for vendor in df['vendor_id'].unique():
        vendor_invoices = df[df['vendor_id'] == vendor]
        for idx1, row1 in vendor_invoices.iterrows():
            for idx2, row2 in vendor_invoices.iterrows():
                if idx1 < idx2:
                    amount_diff = abs(row1['amount_total'] - row2['amount_total']) / row1['amount_total']
                    date_diff = abs((row1['date_issued'] - row2['date_issued']).days)
                    if amount_diff < amount_threshold and date_diff < days_threshold:
                        suspicious.extend([idx1, idx2])
    
    return df.loc[list(set(suspicious))]

def check_payment_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Find invoices paid faster than their payment terms"""
    df = df.copy()
    df['date_issued'] = pd.to_datetime(df['date_issued'], format='mixed')
    df['payment_date'] = pd.to_datetime(df['payment_date'], format='mixed', errors='coerce')
    
    term_to_days = {'NET30': 30, 'NET45': 45, 'NET60': 60}
    df['payment_term_days'] = df['payment_terms'].map(term_to_days)
    df['payment_duration'] = (df['payment_date'] - df['date_issued']).dt.days
    
    suspicious = df[
        (df['status'] == 'Paid') & 
        (df['payment_duration'].notna()) &
        (df['payment_duration'] < df['payment_term_days'] * 0.5)
    ]
    
    return suspicious

def check_weekend_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Find invoices issued on weekends"""
    df = df.copy()
    df['date_issued'] = pd.to_datetime(df['date_issued'], format='mixed')
    return df[df['date_issued'].dt.dayofweek.isin([5, 6])]

def calculate_risk_scores(df: pd.DataFrame, duplicate_invoices: pd.DataFrame,
                         near_duplicates: pd.DataFrame, unusual_amounts: pd.DataFrame,
                         early_payments: pd.DataFrame, weekend_invoices: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk scores for each invoice"""
    def update_risk_factors(df: pd.DataFrame, mask: pd.Series, factor: str, weight: float) -> None:
        """Update risk factors and scores for matching rows."""
        if mask.any():
            # Add comma for existing factors
            has_existing = df.loc[mask, 'risk_factors'].str.len() > 0
            df.loc[mask, 'risk_factors'] = df.loc[mask, 'risk_factors'].where(~has_existing, df.loc[mask, 'risk_factors'] + ', ')
            
            # Add new factor and weight
            df.loc[mask, 'risk_factors'] += factor
            df.loc[mask, 'risk_score'] += weight
    
    # Initialize risk DataFrame
    risk_df = df.copy()
    risk_df['risk_score'] = 0.0
    risk_df['risk_factors'] = ''
    
    weights = {
        'duplicate': 0.35,
        'near_duplicate': 0.25,
        'unusual_amount': 0.20,
        'early_payment': 0.15,
        'weekend': 0.05
    }
    
    def add_risk_factor(row: pd.Series, factor: str, weight: float) -> pd.Series:
        if len(row['risk_factors']) > 0:
            row['risk_factors'] += ', '
        row['risk_factors'] += factor
        row['risk_score'] += weight
        return row
    
    # Create a copy of risk_df to avoid SettingWithCopyWarning
    risk_df = risk_df.copy()
    
    for idx, row in risk_df.iterrows():
        if row['invoice_id'] in duplicate_invoices['invoice_id'].values:
            risk_df.at[idx, 'risk_factors'] = row['risk_factors'] + (',' if row['risk_factors'] else '') + 'Duplicate Invoice'
            risk_df.at[idx, 'risk_score'] += weights['duplicate']
        if row['invoice_id'] in near_duplicates['invoice_id'].values:
            risk_df.loc[idx] = add_risk_factor(risk_df.loc[idx], 'Near-Duplicate Invoice', weights['near_duplicate'])
        if row['invoice_id'] in unusual_amounts['invoice_id'].values:
            risk_df.loc[idx] = add_risk_factor(risk_df.loc[idx], 'Unusual Amount', weights['unusual_amount'])
        if row['invoice_id'] in early_payments['invoice_id'].values:
            risk_df.loc[idx] = add_risk_factor(risk_df.loc[idx], 'Early Payment', weights['early_payment'])
        if row['invoice_id'] in weekend_invoices['invoice_id'].values:
            risk_df.loc[idx] = add_risk_factor(risk_df.loc[idx], 'Weekend Transaction', weights['weekend'])
    
    if risk_df['risk_score'].max() > 0:
        risk_df['risk_score'] = (risk_df['risk_score'] / risk_df['risk_score'].max()) * 100
    
    risk_df['risk_level'] = pd.cut(
        risk_df['risk_score'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    return risk_df

def analyze_vendor_patterns(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze patterns across vendors"""
    vendor_summary = pd.DataFrame()
    vendor_groups = risk_df.groupby('vendor_id')
    
    vendor_summary['total_invoices'] = vendor_groups.size()
    vendor_summary['total_amount'] = vendor_groups['amount_total'].sum()
    vendor_summary['avg_amount'] = vendor_groups['amount_total'].mean()
    vendor_summary['avg_risk_score'] = vendor_groups['risk_score'].mean()
    
    vendor_summary['high_risk_invoices'] = vendor_groups.apply(
        lambda x: len(x[x['risk_level'].isin(['High', 'Very High'])]),
        include_groups=False
    )
    vendor_summary['risk_ratio'] = vendor_summary['high_risk_invoices'] / vendor_summary['total_invoices']
    
    vendor_summary['duplicate_ratio'] = vendor_groups.apply(
        lambda x: len(x[x['risk_factors'].str.contains('Duplicate', na=False)]) / len(x),
        include_groups=False
    )
    vendor_summary['unusual_amount_ratio'] = vendor_groups.apply(
        lambda x: len(x[x['risk_factors'].str.contains('Unusual Amount', na=False)]) / len(x),
        include_groups=False
    )
    vendor_summary['weekend_ratio'] = vendor_groups.apply(
        lambda x: len(x[x['risk_factors'].str.contains('Weekend', na=False)]) / len(x),
        include_groups=False
    )
    
    weights = {
        'risk_ratio': 0.4,
        'duplicate_ratio': 0.3,
        'unusual_amount_ratio': 0.2,
        'weekend_ratio': 0.1
    }
    
    # Calculate vendor risk scores
    vendor_summary['vendor_risk_score'] = (
        vendor_summary['risk_ratio'] * weights['risk_ratio'] * 100 +
        vendor_summary['duplicate_ratio'] * weights['duplicate_ratio'] * 100 +
        vendor_summary['unusual_amount_ratio'] * weights['unusual_amount_ratio'] * 100 +
        vendor_summary['weekend_ratio'] * weights['weekend_ratio'] * 100
    )
    
    # Debug print for top 5 riskiest vendors
    print("\nTop 5 Riskiest Vendors:")
    risky_vendors = vendor_summary.sort_values('vendor_risk_score', ascending=False).head()
    for idx, row in risky_vendors.iterrows():
        print(f"\nVendor {idx}:")
        print(f"Risk Score: {row['vendor_risk_score']:.2f}")
        print(f"Risk Ratio: {row['risk_ratio']:.2f}")
        print(f"Duplicate Ratio: {row['duplicate_ratio']:.2f}")
        print(f"Unusual Amount Ratio: {row['unusual_amount_ratio']:.2f}")
        print(f"Weekend Ratio: {row['weekend_ratio']:.2f}")
    
    vendor_summary['vendor_risk_level'] = pd.cut(
        vendor_summary['vendor_risk_score'],
        bins=[0, 15, 30, 45, 60, 100],  # Lowered thresholds
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    return vendor_summary

def analyze_time_patterns(risk_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze temporal patterns in invoice data"""
    df = risk_df.copy()
    df['date_issued'] = pd.to_datetime(df['date_issued'])
    df['year'] = df['date_issued'].dt.year
    df['month'] = df['date_issued'].dt.month
    df['day_of_week'] = df['date_issued'].dt.dayofweek
    df['week_of_year'] = df['date_issued'].dt.isocalendar().week
    
    monthly_stats = df.groupby(['year', 'month']).agg(
        invoice_count=('invoice_id', 'count'),
        total_amount=('amount_total', 'sum'),
        avg_amount=('amount_total', 'mean'),
        std_amount=('amount_total', 'std'),
        avg_risk=('risk_score', 'mean'),
        max_risk=('risk_score', 'max')
    ).round(2)
    
    monthly_stats['amount_change'] = monthly_stats['total_amount'].pct_change()
    monthly_stats['volume_change'] = monthly_stats['invoice_count'].pct_change()
    monthly_stats['risk_change'] = monthly_stats['avg_risk'].pct_change()
    
    return monthly_stats, df

def get_fraud_summary(df: pd.DataFrame) -> Dict:
    """Get summary of fraud detection results"""
    duplicate_invoices = check_duplicate_invoices(df)
    unusual_amounts = check_unusual_amounts(df)
    near_duplicates = check_near_duplicates(df)
    early_payments = check_payment_terms(df)
    weekend_invoices = check_weekend_transactions(df)
    
    risk_df = calculate_risk_scores(
        df, duplicate_invoices, near_duplicates,
        unusual_amounts, early_payments, weekend_invoices
    )
    
    vendor_summary = analyze_vendor_patterns(risk_df)
    monthly_stats, _ = analyze_time_patterns(risk_df)
    
    return {
        'risk_df': risk_df,
        'duplicate_invoices': duplicate_invoices,
        'unusual_amounts': unusual_amounts,
        'near_duplicates': near_duplicates,
        'early_payments': early_payments,
        'weekend_invoices': weekend_invoices,
        'vendor_summary': vendor_summary,
        'monthly_stats': monthly_stats
    }

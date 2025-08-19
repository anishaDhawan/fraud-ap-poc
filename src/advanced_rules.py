from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_validation import DataFieldValidator

class FraudRule:
    """Base class for fraud detection rules."""
    
    def __init__(self, name: str, required_fields: Set[str], weight: float = 1.0):
        self.name = name
        self.required_fields = required_fields
        self.weight = weight
        self.confidence_score = 1.0  # Adjusted based on available data

    def can_execute(self, df: pd.DataFrame) -> bool:
        """Check if the rule can be executed with available data."""
        return all(field in df.columns for field in self.required_fields)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the rule and return DataFrame with risk factors."""
        raise NotImplementedError("Subclasses must implement execute()")

class RoundAmountRule(FraudRule):
    def __init__(self):
        super().__init__(
            name="Round Amount Detection",
            required_fields={'amount_total'},
            weight=0.3
        )
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect suspiciously round amounts."""
        df['is_round'] = df['amount_total'].apply(
            lambda x: float(x).is_integer() or x % 100 == 0 or x % 1000 == 0
        )
        return df[df['is_round']]

class SequentialInvoiceRule(FraudRule):
    def __init__(self):
        super().__init__(
            name="Sequential Invoice Detection",
            required_fields={'invoice_number', 'vendor_id', 'invoice_date'},
            weight=0.4
        )
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect sequential invoice numbers across different vendors."""
        df['invoice_num_clean'] = df['invoice_number'].str.extract('(\d+)').astype(float)
        
        # Group by date and look for sequential numbers across vendors
        suspicious = []
        for date, group in df.groupby(df['invoice_date']):
            sorted_invoices = group.sort_values('invoice_num_clean')
            for i in range(len(sorted_invoices) - 1):
                if (sorted_invoices.iloc[i+1]['invoice_num_clean'] - 
                    sorted_invoices.iloc[i]['invoice_num_clean'] == 1 and
                    sorted_invoices.iloc[i+1]['vendor_id'] != 
                    sorted_invoices.iloc[i]['vendor_id']):
                    suspicious.extend([sorted_invoices.iloc[i]['invoice_id'],
                                    sorted_invoices.iloc[i+1]['invoice_id']])
        
        return df[df['invoice_id'].isin(suspicious)]

class BankAccountSharingRule(FraudRule):
    def __init__(self):
        super().__init__(
            name="Shared Bank Account Detection",
            required_fields={'vendor_id', 'bank_account'},
            weight=0.7
        )
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect multiple vendors sharing the same bank account."""
        account_groups = df.groupby('bank_account')['vendor_id'].nunique()
        suspicious_accounts = account_groups[account_groups > 1].index
        return df[df['bank_account'].isin(suspicious_accounts)]

class PaymentSplittingRule(FraudRule):
    def __init__(self):
        super().__init__(
            name="Payment Splitting Detection",
            required_fields={'vendor_id', 'amount_total', 'invoice_date'},
            weight=0.5
        )
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect possible payment splitting to avoid approval thresholds."""
        THRESHOLD = 10000  # Example threshold
        TIME_WINDOW = pd.Timedelta(days=5)
        
        suspicious = []
        for vendor in df['vendor_id'].unique():
            vendor_df = df[df['vendor_id'] == vendor].sort_values('invoice_date')
            
            for i, row in vendor_df.iterrows():
                # Look for multiple smaller payments within time window
                window_payments = vendor_df[
                    (vendor_df['invoice_date'] >= row['invoice_date']) &
                    (vendor_df['invoice_date'] <= row['invoice_date'] + TIME_WINDOW)
                ]
                
                if (len(window_payments) > 1 and
                    all(amt < THRESHOLD for amt in window_payments['amount_total']) and
                    window_payments['amount_total'].sum() > THRESHOLD):
                    suspicious.extend(window_payments['invoice_id'].tolist())
        
        return df[df['invoice_id'].isin(suspicious)]

class FraudDetectionEngine:
    """Orchestrates fraud detection rules based on available data."""
    
    def __init__(self):
        self.validator = DataFieldValidator()
        self.rules: List[FraudRule] = [
            RoundAmountRule(),
            SequentialInvoiceRule(),
            BankAccountSharingRule(),
            PaymentSplittingRule(),
            # Add more rules here
        ]
        self.available_rules: List[FraudRule] = []

    def prepare_data(self, invoice_data: pd.DataFrame,
                    vendor_data: Optional[pd.DataFrame] = None,
                    payment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare and validate data for fraud detection."""
        df = invoice_data.copy()
        
        # Merge additional data if available
        if vendor_data is not None:
            df = df.merge(vendor_data, on='vendor_id', how='left')
        if payment_data is not None:
            df = df.merge(payment_data, on='invoice_id', how='left')
        
        # Validate available fields
        self.validator.validate_dataframe(df, 'basic_invoice')
        if vendor_data is not None:
            self.validator.validate_dataframe(df, 'vendor_details')
        if payment_data is not None:
            self.validator.validate_dataframe(df, 'payment_details')
        
        return df

    def detect_fraud(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute all applicable fraud detection rules and return results
        with confidence scores.
        """
        # Determine which rules can be executed
        self.available_rules = [rule for rule in self.rules if rule.can_execute(df)]
        
        # Execute rules and combine results
        results_df = df.copy()
        results_df['risk_factors'] = ''
        results_df['risk_score'] = 0.0
        
        rule_results = {}
        for rule in self.available_rules:
            suspicious_records = rule.execute(df)
            if not suspicious_records.empty:
                for idx in suspicious_records.index:
                    if results_df.loc[idx, 'risk_factors']:
                        results_df.loc[idx, 'risk_factors'] += ', '
                    results_df.loc[idx, 'risk_factors'] += rule.name
                    results_df.loc[idx, 'risk_score'] += rule.weight * rule.confidence_score
                
                rule_results[rule.name] = {
                    'suspicious_count': len(suspicious_records),
                    'confidence_score': rule.confidence_score
                }
        
        # Generate report
        report = {
            'rules_executed': len(self.available_rules),
            'total_rules': len(self.rules),
            'rule_results': rule_results,
            'data_completeness': self.validator.get_completeness_report(),
            'improvement_suggestions': self.validator.suggest_improvements()
        }
        
        return results_df, report

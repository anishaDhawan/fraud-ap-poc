from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_validation import DataFieldValidator
from rule_patterns import (
    VendorAnalysisRules,
    InvoicePatternRules,
    PaymentPatternRules,
    DocumentAnalysisRules
)

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

class Rule(FraudRule):
    """Generic rule class that wraps a detection function."""
    
    def __init__(self, name: str, required_fields: Set[str], 
                 detection_func: callable, weight: float = 1.0):
        super().__init__(name=name, required_fields=required_fields, weight=weight)
        self.detection_func = detection_func
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the wrapped detection function."""
        return self.detection_func(df)

class FraudDetectionEngine:
    """Orchestrates fraud detection rules based on available data."""
    
    def __init__(self):
        self.validator = DataFieldValidator()
        # Initialize rules based on the patterns available in rule_patterns.py
        self.rules: List[FraudRule] = []
        
        # Vendor Analysis Rules
        self.rules.extend([
            Rule("Bank Account Changes", {'vendor_id', 'bank_account', 'invoice_date'},
                 VendorAnalysisRules.detect_bank_changes, 0.8),
            Rule("New Large Vendors", {'vendor_id', 'registration_date', 'amount_total'},
                 VendorAnalysisRules.detect_new_large_vendors, 0.7),
            Rule("Shared Contact Details", {'vendor_id', 'address', 'phone_number', 'email'},
                 VendorAnalysisRules.detect_shared_details, 0.6)
        ])
        
        # Invoice Pattern Rules
        self.rules.extend([
            Rule("Round Amounts", {'amount_total'},
                 InvoicePatternRules.detect_round_amounts, 0.3),
            Rule("Irregular Invoice Sequence", {'vendor_id', 'invoice_number'},
                 InvoicePatternRules.analyze_invoice_sequence, 0.5)
        ])
        
        # Payment Pattern Rules
        self.rules.extend([
            Rule("Split Payments", {'vendor_id', 'amount_total', 'invoice_date'},
                 PaymentPatternRules.detect_split_payments, 0.7),
            Rule("Unusual Payment Timing", {'invoice_date', 'payment_date'},
                 PaymentPatternRules.detect_unusual_timing, 0.4)
        ])
        
        # Document Analysis Rules
        required_fields = {'invoice_id', 'vendor_id', 'amount_total', 'invoice_date'}
        self.rules.extend([
            Rule("Missing Required Fields", required_fields,
                 lambda df: DocumentAnalysisRules.check_required_fields(df, required_fields), 0.6),
            Rule("Inconsistent Formatting", {'vendor_id', 'invoice_number'},
                 DocumentAnalysisRules.detect_inconsistent_formatting, 0.5)
        ])
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
        # Create a fresh copy to avoid chained indexing
        results_df = df.copy()
        results_df.loc[:, 'risk_factors'] = ''
        results_df.loc[:, 'risk_score'] = 0.0
        
        rule_results = {}
        for rule in self.available_rules:
            suspicious_records = rule.execute(df)
            if not suspicious_records.empty:
                # Create mask for suspicious records
                mask = results_df.index.isin(suspicious_records.index)
                
                # Update risk factors
                current_factors = results_df.loc[mask, 'risk_factors']
                has_existing = current_factors.str.len() > 0
                results_df.loc[mask, 'risk_factors'] = (
                    current_factors.where(~has_existing, current_factors + ', ') + rule.name
                )
                
                # Update risk scores
                results_df.loc[mask, 'risk_score'] += rule.weight * rule.confidence_score
                
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

from typing import Dict, List, Set, Optional
import pandas as pd

class DataFieldValidator:
    """Validates and tracks available data fields for fraud detection."""
    
    FIELD_GROUPS = {
        'basic_invoice': {
            'required': {'invoice_id', 'vendor_id', 'amount_total', 'invoice_date'},
            'optional': {'invoice_number', 'description', 'currency'}
        },
        'vendor_details': {
            'required': {'vendor_id'},
            'optional': {
                'vendor_name', 'bank_account', 'tax_id', 'address', 
                'registration_date', 'contact_email', 'phone_number'
            }
        },
        'line_items': {
            'required': {'invoice_id', 'item_description'},
            'optional': {
                'item_quantity', 'unit_price', 'item_total', 
                'product_code', 'unit_of_measure'
            }
        },
        'purchase_order': {
            'required': {'po_number', 'invoice_id'},
            'optional': {
                'po_date', 'delivery_date', 'receiving_date',
                'requester_id', 'department_code'
            }
        },
        'payment_details': {
            'required': {'invoice_id', 'payment_date'},
            'optional': {
                'payment_method', 'payment_account', 'approval_id',
                'approver_id', 'payment_reference'
            }
        },
        'document_metadata': {
            'required': {'invoice_id', 'document_type'},
            'optional': {
                'file_hash', 'creation_date', 'modification_date',
                'author', 'pdf_version', 'scanner_info'
            }
        }
    }

    def __init__(self):
        self.available_fields: Dict[str, Set[str]] = {}
        self.missing_fields: Dict[str, Set[str]] = {}
        self.completeness_scores: Dict[str, float] = {}

    def validate_dataframe(self, df: pd.DataFrame, group: str) -> bool:
        """
        Validates if a dataframe has required fields for a specific group.
        Returns True if all required fields are present.
        """
        if group not in self.FIELD_GROUPS:
            raise ValueError(f"Unknown field group: {group}")

        required = self.FIELD_GROUPS[group]['required']
        optional = self.FIELD_GROUPS[group]['optional']
        all_fields = required.union(optional)
        
        # Check which fields are available
        available = set(df.columns).intersection(all_fields)
        self.available_fields[group] = available
        
        # Track missing fields
        self.missing_fields[group] = required.difference(available)
        
        # Calculate completeness score (including optional fields)
        total_possible = len(all_fields)
        available_count = len(available)
        self.completeness_scores[group] = available_count / total_possible
        
        # Return True if all required fields are present
        return len(self.missing_fields[group]) == 0

    def get_completeness_report(self) -> Dict[str, dict]:
        """
        Returns a detailed report of data completeness and missing fields.
        """
        report = {}
        for group in self.FIELD_GROUPS:
            if group in self.completeness_scores:
                report[group] = {
                    'completeness_score': self.completeness_scores[group],
                    'available_fields': sorted(list(self.available_fields.get(group, set()))),
                    'missing_required': sorted(list(self.missing_fields.get(group, set()))),
                    'missing_optional': sorted(list(
                        self.FIELD_GROUPS[group]['optional'].difference(
                            self.available_fields.get(group, set())
                        )
                    ))
                }
        return report

    def suggest_improvements(self) -> List[str]:
        """
        Generates suggestions for improving fraud detection based on missing fields.
        """
        suggestions = []
        for group, fields in self.missing_fields.items():
            if fields:
                suggestions.append(f"Add {', '.join(fields)} to enable {group} analysis")
        
        # Add suggestions for missing optional fields that enable advanced detection
        for group, group_fields in self.FIELD_GROUPS.items():
            if group in self.available_fields:
                missing_optional = group_fields['optional'].difference(
                    self.available_fields[group]
                )
                if missing_optional:
                    suggestions.append(
                        f"Adding {', '.join(missing_optional)} would enhance "
                        f"{group} fraud detection capabilities"
                    )
        
        return suggestions

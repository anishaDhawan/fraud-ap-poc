import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_invoices(num_invoices=1000, num_vendors=50):
    """
    Generate sample AP invoice data for logistics operations.
    """
    np.random.seed(42)  # For reproducibility
    
    # Vendor details
    vendor_ids = [f'V{str(i).zfill(3)}' for i in range(1, num_vendors + 1)]
    vendor_names = [f'Logistics Provider {i}' for i in range(1, num_vendors + 1)]
    vendor_categories = ['Transportation', 'Warehousing', 'Customs Broker', 'Freight Forwarder', 'Last Mile Delivery']
    
    # Lists for generating realistic data
    payment_terms = ['NET30', 'NET45', 'NET60', 'NET90']
    invoice_status = ['Pending', 'Approved', 'Paid', 'Rejected']
    service_types = ['Road Transport', 'Air Freight', 'Sea Freight', 'Warehousing', 'Customs Clearance', 
                    'Last Mile Delivery', 'Cross-Border', 'Express Delivery']
    
    # Generate base data
    data = {
        'invoice_id': [f'INV-{str(i).zfill(6)}' for i in range(1, num_invoices + 1)],
        'vendor_id': np.random.choice(vendor_ids, num_invoices),
        'vendor_name': [''] * num_invoices,  # Will be filled based on vendor_id
        'vendor_category': [''] * num_invoices,  # Will be filled based on vendor_id
        'date_issued': pd.date_range(start='2024-01-01', end='2025-08-18', periods=num_invoices),
        'date_due': [None] * num_invoices,  # Will be calculated based on payment terms
        'payment_terms': np.random.choice(payment_terms, num_invoices),
        'service_type': np.random.choice(service_types, num_invoices),
        'amount_subtotal': np.random.uniform(1000, 50000, num_invoices).round(2),
        'tax_rate': np.random.choice([0.00, 0.05, 0.10], num_invoices),
        'tax_amount': [0.0] * num_invoices,  # Will be calculated
        'amount_total': [0.0] * num_invoices,  # Will be calculated
        'status': np.random.choice(invoice_status, num_invoices, p=[0.2, 0.3, 0.4, 0.1]),
        'payment_date': [None] * num_invoices,  # Will be filled conditionally
        'invoice_currency': ['USD'] * num_invoices,  # Can be expanded for international operations
        'po_number': [f'PO-{str(random.randint(1000, 9999))}' for _ in range(num_invoices)],
    }
    
    df = pd.DataFrame(data)
    
    # Create vendor mapping
    vendor_mapping = {
        vid: {'name': vname, 'category': np.random.choice(vendor_categories)} 
        for vid, vname in zip(vendor_ids, vendor_names)
    }
    
    # Fill dependent fields
    df['vendor_name'] = df['vendor_id'].map(lambda x: vendor_mapping[x]['name'])
    df['vendor_category'] = df['vendor_id'].map(lambda x: vendor_mapping[x]['category'])
    df['tax_amount'] = (df['amount_subtotal'] * df['tax_rate']).round(2)
    df['amount_total'] = (df['amount_subtotal'] + df['tax_amount']).round(2)
    
    # Calculate due dates based on payment terms
    df['date_due'] = df.apply(lambda x: x['date_issued'] + pd.Timedelta(days=int(x['payment_terms'][3:])), axis=1)
    
    # Add payment dates for 'Paid' invoices
    df.loc[df['status'] == 'Paid', 'payment_date'] = df.loc[df['status'] == 'Paid'].apply(
        lambda x: x['date_due'] - pd.Timedelta(days=random.randint(0, 15)), axis=1
    )
    
    # Inject some anomalies for fraud detection
    num_anomalies = int(num_invoices * 0.05)  # 5% of invoices will have anomalies
    
    # 1. Duplicate invoices with slightly different amounts
    duplicate_indices = np.random.choice(df.index, size=num_anomalies//5, replace=False)
    for idx in duplicate_indices:
        new_row = df.loc[idx].copy()
        new_row['amount_subtotal'] = new_row['amount_subtotal'] * random.uniform(0.95, 1.05)
        new_row['tax_amount'] = (new_row['amount_subtotal'] * new_row['tax_rate']).round(2)
        new_row['amount_total'] = (new_row['amount_subtotal'] + new_row['tax_amount']).round(2)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # 2. Unusually high amounts
    high_amount_indices = np.random.choice(df.index, size=num_anomalies//5, replace=False)
    df.loc[high_amount_indices, 'amount_subtotal'] = df.loc[high_amount_indices, 'amount_subtotal'] * 5
    df.loc[high_amount_indices, 'tax_amount'] = (df.loc[high_amount_indices, 'amount_subtotal'] * 
                                                df.loc[high_amount_indices, 'tax_rate']).round(2)
    df.loc[high_amount_indices, 'amount_total'] = (df.loc[high_amount_indices, 'amount_subtotal'] + 
                                                  df.loc[high_amount_indices, 'tax_amount']).round(2)
    
    # 3. Suspicious payment patterns (paid way too quickly)
    quick_pay_indices = np.random.choice(df[df['status'] == 'Paid'].index, size=num_anomalies//5, replace=False)
    df.loc[quick_pay_indices, 'payment_date'] = df.loc[quick_pay_indices, 'date_issued'] + pd.Timedelta(days=1)
    
    # Sort by date issued
    df = df.sort_values('date_issued').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_invoices()
    
    # Save to CSV in data directory
    df.to_csv('data/sample_invoices.csv', index=False)
    print(f"Generated {len(df)} sample invoices with realistic patterns and injected anomalies.")
    print("Data saved to ../data/sample_invoices.csv")

"""
Report generation for LedgerLock AP Invoice POC.
Exports flagged invoices to Excel/CSV with summary.
"""
import pandas as pd

def export_report(flags: pd.DataFrame, out_path: str) -> None:
    """Export flagged invoices to Excel or CSV. Adds summary sheet if Excel."""
    if out_path.endswith('.xlsx'):
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            flags.to_excel(writer, index=False, sheet_name='Fraud Flags')
            summary = flags['reason'].value_counts().rename_axis('reason').reset_index(name='count')
            summary.to_excel(writer, index=False, sheet_name='Summary')
    else:
        flags.to_csv(out_path, index=False)

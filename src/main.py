"""
CLI entry point for LedgerLock AP Invoice Fraud Detection POC.
"""
import argparse
from src import parser, rules, report
import pandas as pd

def main():
    argp = argparse.ArgumentParser(description="LedgerLock AP Invoice Fraud Detection")
    argp.add_argument('--invoices', required=True, help='Path to invoices CSV')
    argp.add_argument('--vendors', required=True, help='Path to vendor master CSV')
    argp.add_argument('--bol', required=False, help='Path to BOL CSV (optional)')
    argp.add_argument('--out', required=True, help='Output report path (.xlsx or .csv)')
    args = argp.parse_args()

    invoices = parser.load_invoices(args.invoices)
    vendors = parser.load_vendors(args.vendors)
    bols = parser.load_bols(args.bol) if args.bol else pd.DataFrame()

    # Run fraud rules
    flags = pd.concat([
        rules.flag_duplicates(invoices),
        rules.flag_bank_mismatch(invoices, vendors),
        rules.flag_unknown_vendor(invoices, vendors),
        rules.flag_unusual_amounts(invoices, vendors),
        rules.flag_bol_mismatch(invoices, bols) if not bols.empty else pd.DataFrame()
    ], ignore_index=True)

    if flags.empty:
        print("No fraud or anomaly flags detected.")
    else:
        print(f"Flagged {len(flags)} invoices. Reasons:")
        print(flags['reason'].value_counts())
        report.export_report(flags, args.out)
        print(f"Report saved to {args.out}")

if __name__ == "__main__":
    main()

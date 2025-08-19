"""
Streamlit app for AP Invoice Fraud Detection
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from fraud_detection import get_fraud_summary

st.set_page_config(
    page_title="AP Invoice Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("AP Invoice Fraud Detection Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload invoice data (CSV)", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    results = get_fraud_summary(df)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Invoices",
            len(df),
            f"{len(results['risk_df'][results['risk_df']['risk_level'].isin(['High', 'Very High'])]):,} High Risk"
        )
    
    with col2:
        st.metric(
            "Total Amount",
            f"${df['amount_total'].sum():,.2f}",
            f"${results['risk_df'][results['risk_df']['risk_level'].isin(['High', 'Very High'])]['amount_total'].sum():,.2f} High Risk"
        )
    
    with col3:
        st.metric(
            "Unique Vendors",
            df['vendor_id'].nunique(),
            f"{results['vendor_summary'][results['vendor_summary']['vendor_risk_level'].isin(['High', 'Very High'])].shape[0]} High Risk"
        )
    
    with col4:
        avg_risk = results['risk_df']['risk_score'].mean()
        st.metric(
            "Average Risk Score",
            f"{avg_risk:.1f}",
            f"{len(results['duplicate_invoices'])} Duplicates Found"
        )
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Vendor Analysis", "Time Analysis"])
    
    with tab1:
        st.subheader("Risk Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score distribution
            fig = px.histogram(
                results['risk_df'],
                x='risk_score',
                color='risk_level',
                title='Distribution of Risk Scores',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk factors breakdown
            risk_factors = pd.Series([
                factor 
                for factors in results['risk_df']['risk_factors'].str.split(', ')
                for factor in factors
                if isinstance(factors, list)
            ]).value_counts()
            
            fig = px.bar(
                x=risk_factors.values,
                y=risk_factors.index,
                orientation='h',
                title='Risk Factors Breakdown',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # High risk invoices
        st.subheader("High Risk Invoices")
        high_risk = results['risk_df'][
            results['risk_df']['risk_level'].isin(['High', 'Very High'])
        ][['invoice_id', 'vendor_id', 'vendor_name', 'amount_total', 'risk_score', 'risk_factors']]
        st.dataframe(high_risk)
    
    with tab2:
        st.subheader("Vendor Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Vendor risk scatter plot
            fig = px.scatter(
                results['vendor_summary'].reset_index(),
                x='total_amount',
                y='vendor_risk_score',
                size='total_invoices',
                color='vendor_risk_level',
                hover_data=['vendor_id', 'high_risk_invoices'],
                title='Vendor Risk Analysis',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Vendor risk patterns
            pattern_data = results['vendor_summary'][[
                'duplicate_ratio', 'unusual_amount_ratio', 'weekend_ratio'
            ]].mean()
            
            fig = px.bar(
                x=pattern_data.values,
                y=pattern_data.index.str.replace('_ratio', '').str.title(),
                orientation='h',
                title='Average Pattern Occurrence Across Vendors',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # High risk vendors
        st.subheader("High Risk Vendors")
        high_risk_vendors = results['vendor_summary'][
            results['vendor_summary']['vendor_risk_level'].isin(['High', 'Very High'])
        ]
        st.dataframe(high_risk_vendors)
    
    with tab3:
        st.subheader("Time Series Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Invoice volume trend
            monthly_data = results['monthly_stats'].reset_index()
            monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
            
            fig = px.line(
                monthly_data,
                x='date',
                y='invoice_count',
                title='Monthly Invoice Volume',
                template='plotly_white'
            )
            fig.update_xaxes(title='Date', tickformat="%Y-%m")
            fig.update_yaxes(title='Number of Invoices')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk score trend
            fig = px.line(
                monthly_data,
                x='date',
                y='avg_risk',
                title='Average Monthly Risk Score',
                template='plotly_white'
            )
            fig.update_xaxes(title='Date', tickformat="%Y-%m")
            fig.update_yaxes(title='Risk Score')
            st.plotly_chart(fig, use_container_width=True)
        
        # Significant changes
        st.subheader("Significant Pattern Changes")
        significant_changes = monthly_data[
            (abs(monthly_data['amount_change']) > 0.5) |
            (abs(monthly_data['volume_change']) > 0.3) |
            (abs(monthly_data['risk_change']) > 0.4)
        ]
        if len(significant_changes) > 0:
            st.dataframe(
                significant_changes[[
                    'date', 'invoice_count', 'total_amount', 'avg_risk',
                    'amount_change', 'volume_change', 'risk_change'
                ]].set_index('date')
            )
        else:
            st.info("No significant changes detected in the patterns")
else:
    st.info("👆 Upload a CSV file to start the analysis")

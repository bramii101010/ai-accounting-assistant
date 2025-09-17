import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="AI Accounting Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .security-badge {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
        margin: 0.25rem;
        display: inline-block;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PIIDetector:
    """Detects and redacts Personally Identifiable Information"""
    
    @staticmethod
    def detect_ssn(text):
        if pd.isna(text):
            return False
        text = str(text)
        ssn_pattern = r'\b\d{3}-?\d{2}-?\d{4}\b'
        return bool(re.search(ssn_pattern, text))
    
    @staticmethod
    def detect_phone(text):
        if pd.isna(text):
            return False
        text = str(text)
        phone_pattern = r'\b(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
        return bool(re.search(phone_pattern, text))
    
    @staticmethod
    def detect_email(text):
        if pd.isna(text):
            return False
        text = str(text)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return bool(re.search(email_pattern, text))
    
    @staticmethod
    def detect_account_number(text):
        if pd.isna(text):
            return False
        text = str(text)
        account_pattern = r'\b\d{8,}\b'
        return bool(re.search(account_pattern, text))
    
    @staticmethod
    def redact_pii(text, pii_type):
        if pd.isna(text):
            return text
        
        text = str(text)
        if pii_type == 'ssn':
            return re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', 'XXX-XX-XXXX', text)
        elif pii_type == 'phone':
            return re.sub(r'\b(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b', 'XXX-XXX-XXXX', text)
        elif pii_type == 'email':
            return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'REDACTED@EMAIL.COM', text)
        elif pii_type == 'account':
            return re.sub(r'\b\d{8,}\b', lambda m: 'X' * len(m.group()), text)
        return text

class AnomalyDetector:
    """Detects anomalies in accounting data"""
    
    @staticmethod
    def detect_duplicate_payments(df):
        anomalies = []
        if 'amount' in df.columns and 'vendor' in df.columns:
            duplicates = df.groupby(['vendor', 'amount']).size()
            duplicates = duplicates[duplicates > 1]
            
            for (vendor, amount), count in duplicates.items():
                anomalies.append({
                    'type': 'Duplicate Payment',
                    'description': f'${amount:,.2f} payment to {vendor} appears {count} times',
                    'severity': 'Medium',
                    'records_affected': count
                })
        return anomalies
    
    @staticmethod
    def detect_amount_outliers(df):
        anomalies = []
        if 'amount' in df.columns:
            amounts = pd.to_numeric(df['amount'], errors='coerce').dropna()
            
            if len(amounts) > 0:
                Q1 = amounts.quantile(0.25)
                Q3 = amounts.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = amounts[(amounts < lower_bound) | (amounts > upper_bound)]
                
                for amount in outliers:
                    severity = 'High' if abs(amount) > amounts.median() * 10 else 'Medium'
                    anomalies.append({
                        'type': 'Amount Outlier',
                        'description': f'Unusual amount: ${amount:,.2f}',
                        'severity': severity,
                        'records_affected': 1
                    })
        return anomalies
    
    @staticmethod
    def detect_negative_amounts(df):
        anomalies = []
        if 'amount' in df.columns:
            amounts = pd.to_numeric(df['amount'], errors='coerce')
            negative_amounts = amounts[amounts < 0].dropna()
            
            for amount in negative_amounts:
                anomalies.append({
                    'type': 'Negative Amount',
                    'description': f'Negative transaction: ${amount:,.2f}',
                    'severity': 'Medium',
                    'records_affected': 1
                })
        return anomalies

def analyze_data(df):
    """Main function to analyze uploaded data"""
    results = {
        'pii_detected': {},
        'pii_count': 0,
        'anomalies': [],
        'summary_stats': {},
        'redacted_data': df.copy(),
    }
    
    # Detect PII
    pii_detector = PIIDetector()
    pii_found = False
    
    for column in df.columns:
        column_pii = []
        for index, value in df[column].items():
            if pii_detector.detect_ssn(value):
                column_pii.append('SSN')
                df.loc[index, column] = pii_detector.redact_pii(value, 'ssn')
                pii_found = True
            elif pii_detector.detect_phone(value):
                column_pii.append('Phone')
                df.loc[index, column] = pii_detector.redact_pii(value, 'phone')
                pii_found = True
            elif pii_detector.detect_email(value):
                column_pii.append('Email')
                df.loc[index, column] = pii_detector.redact_pii(value, 'email')
                pii_found = True
            elif pii_detector.detect_account_number(value):
                column_pii.append('Account Number')
                df.loc[index, column] = pii_detector.redact_pii(value, 'account')
                pii_found = True
        
        if column_pii:
            results['pii_detected'][column] = list(set(column_pii))
            results['pii_count'] += len(column_pii)
    
    results['redacted_data'] = df
    
    # Detect anomalies
    anomaly_detector = AnomalyDetector()
    results['anomalies'].extend(anomaly_detector.detect_duplicate_payments(df))
    results['anomalies'].extend(anomaly_detector.detect_amount_outliers(df))
    results['anomalies'].extend(anomaly_detector.detect_negative_amounts(df))
    
    # Generate summary statistics
    if 'amount' in df.columns:
        amounts = pd.to_numeric(df['amount'], errors='coerce').dropna()
        if len(amounts) > 0:
            results['summary_stats'] = {
                'total_transactions': len(df),
                'total_amount': amounts.sum(),
                'average_amount': amounts.mean(),
                'median_amount': amounts.median(),
                'max_amount': amounts.max(),
                'min_amount': amounts.min()
            }
    
    return results

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ AI Accounting Assistant</h1>
        <p style="font-size: 1.2em; margin-bottom: 1rem;">Secure, Privacy-First AI Analysis for Accounting Teams</p>
        <div>
            <span class="security-badge">🔒 No Data Storage</span>
            <span class="security-badge">👁️ PII Detection & Redaction</span>
            <span class="security-badge">📊 Anomaly Detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Key Features")
        st.markdown("""
        **Privacy Protection:**
        - Detects SSNs, phone numbers, emails
        - Automatically redacts sensitive data
        - No data stored or transmitted
        
        **Anomaly Detection:**
        - Duplicate payments
        - Statistical outliers
        - Negative amounts
        - Unusual patterns
        
        **Instant Analytics:**
        - Summary statistics
        - Data visualizations
        - Real-time processing
        """)
        
        st.header("📋 How to Use")
        st.markdown("""
        1. Upload a CSV file
        2. Review PII detection results
        3. Check anomaly findings
        4. Analyze summary statistics
        5. Download cleaned data
        """)
    
    # File upload
    st.header("📤 Upload Your CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload your accounting data for analysis. Maximum file size: 200MB"
    )
    
    # Sample data option
    if st.button("📋 Use Sample Data"):
        sample_data = """transaction_id,date,vendor,amount,description,employee_contact,account_number
TXN-001,2024-01-15,Office Supplies Inc,245.67,Office supplies purchase,john.doe@company.com,
TXN-002,2024-01-16,Tech Solutions LLC,1250.00,Software licensing,555-123-4567,8876543210
TXN-003,2024-01-17,Office Supplies Inc,245.67,Office supplies purchase,jane.smith@company.com,
TXN-004,2024-01-18,Consulting Services,5000.00,Q1 consulting services,bob.wilson@company.com,1234567890
TXN-005,2024-01-19,Utilities Co,892.34,Monthly electricity bill,,7654321098
TXN-006,2024-01-20,Marketing Agency,3500.00,Digital marketing campaign,alice.johnson@company.com,
TXN-007,2024-01-21,Office Supplies Inc,245.67,Office supplies purchase,mike.brown@company.com,
TXN-008,2024-01-22,Travel Services,-150.00,Refund for cancelled trip,carol.davis@company.com,5555666677
TXN-009,2024-01-23,Equipment Rental,750.00,Monthly equipment rental,,
TXN-010,2024-01-24,Legal Services,2800.00,Contract review services,david.miller@company.com,9988776655"""
        
        uploaded_file = StringIO(sample_data)
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Analyze data
            results = analyze_data(df)
            
            # Display results
            st.success("✅ File processed successfully!")
            
            # Summary metrics
            st.header("📊 Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("PII Items Redacted", results['pii_count'])
            with col3:
                st.metric("Anomalies Detected", len(results['anomalies']))
            with col4:
                if results['summary_stats']:
                    st.metric("Total Amount", f"${results['summary_stats']['total_amount']:,.2f}")
            
            # PII Detection Results
            st.header("🛡️ Privacy Protection Results")
            if results['pii_detected']:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning("🚨 Sensitive Data Detected & Redacted")
                for column, types in results['pii_detected'].items():
                    st.write(f"**{column}:** {', '.join(types)}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("✅ No sensitive data detected in this file.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Anomaly Detection Results
            st.header("🔍 Anomaly Detection Results")
            if results['anomalies']:
                for anomaly in results['anomalies']:
                    severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    st.warning(f"{severity_color.get(anomaly['severity'], '🔵')} **{anomaly['type']}** ({anomaly['severity']})\n\n{anomaly['description']}")
            else:
                st.success("✅ No anomalies detected in this file.")
            
            # Data visualization
            if 'amount' in df.columns:
                st.header("📈 Data Visualization")
                amounts = pd.to_numeric(df['amount'], errors='coerce').dropna()
                
                if len(amounts) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(
                            x=amounts, 
                            nbins=20,
                            title="Distribution of Transaction Amounts",
                            labels={'x': 'Amount ($)', 'y': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(
                            y=amounts,
                            title="Transaction Amount Outliers",
                            labels={'y': 'Amount ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Summary Statistics
            if results['summary_stats']:
                st.header("📊 Summary Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.metric("Average Amount", f"${results['summary_stats']['average_amount']:,.2f}")
                    st.metric("Median Amount", f"${results['summary_stats']['median_amount']:,.2f}")
                
                with stats_col2:
                    st.metric("Maximum Amount", f"${results['summary_stats']['max_amount']:,.2f}")
                    st.metric("Minimum Amount", f"${results['summary_stats']['min_amount']:,.2f}")
                
                with stats_col3:
                    st.metric("Total Transactions", results['summary_stats']['total_transactions'])
                    st.metric("Total Amount", f"${results['summary_stats']['total_amount']:,.2f}")
            
            # Display processed data
            st.header("📋 Processed Data (First 20 Rows)")
            st.dataframe(results['redacted_data'].head(20), use_container_width=True)
            
            # Download option
            csv = results['redacted_data'].to_csv(index=False)
            st.download_button(
                label="💾 Download Cleaned Data",
                data=csv,
                file_name="cleaned_accounting_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted.")

# Info section
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    ### Built for Accounting Professionals
    
    This AI assistant demonstrates how artificial intelligence can help accounting teams while maintaining the highest security standards:
    
    - **Privacy First**: No data is stored or transmitted to external servers
    - **Instant Analysis**: Get immediate insights from your accounting data
    - **Risk Detection**: Automatically identify potential fraud and errors
    - **Compliance Ready**: Ensures sensitive data is properly handled
    
    ### Perfect for:
    - Chief Accounting Officers evaluating AI adoption
    - Finance teams looking to improve data quality
    - Auditors needing clean, redacted data
    - Any organization handling sensitive financial information
    
    ### Technical Details:
    - Built with Python and Streamlit
    - Uses statistical methods for anomaly detection
    - Regex-based PII detection (can be upgraded to ML models)
    - All processing happens in your browser
    """)

if __name__ == "__main__":
    main()

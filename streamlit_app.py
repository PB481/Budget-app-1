import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Personal Budget Analyzer",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Personal Budget Analyzer")
st.markdown("Upload your transaction data and budget categories to get insights into your finances.")

# --- File Upload Section ---
st.sidebar.header("Upload Your Files")

uploaded_transactions_file = st.sidebar.file_uploader(
    "Upload Transaction File (e.g., ViewTxns_XLS.csv)",
    type=["csv"]
)
uploaded_budget_categories_file = st.sidebar.file_uploader(
    "Upload Budget Categories File (e.g., Budget Category.csv)",
    type=["csv"]
)

# --- Data Loading and Preprocessing ---
transactions_df = None
budget_categories_df = None

if uploaded_transactions_file:
    try:
        transactions_df = pd.read_csv(uploaded_transactions_file)
        # --- Clean column names: strip whitespace and convert to lowercase for robust matching ---
        transactions_df.columns = transactions_df.columns.str.strip()
        st.sidebar.success("Transaction file loaded successfully!")
        st.sidebar.dataframe(transactions_df.head()) # Show a preview
        st.sidebar.info(f"Transaction file columns detected: {list(transactions_df.columns)}") # Debugging info
    except Exception as e:
        st.sidebar.error(f"Error loading transaction file: {e}")

if uploaded_budget_categories_file:
    try:
        budget_categories_df = pd.read_csv(uploaded_budget_categories_file)
        # --- Clean column names: strip whitespace and convert to lowercase for robust matching ---
        budget_categories_df.columns = budget_categories_df.columns.str.strip()
        st.sidebar.success("Budget categories file loaded successfully!")
        st.sidebar.dataframe(budget_categories_df.head()) # Show a preview
        st.sidebar.info(f"Budget categories file columns detected: {list(budget_categories_df.columns)}") # Debugging info
    except Exception as e:
        st.sidebar.error(f"Error loading budget categories file: {e}")

if transactions_df is not None and budget_categories_df is not None:
    st.header("1. Data Processing and Categorization")

    # --- Standardize Column Names (Expected after cleaning) ---
    # Expected transaction columns (original case matters for accessing data)
    EXPECTED_TRANSACTION_COLS = ['Date', 'Description', 'Money In (€)', 'Money Out (€)']
    # Expected budget columns (original case matters for accessing data)
    EXPECTED_BUDGET_COLS = ['Category', 'Keywords'] 

    # Check for expected columns in transactions_df (using the original names)
    # The check now uses the cleaned column names for comparison, but refers to original for user error message
    missing_transaction_cols = [col for col in EXPECTED_TRANSACTION_COLS if col not in transactions_df.columns]
    if missing_transaction_cols:
        st.error(f"Transaction file is missing expected columns: {', '.join(missing_transaction_cols)}. "
                 "Please ensure your transaction file has 'Date', 'Description', 'Money In (€)', and 'Money Out (€)' columns, "
                 "or adjust the code to match your column names exactly, including case and any leading/trailing spaces.")
        st.stop() # Stop execution if critical columns are missing

    # Check for expected columns in budget_categories_df (using the original names)
    missing_budget_cols = [col for col in EXPECTED_BUDGET_COLS if col not in budget_categories_df.columns]
    if missing_budget_cols:
        st.error(f"Budget categories file is missing expected columns: {', '.join(missing_budget_cols)}. "
                 "Please ensure your budget file has 'Category' and 'Keywords' columns, "
                 "or adjust the code to match your column names exactly, including case and any leading/trailing spaces.")
        st.stop() # Stop execution if critical columns are missing
    
    # Convert 'Date' column to datetime objects
    try:
        transactions_df['Date'] = pd.to_datetime(transactions_df['Date'], errors='coerce')
        transactions_df.dropna(subset=['Date'], inplace=True) # Remove rows with invalid dates
    except Exception as e:
        st.error(f"Could not convert 'Date' column to datetime. Please ensure it's in a recognizable date format. Error: {e}")
        st.stop()

    # --- Derive 'Amount' column from 'Money In (€)' and 'Money Out (€)' ---
    # Convert 'Money In (€)' and 'Money Out (€)' to numeric, handling missing values and currency symbols
    # Fill NaN with 0 for calculation
    for col in ['Money In (€)', 'Money Out (€)']:
        transactions_df[col] = pd.to_numeric(
            transactions_df[col].astype(str).str.replace('€', '').str.replace(',', '').str.strip(),
            errors='coerce'
        ).fillna(0)

    # *** FIX: Changed calculation to correctly interpret Money Out (€) as a negative value ***
    transactions_df['Amount'] = transactions_df['Money In (€)'] + transactions_df['Money Out (€)'] 
    transactions_df.dropna(subset=['Amount'], inplace=True) # Drop rows where calculated Amount is not a number

    # Determine 'Type' (Income or Expense) based on Amount
    transactions_df['Type'] = transactions_df['Amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')

    # Ensure 'Description' column is string type for keyword matching
    transactions_df['Description'] = transactions_df['Description'].astype(str).fillna('')

    # --- Transaction Mapping Function ---
    def map_transaction_to_category(description, budget_df):
        """
        Maps a transaction description to a budget category based on keywords.
        Assumes budget_df has 'Category' and 'Keywords' columns.
        'Keywords' should be a comma-separated string of terms for a category.
        """
        description_lower = description.lower()
        for index, row in budget_df.iterrows():
            category_keywords = row.get('Keywords') 
            if pd.isna(category_keywords): 
                continue
            keywords_list = [k.strip().lower() for k in str(category_keywords).split(',') if k.strip()]
            for keyword in keywords_list:
                if keyword in description_lower:
                    return row['Category'] # Return the Category name directly
        return "Uncategorized" # If no keyword matches

    # Apply mapping
    transactions_df['Mapped_Category'] = transactions_df['Description'].apply(
        lambda x: map_transaction_to_category(x, budget_categories_df)
    )

    st.subheader("Mapped Transactions Preview")
    st.dataframe(transactions_df.head())

    # --- Time Analysis of Money In and Out ---
    st.header("2. Time Analysis of Money In and Out")

    # Aggregate by month for time analysis
    transactions_df['YearMonth'] = transactions_df['Date'].dt.to_period('M').astype(str)
    monthly_summary = transactions_df.groupby(['YearMonth', 'Type'])['Amount'].sum().unstack(fill_value=0)
    
    # Ensure both 'Income' and 'Expense' columns exist, fill with 0 if not
    if 'Income' not in monthly_summary.columns:
        monthly_summary['Income'] = 0
    if 'Expense' not in monthly_summary.columns:
        monthly_summary['Expense'] = 0

    # For expenses, make amount positive for better visualization
    monthly_summary['Expense_Abs'] = monthly_summary['Expense'].abs()
    
    # Sort by month
    monthly_summary = monthly_summary.sort_index()

    st.subheader("Monthly Income and Expenses Over Time")
    fig_time = px.line(
        monthly_summary,
        x=monthly_summary.index,
        y=['Income', 'Expense_Abs'], # Plot absolute expense for positive values
        labels={'value': 'Amount (€)', 'YearMonth': 'Month'},
        title='Income vs. Expenses Over Time',
        color_discrete_map={'Income': 'green', 'Expense_Abs': 'red'}
    )
    fig_time.update_layout(hovermode="x unified")
    st.plotly_chart(fig_time, use_container_width=True)

    # --- Key Expenses Tracking ---
    st.header("3. Key Expenses Tracking")

    expenses_df = transactions_df[transactions_df['Type'] == 'Expense'].copy()
    
    if not expenses_df.empty:
        # Group by Mapped_Category and sum absolute amounts for expenses
        expense_by_category = expenses_df.groupby('Mapped_Category')['Amount'].sum().abs().sort_values(ascending=False)

        st.subheader("Expense Breakdown by Category")
        fig_expenses = px.pie(
            names=expense_by_category.index,
            values=expense_by_category.values,
            title='Distribution of Expenses by Category',
            hole=0.3
        )
        st.plotly_chart(fig_expenses, use_container_width=True)

        st.subheader("Top Expenses")
        top_n = st.slider("Show Top N Categories", 5, 20, 10)
        fig_top_expenses = px.bar(
            x=expense_by_category.head(top_n).index,
            y=expense_by_category.head(top_n).values,
            labels={'x': 'Category', 'y': 'Total Expense (€)'},
            title=f'Top {top_n} Expense Categories',
            color=expense_by_category.head(top_n).values, # Color based on value
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_top_expenses, use_container_width=True)
    else:
        st.info("No expenses found in the uploaded transactions to analyze.")


    # --- Downloadable Reporting in HTML ---
    st.header("4. Downloadable Financial Report (HTML)")

    def generate_html_report(transactions_df, monthly_summary, expense_by_category):
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .income {{ color: green; }}
                .expense {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Financial Report - {datetime.now().strftime('%Y-%m-%d')}</h1>

            <h2>Monthly Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Income (€)</th>
                        <th>Expenses (€)</th>
                        <th>Net Balance (€)</th>
                    </tr>
                </thead>
                <tbody>
        """
        # Ensure monthly_summary has 'Income' and 'Expense_Abs' columns before iterating
        # This prevents errors if monthly_summary is empty or missing columns
        if not monthly_summary.empty:
            for index, row in monthly_summary.iterrows():
                net_balance = row.get('Income', 0) - row.get('Expense_Abs', 0)
                html_content += f"""
                        <tr>
                            <td>{index}</td>
                            <td>{row.get('Income', 0):.2f}</td>
                            <td>{row.get('Expense_Abs', 0):.2f}</td>
                            <td class="{'income' if net_balance >= 0 else 'expense'}">{net_balance:.2f}</td>
                        </tr>
                """
        else:
            html_content += """
                    <tr>
                        <td colspan="4">No monthly summary data to display.</td>
                    </tr>
            """
        html_content += """
                </tbody>
            </table>

            <h2>Expense Breakdown by Category</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Total Expense (€)</th>
                    </tr>
                </thead>
                <tbody>
        """
        if not expense_by_category.empty: # Check if expense_by_category is not empty
            for category, amount in expense_by_category.items():
                html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{amount:.2f}</td>
                    </tr>
                """
        else:
            html_content += """
                    <tr>
                        <td colspan="2">No expenses to display.</td>
                    </tr>
            """
        html_content += """
                </tbody>
            </table>

            <h2>All Transactions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Description</th>
                        <th>Amount (€)</th>
                        <th>Type</th>
                        <th>Mapped Category</th>
                    </tr>
                </thead>
                <tbody>
        """
        # Convert DataFrame to HTML table (can be customized)
        if not transactions_df.empty:
            transactions_html_table = transactions_df[['Date', 'Description', 'Amount', 'Type', 'Mapped_Category']].to_html(index=False, float_format='%.2f')
            html_content += transactions_html_table.replace('<thead>', '').replace('</thead>', '').replace('<tbody>', '').replace('</tbody>', '').replace('<table>', '').replace('</table>', '')
        else:
            html_content += """
                    <tr>
                        <td colspan="5">No transactions to display.</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        return html_content
    
    # Check before generating report
    if not transactions_df.empty:
        html_report = generate_html_report(transactions_df, monthly_summary, expense_by_category)
        
        st.download_button(
            label="Download Full Financial Report as HTML",
            data=html_report,
            file_name="financial_report.html",
            mime="text/html"
        )
    else:
        st.info("Upload transaction data to generate a report.")

    # --- Projected Analysis (Machine Learning Algo Placeholder) ---
    st.header("5. Projected Analysis of Money In and Out")
    st.info("This section demonstrates a basic linear regression for projection. For more robust financial forecasting, consider using dedicated time-series models (e.g., ARIMA, Prophet) and more historical data.")

    # Prepare data for projection
    # Use the monthly summary, assuming 'Expense_Abs' as the target to predict.
    # We will predict future expenses.
    
    # Create numerical time index (e.g., months since start)
    monthly_summary['time_idx'] = np.arange(len(monthly_summary))

    # Features (X) is the time index, Target (y) is the absolute expenses
    X = monthly_summary[['time_idx']]
    y = monthly_summary['Expense_Abs']

    if len(X) > 1: # Need at least 2 data points for linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Predict future months (e.g., next 3 months)
        last_month_idx = monthly_summary['time_idx'].max()
        future_months = 3
        future_time_idx = np.array([[last_month_idx + i + 1] for i in range(future_months)])
        
        projected_expenses = model.predict(future_time_idx)

        st.subheader("Projected Monthly Expenses (Next 3 Months)")
        projected_df = pd.DataFrame({
            'Month': [f"Projection Month {i+1}" for i in range(future_months)],
            'Projected Expense (€)': projected_expenses
        })
        st.dataframe(projected_df)

        # Combine historical and projected data for visualization
        historical_expenses = monthly_summary[['YearMonth', 'Expense_Abs']].rename(columns={'Expense_Abs': 'Amount'})
        historical_expenses['Type'] = 'Historical Expense'

        projected_data_for_plot = pd.DataFrame({
            'YearMonth': [f"{datetime.strptime(monthly_summary.index[-1], '%Y-%m').replace(day=1) + pd.DateOffset(months=i+1):%Y-%m}" for i in range(future_months)],
            'Amount': projected_expenses,
            'Type': 'Projected Expense'
        })
        
        combined_expenses = pd.concat([historical_expenses, projected_data_for_plot])

        fig_projection = px.line(
            combined_expenses,
            x='YearMonth',
            y='Amount',
            color='Type',
            title='Historical vs. Projected Monthly Expenses',
            labels={'Amount': 'Amount (€)', 'YearMonth': 'Month'},
            color_discrete_map={'Historical Expense': 'red', 'Projected Expense': 'orange'}
        )
        st.plotly_chart(fig_projection, use_container_width=True)

    else:
        st.info("Not enough data to perform a meaningful projection (need at least 2 months of data).")

else:
    st.info("Please upload both your Transaction File and Budget Categories File to get started.")


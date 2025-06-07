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
    "Upload Transaction File (ViewTxns_XLS07-06-2025.xls - ViewTxns_XLS.csv)",
    type=["csv"]
)
uploaded_budget_categories_file = st.sidebar.file_uploader(
    "Upload Budget Categories File (Budget Category.xlsx - Sheet1.csv)",
    type=["csv"]
)

# --- Data Loading and Preprocessing ---
transactions_df = None
budget_categories_df = None

if uploaded_transactions_file:
    try:
        transactions_df = pd.read_csv(uploaded_transactions_file)
        st.sidebar.success("Transaction file loaded successfully!")
        st.sidebar.dataframe(transactions_df.head()) # Show a preview
    except Exception as e:
        st.sidebar.error(f"Error loading transaction file: {e}")

if uploaded_budget_categories_file:
    try:
        budget_categories_df = pd.read_csv(uploaded_budget_categories_file)
        st.sidebar.success("Budget categories file loaded successfully!")
        st.sidebar.dataframe(budget_categories_df.head()) # Show a preview
    except Exception as e:
        st.sidebar.error(f"Error loading budget categories file: {e}")

if transactions_df is not None and budget_categories_df is not None:
    st.header("1. Data Processing and Categorization")

    # --- Standardize Column Names (User might need to adjust these) ---
    # Assuming standard column names for transactions: Date, Description, Amount
    # Assuming standard column names for budget: Category, Code, Keywords
    # User will need to adjust these if their CSVs have different headers
    
    # Common column names expected
    EXPECTED_TRANSACTION_COLS = ['Date', 'Description', 'Amount']
    EXPECTED_BUDGET_COLS = ['Category', 'Code'] # Keywords is assumed for mapping

    # Check for expected columns in transactions_df
    missing_transaction_cols = [col for col in EXPECTED_TRANSACTION_COLS if col not in transactions_df.columns]
    if missing_transaction_cols:
        st.error(f"Transaction file is missing expected columns: {', '.join(missing_transaction_cols)}. "
                 "Please ensure your transaction file has 'Date', 'Description', and 'Amount' columns, "
                 "or adjust the code to match your column names.")
        st.stop() # Stop execution if critical columns are missing

    # Check for expected columns in budget_categories_df
    missing_budget_cols = [col for col in EXPECTED_BUDGET_COLS if col not in budget_categories_df.columns]
    if missing_budget_cols:
        st.error(f"Budget categories file is missing expected columns: {', '.join(missing_budget_cols)}. "
                 "Please ensure your budget file has 'Category' and 'Code' columns, "
                 "and optionally a 'Keywords' column for automatic mapping, "
                 "or adjust the code to match your column names.")
        st.stop() # Stop execution if critical columns are missing

    # Convert 'Date' column to datetime objects
    try:
        transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
    except Exception as e:
        st.error(f"Could not convert 'Date' column to datetime. Please ensure it's in a recognizable date format. Error: {e}")
        st.stop()

    # Convert 'Amount' to numeric, handling potential non-numeric entries
    try:
        transactions_df['Amount'] = pd.to_numeric(transactions_df['Amount'], errors='coerce')
        transactions_df.dropna(subset=['Amount'], inplace=True) # Drop rows where Amount is not a number
    except Exception as e:
        st.error(f"Could not convert 'Amount' column to numeric. Please ensure it contains valid numbers. Error: {e}")
        st.stop()

    # Determine 'Type' (Income or Expense) based on Amount
    transactions_df['Type'] = transactions_df['Amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')

    # Ensure 'Description' column is string type for keyword matching
    transactions_df['Description'] = transactions_df['Description'].astype(str).fillna('')

    # --- Transaction Mapping Function ---
    def map_transaction_to_category(description, budget_df):
        """
        Maps a transaction description to a budget category code based on keywords.
        Assumes budget_df has 'Category', 'Code', and 'Keywords' columns.
        'Keywords' should be a comma-separated string of terms for a category.
        """
        description_lower = description.lower()
        for index, row in budget_df.iterrows():
            category_keywords = row.get('Keywords') # .get() for safer access
            if pd.isna(category_keywords): # Handle NaN keywords
                continue
            keywords_list = [k.strip().lower() for k in str(category_keywords).split(',') if k.strip()]
            for keyword in keywords_list:
                if keyword in description_lower:
                    return row['Code'] # Return the Code for the matched category
        return "Uncategorized" # If no keyword matches

    # Create a dummy 'Keywords' column if it doesn't exist, for demonstration purposes.
    # In a real scenario, the user *must* provide this column in their budget file.
    if 'Keywords' not in budget_categories_df.columns:
        st.warning("No 'Keywords' column found in Budget Categories file. "
                   "Automatic mapping will be limited. Please consider adding a 'Keywords' "
                   "column to your 'Budget Category.xlsx - Sheet1.csv' file for better mapping. "
                   "For now, only a simple direct match (if description exactly matches category) will occur.")
        # Basic fallback: if description exactly matches category name (less robust)
        budget_categories_df['Keywords'] = budget_categories_df['Category'].str.lower()
        
    # Apply mapping
    transactions_df['Budget_Code'] = transactions_df['Description'].apply(
        lambda x: map_transaction_to_category(x, budget_categories_df)
    )

    # Merge with budget categories to get category names
    transactions_df = pd.merge(
        transactions_df,
        budget_categories_df[['Code', 'Category']],
        left_on='Budget_Code',
        right_on='Code',
        how='left',
        suffixes=('_txn', '_budget')
    ).drop(columns=['Code_budget']) # Drop the redundant 'Code_budget' column

    # Rename 'Category_budget' to a more descriptive name if it exists, else use 'Uncategorized'
    transactions_df['Mapped_Category'] = transactions_df['Category_budget'].fillna('Uncategorized')
    transactions_df.drop(columns=['Category_txn', 'Budget_Code'], inplace=True, errors='ignore') # Clean up temp cols

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
        for index, row in monthly_summary.iterrows():
            net_balance = row['Income'] - row['Expense_Abs']
            html_content += f"""
                    <tr>
                        <td>{index}</td>
                        <td>{row['Income']:.2f}</td>
                        <td>{row['Expense_Abs']:.2f}</td>
                        <td class="{'income' if net_balance >= 0 else 'expense'}">{net_balance:.2f}</td>
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
        if not expense_by_category.empty:
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
        transactions_html_table = transactions_df[['Date', 'Description', 'Amount', 'Type', 'Mapped_Category']].to_html(index=False, float_format='%.2f')
        html_content += transactions_html_table.replace('<thead>', '').replace('</thead>', '').replace('<tbody>', '').replace('</tbody>', '').replace('<table>', '').replace('</table>', '')
        
        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        return html_content

    html_report = generate_html_report(transactions_df, monthly_summary, expense_by_category)
    
    st.download_button(
        label="Download Full Financial Report as HTML",
        data=html_report,
        file_name="financial_report.html",
        mime="text/html"
    )

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


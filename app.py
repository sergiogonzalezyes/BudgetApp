import streamlit as st
import pdfplumber
import re
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
import openai
import os

##############################
# Configuration & Title
##############################
st.set_page_config(layout="wide")
st.title("Bank Statement Parser with AI Categorization & Filtering (Cached)")

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    st.error("The OPENAI_API_KEY environment variable is not set.")
else:
    openai.api_key = api_key

##############################
# Helper Functions
##############################

def parse_amount_trailing_minus(amt_str: str) -> float:
    """Converts strings like '12.98-' or '1,200.00-' to float."""
    amt_str = amt_str.strip().replace(",", "")
    negative = amt_str.endswith("-")
    if negative:
        amt_str = amt_str[:-1]
    value = float(amt_str)
    return -value if negative else value

def detect_statement_year(lines):
    """
    Searches for a statement period in the form "MM/DD/YYYY - MM/DD/YYYY"
    and returns the year from the first date. Defaults to 2025.
    """
    pattern = re.compile(r"(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})")
    for line in lines:
        match = pattern.search(line)
        if match:
            start_date = match.group(1)  # e.g. "09/01/2024"
            year_str = start_date.split("/")[-1]
            return int(year_str)
    return 2025

def extract_transactions_with_accounts(lines, source_label="", statement_year=2025):
    """
    Reads each line from the PDF and:
      - Detects if it's an 'account switch' line (SAVINGS or CHECKING).
      - If it's a transaction line, parse it and assign current_account.
      - If a line doesn't match a transaction pattern (and doesn't start with a date),
        it's appended to the previous transaction's Description.
    Adds a new column 'Source' and computes the Parsed Date using statement_year.
    """
    transactions = []
    current_account = None

    account_switch_pattern = re.compile(
        r"^(\d{2}/\d{2}).*?\s+(SAVINGS|CHECKING).*?([\d,]+\.\d{2}-?)\s*$",
        re.IGNORECASE
    )
    transaction_pattern = re.compile(
        r"^(\d{2}/\d{2})\s+(.*?)\s+([\d,]+\.\d{2}-?)\s+([\d,]+\.\d{2}-?)\s*$"
    )
    skip_keywords = (
        "Ending Balance",
        "Dividends Paid",
        "Acct Service Fee",
        "Total Shares Balance",
        "Account Balance Summary",
        "CONTINUED ON PAGE"
    )

    for line in lines:
        original_line = line.strip()
        if not original_line:
            continue
        if any(keyword in original_line for keyword in skip_keywords):
            continue
        m_account = account_switch_pattern.match(original_line)
        if m_account:
            date_str = m_account.group(1)
            new_account = m_account.group(2).upper()
            raw_balance = m_account.group(3)
            current_account = new_account
            bal = parse_amount_trailing_minus(raw_balance)
            transactions.append([date_str, "Beginning Balance", 0.0, bal, current_account, source_label])
            continue
        m_txn = transaction_pattern.match(original_line)
        if m_txn:
            date_str = m_txn.group(1)
            desc = m_txn.group(2).strip()
            raw_amt = m_txn.group(3)
            raw_bal = m_txn.group(4)
            amt = parse_amount_trailing_minus(raw_amt)
            bal = parse_amount_trailing_minus(raw_bal)
            acct = current_account if current_account else ""
            transactions.append([date_str, desc, amt, bal, acct, source_label])
        else:
            if transactions and not re.match(r"^\d{2}/\d{2}", original_line):
                transactions[-1][1] += " " + original_line

    df = pd.DataFrame(transactions, columns=[
        "Transaction Date",
        "Description",
        "Transaction Amount",
        "Balance",
        "Account",
        "Source"
    ])
    df["Statement Year"] = statement_year
    df["Parsed Date"] = pd.to_datetime(
        df["Transaction Date"] + "/" + df["Statement Year"].astype(str),
        format="%m/%d/%Y",
        errors="coerce"
    )
    return df

def categorize_transactions_ai(df: pd.DataFrame, debug_mode=False) -> pd.DataFrame:
    """
    Uses OpenAI to categorize each transaction's Description.
    Processes all rows.
    """
    df["Category"] = ""
    known_categories = ["Groceries", "Dining", "Gas", "Rent", "Utilities", 
                        "Online Services", "Entertainment", "Transfer", "ATM", 
                        "Credit Card", "Investment", "Food", "Other"]
    for idx, row in df.iterrows():
        desc = row["Description"]
        try:
            prompt = f"""
Please categorize the following bank transaction description into one of these categories:
Categories: {", ".join(known_categories)}
Description: {desc}
Return only the category name exactly (no extra text). If none fit, return 'Other'.
"""
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            category = response.choices[0].message['content'].strip()
            if debug_mode:
                print(f"Desc: {desc}\nResponse: {category}\n---")
            df.at[idx, "Category"] = category
        except Exception as e:
            df.at[idx, "Category"] = "Other"
            if debug_mode:
                print(f"Error categorizing '{desc}': {e}")
    df["Category"] = df["Category"].replace("", "Uncategorized")
    return df

# Cache the expensive AI categorization step.
@st.cache_data(show_spinner=True)
def cached_categorize_transactions(df_csv: str, debug_mode: bool) -> str:
    """
    Accepts a CSV string of the transactions, categorizes them via OpenAI,
    and returns the categorized dataframe as CSV.
    """
    df = pd.read_csv(io.StringIO(df_csv))
    df = categorize_transactions_ai(df, debug_mode=debug_mode)
    return df.to_csv(index=False)

def initialize_chatbot(dataframe: pd.DataFrame) -> str:
    """Converts the dataframe into a CSV string."""
    return dataframe.to_csv(index=False)

def chat_with_dataframe(df_str: str, key: str):
    st.markdown("#### Chat with your filtered data")
    user_query = st.text_input("Ask a question about this data:", key=key)
    if user_query:
        prompt = f"""
You are a helpful assistant that can analyze CSV data.

Here is the CSV data:
{df_str}

You are a helpful assistant that can answer questions about this data.
Pretend you are a financial advisor helping someone analyze their finances to help them hit their financial goals.
Based on the above data, please answer the following question:

{user_query}
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message['content'].strip()
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")

##############################
# Main Streamlit Application
##############################

st.sidebar.header("Upload PDF(s)")
uploaded_files = st.sidebar.file_uploader("Upload your bank statement PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    df_list = []
    for uploaded_file in uploaded_files:
        source_label = uploaded_file.name.split('.')[0]
        pdf_file = io.BytesIO(uploaded_file.read())
        with pdfplumber.open(pdf_file) as pdf:
            raw_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        lines = raw_text.split("\n")
        statement_year = detect_statement_year(lines)
        df_temp = extract_transactions_with_accounts(lines, source_label=source_label, statement_year=statement_year)
        df_list.append(df_temp)
    df_full = pd.concat(df_list, ignore_index=True)

    # Cache the categorization step so it doesn't re-run on filter changes.
    df_full_csv = df_full.to_csv(index=False)
    debug_mode = st.sidebar.checkbox("Debug Categorization Responses?", value=False)
    categorized_csv = cached_categorize_transactions(df_full_csv, debug_mode)
    df_full = pd.read_csv(io.StringIO(categorized_csv))

    # Sidebar Filters
    st.sidebar.subheader("Filters")
    all_accounts = sorted(df_full["Account"].dropna().unique())
    if not all_accounts:
        all_accounts = ["SAVINGS", "CHECKING"]
    selected_accounts = st.sidebar.multiselect("Select account(s):", options=all_accounts, default=all_accounts)
    all_categories = sorted(df_full["Category"].unique())
    selected_categories = st.sidebar.multiselect("Select category(ies):", options=all_categories, default=all_categories)
    
    df_filtered = df_full[
        df_full["Account"].isin(selected_accounts) &
        df_full["Category"].isin(selected_categories)
    ].copy()
    df_filtered.sort_values("Parsed Date", inplace=True)

    st.subheader("Filtered Transactions")
    st.write(f"Showing {len(df_filtered)} transactions (out of {len(df_full)} total).")
    st.dataframe(df_filtered)

    if not df_filtered.empty:
        # Bar Chart: Transaction Amount over time, color by Category
        fig_bar = px.bar(
            df_filtered,
            x="Parsed Date",
            y="Transaction Amount",
            color="Category",
            title="Transaction Amount Over Time (Filtered)",
            hover_data=["Description", "Source", "Account"]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Line Chart: Running Balance over time, separate lines for each Account
        fig_line = px.line(
            df_filtered,
            x="Parsed Date",
            y="Balance",
            color="Account",
            markers=True,
            title="Running Balance Over Time (Filtered)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    df_filtered_str = initialize_chatbot(df_filtered)
    chat_with_dataframe(df_filtered_str, key="chat_filtered")
else:
    st.info("Please upload one or more PDF files on the left sidebar to parse transactions.")

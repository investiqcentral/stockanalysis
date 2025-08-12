import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.set_page_config(page_title='Stock Valuation Calculator', layout='wide', page_icon="./Image/logo.png")

#Font Styles#
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&display=swap');
    * {
        font-family: 'Barlow', sans-serif !important;
    }
    .streamlit-expanderContent {
        font-family: 'Barlow', sans-serif !important;
    }
    .stMarkdown {
        font-family: 'Barlow', sans-serif !important;
    }
    p {
        font-family: 'Barlow', sans-serif !important;
    }
    div {
        font-family: 'Barlow', sans-serif !important;
    }
    .stDataFrame {
        font-family: 'Barlow', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Main App Title and Description ---
st.title("💰 Stock Valuation Calculator")
st.markdown("Use this app to estimate the intrinsic value of a stock using the **Discounted Cash Flow (DCF)** method. The app fetches financial data from Yahoo Finance.")

# --- User Input Section ---
st.write("Please provide the necessary inputs for the valuation model.")

tick_col1,tick_col2,tick_col3 = st.columns([1,1,1])
with tick_col1:
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT):").upper()

# --- Functions for Suggestions ---
@st.cache_data(ttl=3600)
def get_fcf_growth(ticker):
    """Calculates historical FCF growth for a suggestion."""
    try:
        stock = yf.Ticker(ticker)
        cash_flow = stock.cashflow
        if cash_flow.empty:
            return None
        
        fcf_data = cash_flow.loc['Free Cash Flow'].head(4)
        if len(fcf_data) < 2:
            return None

        growth_rates = []
        for i in range(len(fcf_data) - 1):
            growth = (fcf_data.iloc[i] - fcf_data.iloc[i+1]) / fcf_data.iloc[i+1]
            growth_rates.append(growth)

        if growth_rates:
            avg_growth = sum(growth_rates) / len(growth_rates)
            return avg_growth
        return None

    except KeyError:
        st.warning("Could not find 'Free Cash Flow' directly. Using Operating Cash Flow - Capital Expenditures for historical FCF.")
        try:
            op_cash_flow = cash_flow.loc['Operating Cash Flow'].head(4)
            capex = cash_flow.loc['Capital Expenditures'].head(4)
            fcf_data = op_cash_flow - capex
            if len(fcf_data) < 2:
                return None
            
            growth_rates = []
            for i in range(len(fcf_data) - 1):
                growth = (fcf_data.iloc[i] - fcf_data.iloc[i+1]) / fcf_data.iloc[i+1]
                growth_rates.append(growth)

            if growth_rates:
                avg_growth = sum(growth_rates) / len(growth_rates)
                return avg_growth
            return None
        except KeyError:
            return None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_wacc(ticker):
    """Calculates a suggested WACC using CAPM and other data."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        balance_sheet = stock.balance_sheet

        # Get components for WACC calculation
        market_cap = info.get('marketCap')
        beta = info.get('beta')
        total_debt = balance_sheet.loc['Total Debt'].iloc[0]
        
        if market_cap is None or beta is None or total_debt is None:
            st.warning("Could not fetch all necessary data for WACC calculation.")
            return None

        # These are estimations and may not be perfect from yfinance data.
        # Cost of Debt (Rd) - approximation using interest expense and total debt
        # We need a more robust way to get interest expense. Let's use a proxy.
        # This part is a simplification. A real WACC requires bond yields.
        # For this example, let's assume a simple fixed cost of debt.
        cost_of_debt = 0.05  # A simplified assumption for this example
        
        # Tax Rate (T) - approximation from financial statements
        try:
            income_statement = stock.income_stmt
            tax_provision = income_statement.loc['Tax Provision'].iloc[0]
            pretax_income = income_statement.loc['Pretax Income'].iloc[0]
            tax_rate = tax_provision / pretax_income
        except (KeyError, ZeroDivisionError):
            tax_rate = 0.21  # Default to US corporate tax rate if not found
        
        # Risk-Free Rate (Rf) - using a proxy (e.g., 10-Year U.S. Treasury Yield)
        # yfinance doesn't provide this directly, so we'll use a common assumption
        risk_free_rate = 0.045 # This is a hardcoded placeholder
        
        # Market Return (Rm) - average historical market return
        market_return = 0.10 # A common historical average
        
        # Cost of Equity (Re) using CAPM
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

        # Calculate weights
        total_value = market_cap + total_debt
        weight_equity = market_cap / total_value
        weight_debt = total_debt / total_value

        # WACC Formula
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        
        return wacc

    except Exception as e:
        return None
''
col1, col2, col3 = st.columns([3,3,3])
with col1:
    growth_rate = st.number_input("Enter the average annual growth rate (%)") / 100

# Display FCF growth suggestion
    if ticker:
        fcf_growth_suggestion = get_fcf_growth(ticker)
        if fcf_growth_suggestion is not None:
            st.info(f"💡 **Suggestion:** The historical average annual FCF growth rate for **{ticker}** over the last few years is approximately **{fcf_growth_suggestion:.2%}**. Or you can also use the growth rate from this link: https://www.gurufocus.com/term/cashflow-growth-3y/{ticker}")
        else:
            st.warning(f"Could not calculate historical FCF growth for {ticker}. Please input your own estimate. Or you can also use the growth rate from this link: https://www.gurufocus.com/term/cashflow-growth-3y/{ticker}")

with col2:
    discount_rate = st.number_input("Enter the discount rate (WACC) (%)") / 100

# Display WACC suggestion
    if ticker:
        wacc_suggestion = get_wacc(ticker)
        if wacc_suggestion is not None:
            st.info(f"💡 **Suggestion:** The calculated WACC for **{ticker}** is approximately **{wacc_suggestion:.2%}**. Or you can also use the WACC from this link: https://www.gurufocus.com/term/wacc/{ticker}")
        else:
            st.warning(f"Could not calculate WACC for {ticker}. Please input your own estimate. Or you can also use the WACC from this link: https://www.gurufocus.com/term/wacc/{ticker}")

with col3:
    perpetual_growth_rate = st.slider("Select the perpetual growth rate (%):", 0.0, 5.0, 2.5, 0.1) / 100
    st.info(f"💡 **Suggestion:** The perpetual growth rate is usually 1%-2% or 3% in the best case but more likely 1% or 2%.")

st.markdown("---")

# --- Function to Perform Valuation ---
@st.cache_data(ttl=3600)
def calculate_dcf(ticker, growth_rate, discount_rate, perpetual_growth_rate):
    try:
        stock = yf.Ticker(ticker)
        cash_flow = stock.cashflow
        balance_sheet = stock.balance_sheet
        if cash_flow.empty or balance_sheet.empty:
            st.error("Could not fetch cash flow or balance sheet data. Please check the ticker.")
            return None

        try:
            latest_fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
        except KeyError:
            try:
                op_cash_flow = cash_flow.loc['Operating Cash Flow'].iloc[0]
                capex = cash_flow.loc['Capital Expenditures'].iloc[0]
                latest_fcf = op_cash_flow - capex
            except KeyError:
                st.error("Could not find 'Free Cash Flow' or its components in the financial data.")
                return None

        total_debt = balance_sheet.loc['Total Debt'].iloc[0]
        cash_and_equivalents = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
        shares_outstanding = stock.info['sharesOutstanding']

        st.subheader("1. DCF Calculation", divider ='gray')
        dcf_text = "Discounted Cash Flow (DCF) is a valuation method to estimate the value of an asset today based on its future cash flows. It's independent of external factors and instead just relies on the company's ability to generate cash flows. It uses a set of assumptions based on historical data to be able to project how much the company is going to be generating in cash in the next five to ten years. The discount rate, WACC, is used to bring back all the future cash flows to the present. (Discounting thing has to do with the time value of money which is a concept that says that a sum of money today is worth more than a sum of money in the future.) After the forecasted period of say five to ten years, the company doesn't just disintegrate and it keeps on going and keeps on selling. It means that after that forecasted period it is needed to assume a value for it which is known as terminal value (TV). So the terminal value is the value of the company after the forecasted period. Because the calculation is based on historical growth data, the output value is more accurate when the historical data is consistent."
        with st.expander("💡 What is Discounted Cash Flow (DCF)?"): 
            st.markdown(dcf_text, unsafe_allow_html=True)
        ''
        st.write(f"Latest Free Cash Flow (FCF): ${latest_fcf:,.2f}")

        projected_fcf = {}
        df_fcf = pd.DataFrame(columns=['Year', 'Projected FCF'])
        for i in range(1, 11):
            projected_fcf[i] = latest_fcf * ((1 + growth_rate) ** i)
            df_fcf.loc[i] = [f"Year {i}", f"${projected_fcf[i]:,.2f}"]

        terminal_value = (projected_fcf[10] * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)
        df_fcf.loc['terminal'] = ["Terminal Value", f"${terminal_value:,.2f}"]

        total_pv = 0
        df_pv = pd.DataFrame(columns=['Year', 'Discounted FCF'])
        for i in range(1, 11):
            pv = projected_fcf[i] / ((1 + discount_rate) ** i)
            total_pv += pv
            df_pv.loc[i] = [f"Year {i}", f"${pv:,.2f}"]

        pv_terminal_value = terminal_value / ((1 + discount_rate) ** 10)
        df_pv.loc['terminal'] = ["Terminal Value", f"${pv_terminal_value:,.2f}"]
        total_pv += pv_terminal_value

        cal1,cal2 = st.columns([3,3])
        with cal1:
            st.subheader("Projected Free Cash Flows (FCF)")
            df_fcf = pd.DataFrame(df_fcf)
            st.dataframe(df_fcf, hide_index=True, use_container_width=True, height=420)
        with cal2:
            st.subheader("Discounted Cash Flows (DCF)")
            df_pv = pd.DataFrame(df_pv)
            st.dataframe(df_pv, hide_index=True, use_container_width=True, height=420)
        st.write(f"The total present value of all future cash flows: **${total_pv:,.2f}**")

        st.subheader("2. Intrinsic Value Calculation", divider ='gray')
        enterprise_value = total_pv
        equity_value = enterprise_value + cash_and_equivalents - total_debt
        shares_outstanding = stock.info['sharesOutstanding']
        intrinsic_value_per_share = equity_value / shares_outstanding

        st.write(f"Enterprise Value (Total PV of FCFs): **${enterprise_value:,.2f}**")
        st.write(f"Cash and Equivalents Value: **${cash_and_equivalents:,.2f}**")
        st.write(f"Total Debt Value: **${total_debt:,.2f}**")
        st.write(f"Equity Value (Enterprise Value + Cash - Debt): **${equity_value:,.2f}**")
        st.write(f"Shares Outstanding: **{shares_outstanding:,.0f}**")

        st.subheader("3. Final Result", divider ='gray')
        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.metric(label=f"Intrinsic Value per Share", value=f"${intrinsic_value_per_share:,.2f}")
        with result_col2:
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            st.metric(label="Current Market Price", value=f"${current_price:,.2f}")
        with result_col3:
            mos_value = ((float(intrinsic_value_per_share) - current_price)/float(intrinsic_value_per_share)) * 100
            st.metric(label="Margin of Safety", value=f"{mos_value:,.2f}%")

        st.subheader("Valuation Conclusion")
        conclusion_col1, colclusion_col2 = st.columns(2)
        with conclusion_col1:
            if intrinsic_value_per_share > current_price:
                st.success(f"Based on the DCF model, the stock appears **undervalued**.")
            else:
                st.warning(f"Based on the DCF model, the stock appears **overvalued**.")

        # --- Sensitivity Analysis Table ---
        st.subheader("4. Sensitivity Analysis", divider='gray')
        st.markdown("This table shows the intrinsic value per share based on different WACC and Perpetual Growth Rate assumptions.")

        wacc_range = [discount_rate - 0.02, discount_rate, discount_rate + 0.02]
        growth_range = [perpetual_growth_rate - 0.01, perpetual_growth_rate, perpetual_growth_rate + 0.01]
        
        wacc_range = [max(0.01, rate) for rate in wacc_range]
        growth_range = [max(0.00, rate) for rate in growth_range]
        
        # Updated line to change column headers
        col_headers = [f"WACC: {w*100:.1f}%" for w in wacc_range]
        # Updated line to change row headers
        row_headers = [f"Growth rate: {g*100:.1f}%" for g in growth_range]
        
        sensitivity_df = pd.DataFrame(index=row_headers, columns=col_headers)

        for g_rate in growth_range:
            for w_rate in wacc_range:
                terminal_value_sa = (projected_fcf[10] * (1 + g_rate)) / (w_rate - g_rate)
                pv_terminal_value_sa = terminal_value_sa / ((1 + w_rate) ** 10)
                
                total_pv_sa = sum(projected_fcf[i] / ((1 + w_rate) ** i) for i in range(1, 11)) + pv_terminal_value_sa
                
                equity_value_sa = total_pv_sa + cash_and_equivalents - total_debt
                intrinsic_value_per_share_sa = equity_value_sa / shares_outstanding
                
                # Use the new row and column headers to populate the DataFrame
                sensitivity_df.loc[f"Growth rate: {g_rate*100:.1f}%", f"WACC: {w_rate*100:.1f}%"] = f"${intrinsic_value_per_share_sa:,.2f}"

        # Define the new styling function
        def highlight_cells(data):
            df_styled = pd.DataFrame('', index=data.index, columns=data.columns)
            
            # Highlight the user's specific input cell with blue
            user_wacc_label = f"WACC: {discount_rate*100:.1f}%"
            user_growth_label = f"Growth rate: {perpetual_growth_rate*100:.1f}%"
            if user_growth_label in df_styled.index and user_wacc_label in df_styled.columns:
                df_styled.loc[user_growth_label, user_wacc_label] = 'background-color: #3BAFDA'

            # Best case (highest growth, lowest WACC) is row 3, column 1
            df_styled.iloc[2, 0] = 'background-color: #37BC9B'

            # Worst case (lowest growth, highest WACC) is row 1, column 3
            df_styled.iloc[0, 2] = 'background-color: #DA4453'

            return df_styled

        # Apply the styling
        sen_col1, sen_col2 = st.columns([3,3])
        with sen_col1:
            st.dataframe(sensitivity_df.style.apply(highlight_cells, axis=None), use_container_width=True)
        with sen_col2:
            st.info(f"💡 Sensitivity analysis is used to test how changes in key assumptions impact a company's final valuation. Green color is for a bullish scenario, red color is for a bearish scenario and blue color is the base case.")

        return intrinsic_value_per_share, current_price

    except Exception as e:
        st.error(f"An error occurred: {e}. Please check the ticker and inputs.")
        return None

# Button to Trigger Calculation
if st.button("Calculate Intrinsic Value"):
    if ticker:
        calculate_dcf(ticker, growth_rate, discount_rate, perpetual_growth_rate)
    else:
        st.error("Please enter a stock ticker to proceed.")

st.subheader("", divider ='gray')
iiqc1, iiqc2 = st.columns ([3,1])
with iiqc1:
    st.write("")
    st.markdown("**Disclaimer:**")
    st.write("This calculator is designed to enable beginner investors to analyze stocks effectively and with ease. Please note that the information in this page is intended for educational purposes only and it does not constitute investment advice or a recommendation to buy or sell any security. We are not responsible for any losses resulting from trading decisions based on this information.")
with iiqc2:
    invest_iq_central='./Image/InvestIQCentral.png'
    st.image(invest_iq_central,width=300)
''

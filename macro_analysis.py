import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
import plotly.express as px
import numpy as np 
import requests
import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
from groq import Groq
import re

st.set_page_config(layout="wide", page_title="Economic Analysis Dashboard")

ai_model = 'llama-3.3-70b-versatile'
FRED_API_KEY = st.secrets["FRED_API_KEY"]
NUM_POINTS = 20
GROWTH_COLOR = 'red'
PMI_API_URL = "https://api.db.nomics.world/v22/series/ISM/pmi?facets=1&format=json&limit=1000&observations=1" 

compare_tickers = ["^GSPC","GC=F","SI=F","CL=F","BTC-USD"]
ticker_names = {
    "^GSPC": "S&P 500",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "CL=F": "Crude Oil",
    "BTC-USD": "Bitcoin"
}
custom_colors = {
    "S&P 500": "#5E9BEB",   
    "Gold": "#FFCE54",      
    "Silver": "#FB6E51",    
    "Crude Oil": "#AAB2BD", 
    "Bitcoin": "#AC92EC"    
}

LOG_SCALE_SERIES = ["GDPC1", "UNRATE", "CPIAUCSL", "M2SL"]
REVERSED_DELTA_SERIES = ["UNRATE", "CPIAUCSL", "M2SL"]

SERIES_CONFIG = {
    "Economic and Labor Market": [
        {"name": "Real GDP", "id": "GDPC1", "type": "Bar+Line (Growth)", "color": "#5E9BEB"},
        {"name": "Non-farm Payroll (NFP)", "id": "PAYEMS", "type": "Line+Growth", "color": "#3BAFDA"},
        {"name": "Industrial Production Index", "id": "INDPRO", "type": "Line+Growth", "color": "#4FC1E9"},
        {"name": "Unemployment Rate", "id": "UNRATE", "type": "Area", "color": "#4A89DC"},
    ],
    "Inflation and Monetary Policy": [
        {"name": "Consumer Price Index (CPI)", "id": "CPIAUCSL", "type": "Bar+Line (Growth)", "color": "#F6BB42"},
        {"name": "FED Funds Rate", "id": "DFF", "type": "Area", "color": "#FB6E51"},
        {"name": "Money supply (M2)", "id": "M2SL", "type": "Bar+Line (Growth)", "color": "#F6BB42"},
        {"name": "Inflation", "id": "FPCPITOTLZGUSA", "type": "Area", "color": "#E9573F"},
    ],
    "Leading Economic Indicators": [
        {"name": "Yield curve (10Y-2Y)", "id": "T10Y2Y", "type": "Line+Zero", "color": "#EC87C0"},
        {"name": "Consumer Sentiment Index (MCSI)", "id": "UMCSENT", "type": "Line+Target", "color": "#AC92EC"},
        {"name": "ISM Manufacturing PMI", "id": "PMI_ISM", "type": "Line+Fifty", "color": "#A0D468"},
        {"name": "Leading Index", "id": "USSLIND", "type": "Line+Zero", "color": "#37BC9B"},
    ],
    "Debt Metrics": [
        {"name": "Debt to GDP ratio", "id": "GFDEGDQ188S", "type": "Line", "color": "#48CFAD"},
        {"name": "Real Broad Dollar Index", "id": "RTWEXBGS", "type": "Line+100", "color": "#4FC1E9"},
    ],
    "Recession Indicators": [
        {"name": "Real-time Sahm Rule Recession Indicator", "id": "SAHMREALTIME", "type": "Line+0.5", "color": "#F05050"},
        {"name": "GDP-Based Recession Indicator Index", "id": "JHGDPBRINDX", "type": "Line+67", "color": "#F05050"},
    ],

}

SERIES_DESCRIPTIONS = {
    "Real GDP": "The **total value** of all final goods and services produced in an economy, **adjusted for inflation**.",
    "Non-farm Payroll (NFP)": "The **net change** in the number of **paid US workers**, excluding farm workers and some government/non-profit jobs. strong job growth = expansion, job losses = contraction",
    "Industrial Production Index": "Measures the **total output** of the manufacturing, mining, and electric and gas utility industries. growing production = economic expansion",
    "Unemployment Rate": "The **percentage** of the total labor force that is **unemployed** but actively seeking employment. rising rate = a recession has started",
    "Consumer Price Index (CPI)": "Measures the **average change** over time in the prices paid by urban consumers for a **market basket** of consumer goods and services. High inflation can persist into early recession phases.",
    "FED Funds Rate": "The **target interest rate** set by the Federal Reserve for **banks to lend to one another** overnight.",
    "Money supply (M2)": "A broad measure of **money in circulation**, including currency, checking deposits, savings deposits, and money market funds.",
    "Inflation": "The **rate at which the general level of prices** for goods and services is **rising**, and consequently, purchasing power is falling.",
    "Yield curve (10Y-2Y)": "The **difference in interest rates** (yields) between **10-year and 2-year** US Treasury bonds. A reading **below 0** indicates a recession signal.",
    "Consumer Sentiment Index (MCSI)": "A gauge of **consumers' optimism** about their personal finances and the state of the economy. A reading **below 80** indicates a recession signal.",
    "ISM Manufacturing PMI": "Measures the **economic health of the manufacturing sector**. A reading **above 50** indicates expansion; **below 50** indicates contraction.",
    "Debt to GDP ratio": "The **ratio** of a country's **total government debt** to its total Gross Domestic Product.",
    "Real-time Sahm Rule Recession Indicator": "Sahm Recession Indicator signals the start of a recession when the three-month moving average of the national unemployment rate (U3) rises by 0.50 percentage points or more relative to the minimum of the three-month averages from the previous 12 months.",
    "GDP-Based Recession Indicator Index": "If the value of the index rises above 67% that is a historically reliable indicator that the economy has entered a recession. Once this threshold has been passed, if it falls below 33% that is a reliable indicator that the recession is over.",
    "Real Broad Dollar Index": "Low index indicates weak dollar value relative to other currencies and high index indicates strong dollar value relative to other currencies.",
    "Leading Index": "A composite index of multiple leading indicators, including stock prices, manufacturing orders, jobless claims, and building permits. Consistent decline signals recession risk.",
}

end_date = datetime.datetime.today()
start_date = end_date - relativedelta(years=1)
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')
def relativereturn(df):
    """Calculates cumulative relative return starting at 0."""
    rel = df.pct_change()
    cumret = (1 + rel).cumprod() - 1
    cumret = cumret.fillna(0)
    return cumret

TICKERS = {
    'S&P 500': '^GSPC',
    'Dow Jones': '^DJI',
    'Nasdaq': '^IXIC',
    'Russell 2000': '^RUT'
}

PERIOD_CHART = "1y"
EMA_PERIODS = [20, 50, 200]
CHART_TICKER_NAME = 'S&P 500'
VIX_SERIES_ID = 'VIXCLS'

EMA_COLORS = {
    20: 'green',
    50: 'gold', 
    200: 'red'
}

sector_etfs = {
    "SPY": "S&P Index", 
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary"
}

@st.cache_data(ttl=3600)
def load_and_calculate_data(tickers_list, start, end):
    """Loads price data, calculates cumulative return, and gets last price."""
    #st.info(f"Downloading data from {start} to {end}...")
    try:
        data = yf.download(tickers_list, start, end)['Close']
    except Exception as e:
        st.error(f"Failed to download data from Yahoo Finance. Error: {e}")
        return pd.DataFrame(), pd.DataFrame() 
    data.rename(columns=ticker_names, inplace=True)
    data.dropna(axis=1, how='all', inplace=True) 
    df_cumulative_return = relativereturn(data)
    last_valid_prices = data.ffill().iloc[-1]
    df_current_data = pd.DataFrame({
        'Current Price': last_valid_prices
    })
    df_current_data.dropna(inplace=True)
    return df_cumulative_return, df_current_data

def hex_to_rgba(hex_color, alpha):
    """Converts a hex color code to an RGBA string with specified alpha transparency."""
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

@st.cache_data(ttl=3600)
def fetch_pmi_data(url: str) -> pd.DataFrame:
    """
    Fetches PMI data from the DB.nomics API, processes the JSON, and returns a pandas DataFrame.
    """
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        series_data = data.get("series", {}).get("docs", [])
        
        if not series_data:
            st.error("Error: No time series data found in the API response for PMI.")
            return pd.DataFrame()
            
        dates = series_data[0].get("period_start_day", [])
        values = series_data[0].get("value", [])
        
        df_full = pd.DataFrame({
            'Date': dates,
            'Value': values
        })
        df_full['Date'] = pd.to_datetime(df_full['Date'])
        df_full['Value'] = pd.to_numeric(df_full['Value'])

        df_full.set_index('Date', inplace=True)
        df_data = df_full.dropna().tail(NUM_POINTS).reset_index()

        return df_data

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching PMI data from the API: Network error or API is unavailable. Details: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during PMI data processing: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_and_prepare_data(series_id, api_key):
    """
    Fetches FRED series and calculates metrics. 
    NOTE: This function is only used for FRED series (not PMI).
    """
    if not api_key:
        return None, None, None, None, None

    try:
        fred = Fred(api_key=api_key)
        series_data = fred.get_series(series_id).dropna()
        df_full = series_data.to_frame(name='Value')
        growth_rate_value = np.nan
        growth_label = ''

        if series_id in ["GDPC1", "PAYEMS", "INDPRO", "CPIAUCSL", "M2SL"]:
            series_info = fred.get_series_info(series_id)
            freq = series_info.get('frequency_short', 'M').upper()
            
            if series_id in ["CPIAUCSL", "M2SL"]:
                periods_in_year = 4 if freq == 'Q' else 12 if freq == 'M' else 1
                period_lag = periods_in_year
                growth_label = 'YoY Growth %'
            else:
                period_lag = 1
                growth_label = f'{freq}o{freq} Growth %'
            
            full_series_growth = series_data.pct_change(periods=period_lag) * 100
            
            growth_rate_value = full_series_growth.iloc[-1]
            
            df_full = pd.concat([series_data.rename('Value'), full_series_growth.rename('Growth_Rate')], axis=1)
            
        df = df_full.dropna().tail(NUM_POINTS).reset_index()
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        
        if series_id in ["CPIAUCSL", "M2SL"]:
            growth_label = 'YoY Growth %'
        elif series_id in ["GDPC1", "PAYEMS", "INDPRO"]:
            series_info = fred.get_series_info(series_id)
            freq = series_info.get('frequency_short', 'M').upper()
            growth_label = f'{freq}o{freq} Growth %'
        
        current_value = df['Value'].iloc[-1]
        
        last_date = df['Date'].iloc[-1]
        
        if series_id == "GDPC1" or series_data.index.freqstr in ['QS', 'QE', 'Q', 'Q-JAN', 'Q-FEB', 'Q-MAR', 'Q-APR', 'Q-MAY', 'Q-JUN', 'Q-JUL', 'Q-AUG', 'Q-SEP', 'Q-OCT', 'Q-NOV', 'Q-DEC']:
            quarter = (last_date.month - 1) // 3 + 1
            last_date_str = f"Q{quarter} {last_date.year}"
        elif series_data.index.freqstr in ['AS', 'A']:
            last_date_str = str(last_date.year)
        else:
            last_date_str = last_date.strftime('%b %Y')

        return df, current_value, growth_rate_value, growth_label, last_date_str

    except Exception as e:
        st.error(f"Error fetching data for series {series_id}: {e}")
        return None, None, None, None, None

def plot_combined_chart(df, name, series_id, main_type, main_color, growth_color, growth_label):
    """Plots a chart with the main series and a secondary growth rate (Bar/Line + Line)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if main_type == "Bar":
        fig.add_trace(
            go.Bar(
                x=df['Date'], 
                y=df['Value'], 
                name=name, 
                marker_color=main_color,
            ),
            secondary_y=False,
        )
    elif main_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['Value'], 
                name=name, 
                line=dict(color=main_color, width=3, shape='spline', smoothing=1.3), 
                mode='lines+markers'
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Growth_Rate'], 
            name=growth_label, 
            line=dict(color=growth_color, width=1), 
            mode='lines+markers'
        ),
        secondary_y=True,
    )
    yaxis_type = "log" if series_id in LOG_SCALE_SERIES else "linear"
    fig.update_layout(
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center'),
        margin=dict(l=20, r=20, t=0, b=20),
        height=300
    )  
    fig.update_yaxes(title_text=name, secondary_y=False, type=yaxis_type)
    fig.update_yaxes(
        title_text="", 
        secondary_y=True, 
        showgrid=False, 
        showline=False, 
        zeroline=False,
        visible=False
    )
    fig.update_xaxes(title_text=None)
    return fig

def plot_single_area_line(df, name, series_id, chart_type, line_color, target_line=None):
    """Plots simple Area/Line charts with custom horizontal lines, including Line+Fifty for PMI."""
    yaxis_type = "log" if series_id in LOG_SCALE_SERIES else "linear"
    
    if chart_type == "Area":
        fill_color_rgba = hex_to_rgba(line_color, 0.2) 
        fig = px.area(df, x='Date', y='Value', log_y=(yaxis_type == "log"))
        fig.update_traces(fillcolor=fill_color_rgba, line=dict(color=line_color, width=2))    
    elif chart_type in ["Line", "Line+Zero", "Line+Target", "Line+Fifty", "Line+0.5", "Line+67", "Line+100"]: 
        fig = px.line(df, x='Date', y='Value', log_y=(yaxis_type == "log"))
        fig.update_traces(line=dict(color=line_color, width=3, shape='spline', smoothing=1.3), mode='lines+markers')  
    
    hline_color = "darkred"

    if chart_type == "Line+Zero":
        fig.add_hline(y=0, line_dash="dash", line_color=hline_color, annotation_text="Trigger Line", annotation_position="top left")   
    if chart_type == "Line+Target":
        fig.add_hline(y=80, line_dash="dash", line_color=hline_color, annotation_text="Target of 80", annotation_position="top left")
    if chart_type == "Line+Fifty": 
        fig.add_hline(y=50, line_dash="dash", line_color=hline_color, annotation_text="Expansion/Contraction (50)", annotation_position="top left")
    if chart_type == "Line+0.5":
        fig.add_hline(y=0.5, line_dash="dash", line_color=hline_color, annotation_text="Recession Alert", annotation_position="top left")
    if chart_type == "Line+67":
        fig.add_hline(y=67, line_dash="dash", line_color=hline_color, annotation_text="Recession Alert", annotation_position="top left")
    if chart_type == "Line+100":
        fig.add_hline(y=100, line_dash="dash", line_color=hline_color, annotation_text="Index = 100", annotation_position="top left")
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=15, b=20),
        height=300,
        xaxis_title=None,
        yaxis_title=name,
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center'),
    )
    if yaxis_type == "log":
        fig.update_yaxes(title_text=f"{name}")
    return fig

st.title("U.S. Macroeconomic Dashboard")
st.markdown(f"Data sourced from FRED, Yahoo Finance and DB.nomics.")
st.write("")
st.write("")

#################################################################################################################################################################
overview_data, stock_market_data, crypto_market_data, commodity_market_data = st.tabs (["Economic Overview","Stock Market", "Crypto Market", "Commodity Market"])

with overview_data:
    df_returns, df_current_data = load_and_calculate_data(compare_tickers, start_str, end_str)
    col1, col2 = st.columns([3, 1])
    with col1:
        if df_returns.empty:
            st.warning("⚠️ Could not generate chart. No common historical data was found for the assets in the selected 10-year period. Please try again.")
        else:
            fig = go.Figure()
            for col in df_returns.columns:
                color = custom_colors.get(col, "#000000") 
                fig.add_trace(go.Scatter(
                    x=df_returns.index,
                    y=df_returns[col] * 100,
                    mode='lines',
                    name=col,
                    line=dict(color=color) 
                ))
            fig.update_layout(
                title={"text":f"Cumulative Relative Return Over 1 Year", "font": {"size": 25}},
                xaxis_title="Date",
                yaxis_type="linear", 
                yaxis_title="Cumulative Return (%)",
                yaxis_tickformat=".0f",
                #legend_title="Asset",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center'),
                margin=dict(l=20, r=40, t=80, b=10),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            min_date_str = df_returns.index.min().strftime('%Y-%m-%d')
            st.caption(f"Data source: Yahoo Finance. Comparison generated over {end_date.year - start_date.year} years.")
    with col2:
        st.subheader("Current Prices")
        if df_current_data.empty:
            st.info("No current data available.")
        else:
            for ticker_name in df_current_data.index:
                price = df_current_data.loc[ticker_name, 'Current Price']
                if price >= 1:
                    formatted_price = f"${price:,.2f}"
                else:
                    formatted_price = f"${price:,.4f}"
                st.metric(
                    label=ticker_name,
                    value=formatted_price
                )

    for group_name, series_list in SERIES_CONFIG.items():
        st.subheader(f"{group_name}", divider = "rainbow")
        group_cols = st.columns(2) 
        col_index_in_group = 0
        
        for config in series_list:
            name = config["name"]
            series_id = config["id"]
            chart_req = config["type"]
            main_color = config["color"]
            current_group_col = group_cols[col_index_in_group % 2]
            
            if series_id == "PMI_ISM":
                df_data = fetch_pmi_data(PMI_API_URL)
                if not df_data.empty:
                    current_value = df_data['Value'].iloc[-1]
                    growth_rate_value = np.nan 
                    growth_label = ""
                    last_date_str = df_data['Date'].iloc[-1].strftime('%b %Y') 
                else:
                    df_data, current_value, growth_rate_value, growth_label, last_date_str = None, None, None, None, None
            else:
                df_data, current_value, growth_rate_value, growth_label, last_date_str = fetch_and_prepare_data(series_id, FRED_API_KEY)
                
            with current_group_col:
                st.write("")
                st.markdown(f"#### {name}")
                metric_col, desc_col = st.columns([0.4, 0.6])

                if df_data is not None and not df_data.empty:
                    
                    value_format = f"{current_value:,.2f}"
                    
                    if series_id in ["GDPC1", "PAYEMS", "INDPRO", "CPIAUCSL", "M2SL"] and not np.isnan(growth_rate_value):
                        delta_format = f"{growth_rate_value:,.2f} %"
                        if series_id in REVERSED_DELTA_SERIES:
                            delta_color_option = "inverse"
                        else:
                            delta_color_option = "normal"
                        delta_metric_display = f"{delta_format} ({growth_label.split(' ')[0]})"
                    else:
                        delta_metric_display = None 
                        delta_color_option = "off"
                    
                    with metric_col:
                        st.metric(
                            label=last_date_str if last_date_str else name, 
                            value=value_format,
                            delta=delta_metric_display,
                            delta_color=delta_color_option
                        )
                    with desc_col:
                        st.caption(SERIES_DESCRIPTIONS.get(name, "Description not available."))
                    
                    if chart_req in ["Bar+Line (Growth)", "Line+Growth"]:
                        if 'Growth_Rate' in df_data.columns:
                            main_plot_type = "Bar" if "Bar" in chart_req else "Line"
                            fig = plot_combined_chart(
                                df_data, name, series_id, main_plot_type, main_color, GROWTH_COLOR, growth_label
                            )
                        else:
                            fig = plot_single_area_line(df_data, name, series_id, "Line", main_color)
                    elif chart_req in ["Area", "Line", "Line+Zero", "Line+Target", "Line+Fifty", "Line+0.5", "Line+67", "Line+100"]: 
                        fig = plot_single_area_line(
                            df_data, 
                            name, 
                            series_id, 
                            chart_req, 
                            main_color
                        )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    st.write("")
                    st.write("")
                else:
                    st.warning(f"No data available for {name} ({series_id}).")
            col_index_in_group += 1

##########################################################################################################################
    
    fred = Fred(api_key=FRED_API_KEY)
    
    SERIES_MAP = {
        'GDPC1': 'Real GDP',
        'PAYEMS': 'Non-farm Payroll',
        'INDPRO': 'Industrial Production',
        'CPIAUCSL': 'Consumer Price Index',
        'UNRATE': 'Unemployment Rate',
        'T10Y2Y': 'Yield Curve (10Y-2Y Spread)',
        'M2SL': 'Money Supply (M2)',
        'UMCSENT': 'Consumer Sentiment Index',
        'GFDEGDQ188S': 'Debt to GDP Ratio',
        'DFF': 'FED Fund Rate',
        'FPCPITOTLZGUSA': 'Inflation (Annual % Chg)'
    }
    
    LATEST_OBSERVATIONS = 30
    
    @st.cache_data(ttl=3600)
    def get_latest_fred_data_and_process(series_map, n_obs):
        """
        Fetches the latest N data points for each FRED series,
        aligns them to monthly frequency, and forward-fills missing values.
        """
        df_combined = pd.DataFrame()
        
        for series_id, series_name in series_map.items():
            try:
                start_date = pd.to_datetime('today') - pd.DateOffset(years=2, months=6)
                series = fred.get_series(series_id, observation_start=start_date.strftime('%Y-%m-%d'))
                processed_series = series.resample('MS').mean()    
                df_temp = processed_series.to_frame(name=series_name)
                if df_combined.empty:
                    df_combined = df_temp
                else:
                    df_combined = pd.merge(df_combined, df_temp, how='outer', left_index=True, right_index=True)
            except Exception as e:
                st.warning(f"Could not fetch or process series **{series_name}** ({series_id}): {e}")
                continue
        df_combined = df_combined.sort_index()
        df_combined = df_combined.ffill()
        df_combined = df_combined.tail(n_obs)
        df_combined = df_combined.reset_index().rename(columns={'index': 'Date'})
        df_combined['Date'] = df_combined['Date'].dt.strftime('%Y-%m-%d')
        return df_combined
    
    st.subheader("AI Economy Analysis", divider = "rainbow")
    st.caption(f"Based on the **latest {LATEST_OBSERVATIONS} months** of data aligned to a **Monthly** frequency.")
    
    try:
        with st.spinner(f'Fetching and processing the latest {LATEST_OBSERVATIONS} months of FRED data...'):
            df_latest = get_latest_fred_data_and_process(SERIES_MAP, LATEST_OBSERVATIONS)   
        # if not df_latest.empty:
        #     st.subheader(f"Latest {LATEST_OBSERVATIONS} Monthly Economic Time Series Data") 
        #     st.dataframe(df_latest, use_container_width=True) 
        # else:
        #     st.error("No data could be retrieved. Please check your API key and network connection.")
    
        analysis = ""
        try:
            api_key = st.secrets["GROQ_API_KEY"]
            client = Groq(api_key=api_key)
            summary_prompt = f"""
                Analyze the economic data. Provide:
                - current U.S. economic condition (expansion, moving to peak, peak, moving to contraction, contraction, moving to trough, trough, moving to expansion) with explanations 
                - the concluded answer with this format: Economic Cycle level - [expansion or moving to peak or peak or moving to contraction or contraction or moving to trough or trough or moving to expansion]
                """
    
            def analyze_stock(prompt_text, tokens):
                response = client.chat.completions.create(
                    model=ai_model,
                    messages=[
                        {"role": "system", "content": "You are an experienced financial analyst with expertise in both fundamental and technical analysis."},
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens= tokens,
                    temperature=0.7
                )
                        
                raw_response = response.choices[0].message.content
                try:
                    cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
                except: 
                    cleaned_response = raw_response
                return cleaned_response
            summary_analysis = analyze_stock(summary_prompt,10000)
            analysis = {
                'summary': summary_analysis,
            }
        except Exception as e:
            analysis = ""
        ai_ans1, ai_ans2 = st.columns([3,3])
        with ai_ans1:
            try:
                with st.spinner('Analyzing stock data...'):
                    cleaned_text = analysis['summary'].replace('\\n', '\n').replace('\\', '')
                    special_chars = ['$', '>', '<', '`', '|', '[', ']', '(', ')', '+', '{', '}', '!', '&']
                    for char in special_chars:
                        cleaned_text = cleaned_text.replace(char, f"\\{char}")
                    st.markdown(cleaned_text, unsafe_allow_html=True)
            except Exception as e:
                st.warning("AI analysis is currently unavailable.")
    
        with ai_ans2:
            def remove_markdown(text):
                """Removes common Markdown characters and code blocks from a string."""
                text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
                text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
                text = re.sub(r'^\s*[-*+]?\s*\d*\.?\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'([*_]{1,2})', '', text)
                text = re.sub(r'\n{2,}', '\n', text)
                return text.strip()
            cleaned_text = remove_markdown(cleaned_text)
            delimiter = 'Economic Cycle level - '
            extracted_value = cleaned_text.split(delimiter)[-1].strip()
            current_stage = extracted_value.lower() 
    
            CYCLE_PHASES = [
                'moving to expansion', 'expansion', 'moving to peak',
                'peak', 'moving to contraction', 'contraction',
                'moving to trough', 'trough'
            ]
        
            try:
                current_index = CYCLE_PHASES.index(current_stage)
            except ValueError:
                st.error(f"Error: '{current_stage}' is not a recognized cycle phase.")
                st.stop()
            
            x_phase_points = np.linspace(0, 1.75 * np.pi, len(CYCLE_PHASES))
            offset = x_phase_points[3] - np.pi / 2
            x = np.linspace(0, 1.75 * np.pi, 100)
            y = np.sin(x - offset) * 1 
            x_position_for_stage = x_phase_points[current_index]
            y_position_for_stage = np.interp(x_position_for_stage, x, y)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='#4FC1E9', width=4),
                name='Economic Growth'
            ))
            fig.add_trace(go.Scatter(
                x=[x_position_for_stage],
                y=[y_position_for_stage],
                mode='markers',
                marker=dict(size=30, color='#4FC1E9', opacity=1, line=dict(width=2, color='white')),
                name='Current Position',
                hoverinfo='text',
                text=f"Stage: {current_stage.title()}"
            ))
            num_segments = len(CYCLE_PHASES)
            color_map = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#3D9970', '#FFDC00', '#FF851B', '#FF4136']
            x_segment_starts = x_phase_points
            gradient_bar_y_level = -1.2
            for i in range(num_segments - 1):
                x_start = x_segment_starts[i]
                x_end = x_segment_starts[i+1]
                x_segment = np.linspace(x_start, x_end, 10)
                y_segment = np.full_like(x_segment, gradient_bar_y_level)
                fig.add_trace(go.Scatter(
                    x=x_segment,
                    y=y_segment,
                    mode='lines',
                    line=dict(color=color_map[i], width=15),
                    hoverinfo='skip',
                    showlegend=False,
                ))
            fig.update_layout(
                title={"text":f"Economic Cycle Visual Chart", "font": {"size": 25}},
                #plot_bgcolor='black',
                xaxis=dict(
                    tickmode='array',
                    tickvals=x_phase_points,
                    ticktext=[phase.title() for phase in CYCLE_PHASES],
                    showgrid=False,
                    tickangle=-40,
                ),
                yaxis=dict(
                    title='Economic Growth Level',
                    showticklabels=False,
                    showgrid=False,
                    zeroline=True,
                    zerolinecolor='gray',
                    zerolinewidth=2
                ),
                showlegend=False,
                height=450,
                yaxis_range=[-1.2, 1.2]
            )
            fig.add_annotation(
                x=0.5, y=1,
                xref="paper", yref="paper",
                text=f"Current Economic Cycle Stage: {current_stage.upper()}",
                showarrow=False,
                font=dict(size=18, color="#4FC1E9"),
                yanchor="middle",
                xanchor="center"
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

##########################################################################################################################

with stock_market_data:
    @st.cache_data(ttl=3600)
    def get_current_data(tickers):
        """
        Fetches the current price and daily change for the specified indices.
        We fetch 2 days of data to accurately calculate the daily change (current vs. previous close).
        """
        data = {}
        ticker_list = list(tickers.values())
        df_all = yf.download(ticker_list, period="2d", interval="1d", progress=False)
        for name, ticker in tickers.items():
            try:
                if len(ticker_list) == 1 or not isinstance(df_all['Close'], pd.DataFrame):
                    df = df_all['Close']
                else:
                    df = df_all['Close'][ticker]
                if not df.empty and len(df) >= 2:
                    latest_close = df.iloc[-1]
                    previous_close = df.iloc[-2] 
                    daily_change = latest_close - previous_close
                    daily_change_percent = (daily_change / previous_close) * 100               
                    data[name] = {
                        'price': latest_close,
                        'delta': f'{daily_change:+.2f} ({daily_change_percent:+.2f}%)',
                        'raw_delta': daily_change
                    }
                elif not df.empty:
                    data[name] = {
                        'price': df.iloc[-1],
                        'delta': "N/A (No previous day data)",
                        'raw_delta': 0
                    }
                else:
                    data[name] = {'price': 0, 'delta': "N/A", 'raw_delta': 0}               
            except Exception as e:
                st.error(f"Error fetching current data for {name}: {e}")
                data[name] = {'price': 0, 'delta': "Error", 'raw_delta': 0}
        return data

    @st.cache_data(ttl=3600)
    def get_historical_data_with_emas(ticker_name, ticker, period, ema_periods):
        """
        Fetches historical 'Close' price for the given ticker and calculates EMAs.
        It fetches a longer period to ensure 200-day EMA calculation is accurate.
        """
        end_date = datetime.datetime.today()
        start_date = end_date - relativedelta(days=365 * 2) 
        df_series = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
        if df_series.empty:
            return pd.DataFrame()
        df = pd.DataFrame(df_series)
        df.columns = ['Price']
        for p in ema_periods:
            df[f'EMA {p}'] = df['Price'].ewm(span=p, adjust=False).mean()

        if period.endswith('y'):
            years = int(period[:-1]) 
            data_start_date = end_date - relativedelta(days=365 * years)
            df_filtered = df[df.index >= data_start_date]
        else:
            df_filtered = df
        return df_filtered

    @st.cache_data(ttl=3600)
    def get_vix_data(series_id, api_key, period):
        """
        Fetches VIX data from the FRED API and filters it for the specified period.
        """
        try:
            fred = Fred(api_key=api_key)
            series_data = fred.get_series(series_id).dropna()
            df_vix = series_data.to_frame(name='VIX')
            end_date = datetime.datetime.today()
            if period.endswith('y'):
                years = int(period[:-1]) 
                data_start_date = end_date - relativedelta(days=365 * years)
                df_filtered = df_vix[df_vix.index >= data_start_date]
            else:
                df_filtered = df_vix
                
            return df_filtered
        except Exception as e:
            st.error(f"Error fetching VIX data from FRED: {e}")
            return pd.DataFrame()

    st.header("US Stock Market Overview", divider="rainbow")
    st.subheader("Daily Performance")

    ########################################################## performance  ##########################################################

    col1, col2, col3, col4 = st.columns(4)

    current_data = get_current_data(TICKERS)
    cols = [col1, col2, col3, col4]
    index_names = list(TICKERS.keys())
    for i, name in enumerate(index_names):
        data = current_data[name]
        price_str = f"${data['price']:,.2f}" if isinstance(data['price'], (int, float)) and data['price'] != 0 else data['price']
        cols[i].metric(
            label=name, 
            value=price_str, 
            delta=data['delta'],
            delta_color="normal"
        )

    st.write("")
    st.write("")

    ########################################################## Charts ##########################################################

    st.write("")
    st.write("")
    #st.header(f"Index and Volatility Charts ({PERIOD_CHART})")
    chart_col1, chart_col2 = st.columns([3,2])

    sp_500_ticker = TICKERS[CHART_TICKER_NAME]
    historical_df = get_historical_data_with_emas(CHART_TICKER_NAME, sp_500_ticker, PERIOD_CHART, EMA_PERIODS)
    vix_df = get_vix_data(VIX_SERIES_ID, FRED_API_KEY, PERIOD_CHART)

    with chart_col1:
        #st.subheader(f"{CHART_TICKER_NAME} Price (Log Scale)")
        if not historical_df.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=historical_df.index, 
                    y=historical_df['Price'], 
                    mode='lines', 
                    name='Price',
                    line=dict(color='#4FC1E9', width=2)
                )
            )
            for p in EMA_PERIODS:
                line_color = EMA_COLORS.get(p, 'gray') 
                
                fig.add_trace(
                    go.Scatter(
                        x=historical_df.index, 
                        y=historical_df[f'EMA {p}'], 
                        mode='lines', 
                        name=f'EMA {p}',
                        line=dict(color=line_color, dash='dot', width=1)
                    )
                )
            fig.update_layout(
                title={"text":f"S&P 500 Index", "font": {"size": 25}},
                yaxis_title='Price',
                #xaxis_title='Date',
                yaxis_type="log", 
                hovermode="x unified",
                legend_title="Indicators",
                #xaxis=dict(rangeslider_visible=True),
                margin=dict(l=20, r=20, t=40, b=20),
                height=400, 
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No {PERIOD_CHART} data available for {CHART_TICKER_NAME}.")

    with chart_col2:
        #st.subheader("VIX (Fear Index)")
        if not vix_df.empty:
            fig_vix = go.Figure()

            fig_vix.add_trace(
                go.Scatter(
                    x=vix_df.index, 
                    y=vix_df['VIX'], 
                    mode='lines', 
                    name='VIX',
                    line=dict(color='#AC92EC', width=2)
                )
            )
            fig_vix.update_layout(
                title={"text":f"CBOE Volatility Index: VIX", "font": {"size": 25}},
                yaxis_title='VIX Value',
                #xaxis_title='Date',
                hovermode="x unified",
                #xaxis=dict(rangeslider_visible=True),
                margin=dict(l=20, r=20, t=40, b=20),
                height=300, 
            )
            fig_vix.add_hline(y=30, line_dash="dash", line_color="darkred", annotation_text="VIX 30 (High Volatility)", annotation_position="top left")
            st.plotly_chart(fig_vix, use_container_width=True)
        else:
            st.warning(f"No VIX data available for {PERIOD_CHART}.")

        st.caption("A VIX reading below 20 is generally considered a sign of a calm stock market, but when the value climbs above 30, it signals that there is significant fear in the market. Generally, a value rising above 30 suggests a buying signal.")
    #st.caption("Data provided by Yahoo Finance (via yfinance) and FRED (via fredapi).")

    st.write("")
    st.write("")
    ########################################################## Heat Map ##########################################################

    @st.cache_data(ttl=3600) 
    def get_all_stock_data(sectors_dict):
        # st.subheader("Market Heatmap")
        # st.caption("A performance overview of leading stocks across major sectors, where deeper shades of red indicate greater declines and deeper shades of green represent stronger gains.")
        all_tickers = [ticker for sublist in sectors_dict.values() for ticker in sublist]
        data = yf.download(all_tickers, period="2d", group_by='ticker')
        stock_info = []
        for sector, tickers in sectors_dict.items():
            for ticker in tickers:
                try:
                    if ticker in data.columns and len(data[ticker]['Close']) >= 2:
                        close_today = data[ticker]['Close'].iloc[-1]
                        close_yesterday = data[ticker]['Close'].iloc[-2]
                        perf = ((close_today - close_yesterday) / close_yesterday) * 100
                        market_cap = yf.Ticker(ticker).info.get('marketCap', 0)
                        long_name = yf.Ticker(ticker).info.get('longName', ticker)
                        stock_info.append({
                            'Sector': sector,
                            'Ticker': ticker,
                            'Company': long_name,
                            'Performance': perf,
                            'Market Cap': market_cap
                        })
                    else:
                        st.warning(f"Insufficient data for {ticker}. Skipping.")
                except Exception as e:
                    st.error(f"An error occurred for ticker {ticker} in sector {sector}: {e}")
        return pd.DataFrame(stock_info)

    sp500_sectors = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSM', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'INTC'],
        'Healthcare': ['LLY', 'JNJ', 'UNH', 'MRK', 'ABBV', 'PFE', 'TMO', 'AMGN', 'DHR', 'CVS'],
        'Financials': ['JPM', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'V', 'PYPL', 'SPGI', 'AXP'],
        'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'BKNG'],
        'Communication Services': ['META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T'],
        'Industrials': ['CAT', 'BA', 'HON', 'GE', 'RTX', 'MMM', 'UNP', 'LMT'],
        'Consumer Staples': ['WMT', 'PG', 'KO', 'COST', 'PEP', 'MDLZ', 'MO', 'UL'],
        'Energy': ['XOM', 'CVX', 'SLB', 'EOG'],
        'Utilities': ['NEE', 'DUK', 'SO', 'EXC', 'D'],
        'Materials': ['LIN', 'BHP', 'RIO', 'SHW', 'APD'],
        'Real Estate': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA']
    }

    with st.spinner('Fetching market data... This may take a moment for all sectors.'):
        df_all_sectors = get_all_stock_data(sp500_sectors)

    if not df_all_sectors.empty:
        df_all_sectors = df_all_sectors[df_all_sectors['Market Cap'] > 0].copy()
        fig = px.treemap(
            df_all_sectors,
            path=[px.Constant("Market Overview"), 'Sector', 'Ticker'],
            values='Market Cap',
            color='Performance',
            custom_data=['Company', 'Performance', 'Market Cap', 'Sector'],
            color_continuous_scale=[(0, 'rgb(210, 0, 0)'), (0.5, 'rgb(0, 51, 51)'), (1, 'rgb(0, 210, 0)')],
            color_continuous_midpoint=0,
        )
        fig.update_traces(
            textinfo="label+text",
            text=df_all_sectors['Performance'].apply(lambda x: f'{x:+.2f}%'),
            hovertemplate=(
                '<b>%{customdata[0]} (%{label})</b><br>'
                'Sector: %{customdata[3]}<br>'
                'Market Cap: %{customdata[2]:$,.0f}<br>'
                'Performance: %{customdata[1]:.2f}%<extra></extra>'
            )
        )
        fig.update_layout(
            title={
                'text': "Market Heatmap",
                # 'y':0.95,
                # 'x':0.05,
                'xanchor': 'left',
                'yanchor': 'top',
                'font': {'size': 25}
            },
            margin=dict(t=80, l=10, r=10, b=10),
            font=dict(size=14, color='#DDDDDD'),
            #paper_bgcolor='#1C1C1C',
            plot_bgcolor='#F6F7FA',
            height=550,
            coloraxis={'showscale': False},
            coloraxis_colorbar=dict(title="",tickfont=dict(color='#DDDDDD'),title_font=dict(color='#DDDDDD')),
        )
        fig.update_traces(textfont_color="white", selector=dict(type='treemap'))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The heatmap illustrates the most actively traded stock tickers and their relative performance across sectors. Lower-performing securities are represented in shades of red, while higher-performing securities are depicted in shades of green.")
    else:
        st.error("Could not fetch valid stock data for any sector. Please check ticker symbols or try again later.")

    st.write("")

    ########################################################## Sector comparison ##########################################################

    @st.cache_data(ttl=3600)
    def fetch_sector_data(etf_tickers, period="1y"):
        """Fetches adjusted close prices for given ETFs using yfinance."""
        data = yf.download(list(etf_tickers.keys()), period=period)['Close'] 
        return data

    @st.cache_data(ttl=3600)
    def calculate_relative_returns(data_df):
        """
        Calculates relative returns normalized to 0% at the start of the period.
        """
        if data_df.empty:
            return pd.DataFrame()
        daily_returns = data_df.pct_change()
        cumulative_returns = (1 + daily_returns.fillna(0)).cumprod()
        relative_returns_percentage = (cumulative_returns / cumulative_returns.iloc[0] * 100) - 100 
        return relative_returns_percentage
    data_load_state = st.info("Loading sector data... please wait.")
    etf_data = fetch_sector_data(sector_etfs, period="1y")
    data_load_state.empty() 
    if not etf_data.empty:
        #st.success("Data loaded successfully!")
        relative_returns_df = calculate_relative_returns(etf_data)
        relative_returns_df = relative_returns_df.rename(columns=sector_etfs)
        col1, col2 = st.columns([3,1]) 
        with col1:
            #st.subheader("Sector Relative Performance Over Last Year"
            df_melted = relative_returns_df.reset_index().melt(
                id_vars='Date',
                var_name='Sector',
                value_name='Relative Change (%)'
            )
            fig_lines = go.Figure()
            for sector in relative_returns_df.columns:
                df_subset = df_melted[df_melted['Sector'] == sector]
                
                line_width = 3.5 if sector == "S&P Index" else 1.5
                line_color = 'blue' if sector == "S&P Index" else None 

                fig_lines.add_trace(go.Scatter(
                    x=df_subset['Date'], 
                    y=df_subset['Relative Change (%)'], 
                    mode='lines',
                    name=sector,
                    line=dict(width=line_width, color=line_color) 
                ))
            fig_lines.update_layout(
                title={"text":f"Sector Performance Over 1 Year", "font": {"size": 25}},
                template="plotly_dark", 
                hovermode="x unified",
                #xaxis_title="Date",
                yaxis_title="Relative Change (%)",
                font=dict(size=12),
                #legend_title_text='Sector',
                # legend=dict(
                #     orientation="h",
                #     yanchor="bottom",
                #     y=1.02,
                #     xanchor="center",
                #     x=0.5
                # ),
                yaxis_range=[df_melted['Relative Change (%)'].min() - 2, df_melted['Relative Change (%)'].max() + 2],
                xaxis_rangeslider_visible=False,
                margin=dict(t=80, l=10, r=10, b=10),
                height=450 
            )    
            st.plotly_chart(fig_lines, use_container_width=True)

        with col2:
            st.write("")
            st.write("")
            #st.subheader("Total Sector Performance")
            total_performance = relative_returns_df.iloc[-1].rename("Total Return (%)").sort_values(ascending=False)
            total_performance_df = total_performance.reset_index()
            total_performance_df.columns = ['Sector', 'Total Return (%)']
            colors_bar = ['#8CC152' if x >= 0 else '#DA4453' for x in total_performance_df['Total Return (%)']]
            fig_bar = px.bar(
                total_performance_df,
                x='Total Return (%)',
                y='Sector',
                orientation='h',
                title='', 
                height=450 
            )
            fig_bar.update_traces(marker_color=colors_bar)
            max_abs_return = total_performance_df['Total Return (%)'].abs().max()
            CAP_LIMIT = max_abs_return * 1.05 
            max_limit = max(abs(total_performance_df['Total Return (%)'].min()), total_performance_df['Total Return (%)'].max()) * 1.05
            fig_bar.update_layout(
                #title={"text":f"Sector Performance", "font": {"size": 25}},
                template="plotly_dark",
                yaxis_title="", 
                xaxis_title="Percentage Change",
                font=dict(size=12),
                showlegend=False, 
                xaxis_tickformat=".0f", 
                xaxis_range=[-max_limit, max_limit],
            )
            fig_bar.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error("Failed to load data for sector ETFs. Please check ticker symbols or your internet connection.")

###################################################################################################################

with crypto_market_data:

    CRYPTO_TICKERS = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'XRP': 'XRP-USD',
        'BNB': 'BNB-USD',
        'Solana': 'SOL-USD',
        'Cardano': 'ADA-USD',
        'Dogecoin': 'DOGE-USD',
        'TRON': 'TRX-USD',
        'Polkadot': 'DOT-USD',
        'Chainlink': 'LINK-USD',
        'Polygon': 'MATIC-USD',
        'Litecoin': 'LTC-USD',
        'Bitcoin Cash': 'BCH-USD',
        'Avalanche': 'AVAX-USD',
        'Shiba Inu': 'SHIB-USD',
        'Uniswap': 'UNI-USD'
    }

    MAIN_TICKERS = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'XRP': 'XRP-USD',
        'BNB': 'BNB-USD',
        }

    CHART_COIN_NAME = 'Bitcoin'
    CHART_COIN_NAME2 = 'Ethereum'
    PERIOD_CRYPTO_CHART = "1y"
    EMA_PERIODS = [20, 50, 200]
    EMA_COLORS = {
        20: 'green',
        50: 'gold',
        200: 'red'
    }

    @st.cache_data(ttl=3600)
    def get_crypto_current_data(tickers):
        data = {}
        ticker_list = list(tickers.values())
        df_all_historical = yf.download(ticker_list, period="5d", interval="1d", progress=False)
        for name, ticker in tickers.items():
            try:
                if len(ticker_list) == 1 or not isinstance(df_all_historical['Close'], pd.DataFrame):
                    df_hist = df_all_historical['Close']
                else:
                    df_hist = df_all_historical['Close'][ticker]
                latest_close = 0
                daily_change = 0
                daily_change_percent = 0
                if not df_hist.empty and len(df_hist) >= 2:
                    latest_close = df_hist.iloc[-1]
                    previous_close = df_hist.iloc[-2]
                    daily_change = latest_close - previous_close
                    daily_change_percent = (daily_change / previous_close) * 100
                elif not df_hist.empty:
                    latest_close = df_hist.iloc[-1]
                    daily_change_percent = 0 
                else:
                    pass 
                info = yf.Ticker(ticker).info
                market_cap = info.get('marketCap', 0) 
                data[name] = {
                    'price': latest_close,
                    'delta': f'{daily_change:+.2f} ({daily_change_percent:+.2f}%)' if latest_close != 0 else "N/A",
                    'raw_delta': daily_change,
                    'market_cap': market_cap,
                    'performance_percent': daily_change_percent 
                }
            except Exception as e:
                data[name] = {'price': 0, 'delta': "Error", 'raw_delta': 0, 'market_cap': 0, 'performance_percent': 0}
        return data

    @st.cache_data(ttl=3600)
    def get_crypto_historical_data_with_emas(ticker_name, ticker, period, ema_periods):
        end_date = datetime.datetime.today()
        start_date = end_date - relativedelta(days=365 * 2)
        df_series = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
        if df_series.empty:
            return pd.DataFrame()

        df = pd.DataFrame(df_series)
        df.columns = ['Price']

        for p in ema_periods:
            df[f'EMA {p}'] = df['Price'].ewm(span=p, adjust=False).mean()

        if period.endswith('y'):
            years = int(period[:-1])
            data_start_date = end_date - relativedelta(days=365 * years)
            df_filtered = df[df.index >= data_start_date]
        else:
            df_filtered = df

        return df_filtered

    @st.cache_data(ttl=3600)
    def fetch_all_crypto_data(tickers, period="1y"):
        """Fetches adjusted close prices for given crypto tickers using yfinance."""
        data = yf.download(list(tickers.values()), period=period)['Close']
        return data

    @st.cache_data(ttl=3600)
    def calculate_relative_returns(data_df):
        """
        Calculates relative returns normalized to 0% at the start of the period.
        """
        if data_df.empty:
            return pd.DataFrame()
        daily_returns = data_df.pct_change()
        cumulative_returns = (1 + daily_returns.fillna(0)).cumprod()
        relative_returns_percentage = (cumulative_returns / cumulative_returns.iloc[0] * 100) - 100
        return relative_returns_percentage

    @st.cache_data(ttl=3600)
    def create_correlation_heatmap(tickers, period="1y"):
        data_df = fetch_all_crypto_data(tickers, period=period)
        if data_df.empty:
            return None
        returns_df = data_df.pct_change().dropna()
        correlation_matrix = returns_df.corr()
        coin_map = {v: k for k, v in MAIN_TICKERS.items()}
        correlation_matrix.rename(index=coin_map, columns=coin_map, inplace=True)
        fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=correlation_matrix.values.round(2),
                texttemplate="%{text}",
                textfont={"size":10}
            ))
        fig.update_layout(
            title={"text": "Daily Return Correlation Heatmap", "font": {"size": 25}},
            #xaxis=dict(title="Coin", tickangle=-45),
            #yaxis=dict(title="Coin"),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
            width=400
        )
        return fig

    @st.cache_data(ttl=3600) 
    def create_crypto_market_heatmap(current_crypto_data):
        """
        Creates a treemap heatmap showing crypto market cap and daily performance.
        """
        heatmap_data = []
        for name, data in current_crypto_data.items():
            if data['market_cap'] > 0: # Only include coins with valid market cap
                heatmap_data.append({
                    'Coin': name,
                    'Market Cap': data['market_cap'],
                    'Performance': data['performance_percent']
                })
        if not heatmap_data:
            return None

        df_heatmap = pd.DataFrame(heatmap_data)
        df_heatmap = df_heatmap.sort_values(by='Market Cap', ascending=False)

        fig = px.treemap(
            df_heatmap,
            path=[px.Constant("Crypto Market"), 'Coin'], # Group under a single "Crypto Market" root
            values='Market Cap',
            color='Performance',
            color_continuous_scale=[(0, 'rgb(210, 0, 0)'), (0.5, 'rgb(0, 51, 51)'), (1, 'rgb(0, 210, 0)')], # Red to Green
            color_continuous_midpoint=0, # Center the color scale at 0% change
            hover_data=['Performance', 'Market Cap']
        )
        fig.update_traces(
            textinfo="label+text",
            text=df_heatmap['Performance'].apply(lambda x: f'{x:+.2f}%'),
            hovertemplate=(
                '<b>%{label}</b><br>'
                'Market Cap: %{customdata[1]:$,.0f}<br>'
                'Performance: %{customdata[0]:.2f}%<extra></extra>'
            )
        )
        fig.update_layout(
            title={
                'text': "Crypto Market Heatmap",
                'xanchor': 'left', 'yanchor': 'top', 'font': {'size': 25}
            },
            margin=dict(t=80, l=10, r=10, b=10),
            height=550,
            coloraxis={'showscale': False}
        )
        fig.update_traces(textfont_color="white", selector=dict(type='treemap'))
        return fig

    def crypto_dashboard():
        st.header("Crypto Market Overview", divider="rainbow")
        st.subheader("Daily Performance")
        
        current_data_for_metrics_and_heatmap = get_crypto_current_data(CRYPTO_TICKERS)
        top_coins_for_metrics = list(current_data_for_metrics_and_heatmap.items())[:4] # Display top 4 in metrics
        cols = st.columns(len(top_coins_for_metrics))
        
        for i, (name, data) in enumerate(top_coins_for_metrics):
            price_str = f"${data['price']:,.2f}" if isinstance(data['price'], (int, float)) and data['price'] != 0 else data['price']
            cols[i].metric(
                label=name,
                value=price_str,
                delta=data['delta'],
                delta_color="normal"
            )
        
        st.write("")
        st.write("")

        #########################################################################################################################################################

        st.write("")
        st.write("")
        
        chartcol1, chartcol2 = st.columns(2)
        with chartcol1:
            chart_ticker = CRYPTO_TICKERS[CHART_COIN_NAME]
            historical_df = get_crypto_historical_data_with_emas(CHART_COIN_NAME, chart_ticker, PERIOD_CRYPTO_CHART, EMA_PERIODS)
            if not historical_df.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=historical_df.index,
                        y=historical_df['Price'],
                        mode='lines',
                        name='Price',
                        line=dict(color='#FF9900', width=3)
                    )
                )
                for p in EMA_PERIODS:
                    line_color = EMA_COLORS.get(p, 'gray')

                    fig.add_trace(
                        go.Scatter(
                            x=historical_df.index,
                            y=historical_df[f'EMA {p}'],
                            mode='lines',
                            name=f'EMA {p}',
                            line=dict(color=line_color, dash='dot', width=1.5)
                        )
                    )
                fig.update_layout(
                    title={"text":f"{CHART_COIN_NAME} Price Chart", "font": {"size": 25}},
                    yaxis_title='Price (USD)',
                    yaxis_type="log",
                    hovermode="x unified",
                    legend_title="Indicators",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No {PERIOD_CRYPTO_CHART} data available for {CHART_COIN_NAME}.")
            
        with chartcol2:
            chart_ticker = CRYPTO_TICKERS[CHART_COIN_NAME2]
            historical_df = get_crypto_historical_data_with_emas(CHART_COIN_NAME2, chart_ticker, PERIOD_CRYPTO_CHART, EMA_PERIODS)
            if not historical_df.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=historical_df.index,
                        y=historical_df['Price'],
                        mode='lines',
                        name='Price',
                        line=dict(color='#5E9BEB', width=3)
                    )
                )
                for p in EMA_PERIODS:
                    line_color = EMA_COLORS.get(p, 'gray')

                    fig.add_trace(
                        go.Scatter(
                            x=historical_df.index,
                            y=historical_df[f'EMA {p}'],
                            mode='lines',
                            name=f'EMA {p}',
                            line=dict(color=line_color, dash='dot', width=1.5)
                        )
                    )
                fig.update_layout(
                    title={"text":f"{CHART_COIN_NAME2} Price Chart", "font": {"size": 25}},
                    yaxis_title='Price (USD)',
                    yaxis_type="log",
                    hovermode="x unified",
                    legend_title="Indicators",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No {PERIOD_CRYPTO_CHART} data available for {CHART_COIN_NAME2}.")

        st.write("")
        st.write("")
        ########################################################## Heat Map ##########################################################

        market_heatmap_fig = create_crypto_market_heatmap(current_data_for_metrics_and_heatmap)
        if market_heatmap_fig:
            st.plotly_chart(market_heatmap_fig, use_container_width=True)
        else:
            st.warning("Could not generate crypto market heatmap.")
        st.caption("The heatmap illustrates the most actively traded crypto tickers. Lower-performing securities are represented in shades of red, while higher-performing securities are depicted in shades of green.")
            
        st.write("")
        st.write("")

        ########################################################## charts ##########################################################

        st.write("")
        st.write("")

        col_chart, col_correlation_heatmap = st.columns([2, 1])
        with col_chart:
            data_load_state = st.info("Loading crypto history data for comparison... please wait.")
            crypto_history_data = fetch_all_crypto_data(MAIN_TICKERS, period=PERIOD_CRYPTO_CHART)
            data_load_state.empty()
            if not crypto_history_data.empty:
                coin_map = {v: k for k, v in MAIN_TICKERS.items()}
                crypto_history_data = crypto_history_data.rename(columns=coin_map)
                relative_returns_df = calculate_relative_returns(crypto_history_data)
                df_melted = relative_returns_df.reset_index().melt(
                    id_vars='Date',
                    var_name='Coin',
                    value_name='Relative Change (%)'
                )
                fig_perf = px.line(
                    df_melted,
                    x='Date',
                    y='Relative Change (%)',
                    color='Coin',
                    title='Crypto Performance Comparison (Start Date Normalized to 0%)',
                    height=430
                )
                fig_perf.update_traces(
                hovertemplate=(
                    '<b>%{fullData.name}</b>: %{y:.2f}%'
                    '<extra></extra>'
                    )
                )   
                fig_perf.update_layout(
                    title={"text":f"Major Crypto Performance Over 1 Year", "font": {"size": 25}},
                    hovermode="x unified",
                    template="plotly_dark",
                    xaxis_title="",
                    yaxis_title="Relative Change (%)",
                    font=dict(size=12),
                    yaxis_range=[df_melted['Relative Change (%)'].min() - 2, df_melted['Relative Change (%)'].max() + 2],
                    margin=dict(t=40, l=20, r=20, b=20),
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig_perf, use_container_width=True)

            else:
                st.error(f"Failed to load historical data for crypto tickers.")
            
        with col_correlation_heatmap:
            correlation_heatmap_fig = create_correlation_heatmap(MAIN_TICKERS, period=PERIOD_CRYPTO_CHART)
            if correlation_heatmap_fig:
                st.plotly_chart(correlation_heatmap_fig, use_container_width=True)
            else:
                st.warning("Could not generate correlation heatmap.")
    crypto_dashboard()

###########################################################################################################################################

with commodity_market_data:
    COMMODITY_TICKERS = {
        'Gold (GC=F)': 'GC=F',
        'Silver (SI=F)': 'SI=F',
        'Crude Oil (CL=F)': 'CL=F',
    }
    CORRELATION_PAIRS = {
        'Gold': 'GC=F',
        'S&P 500': '^GSPC',
        'Bitcoin': 'BTC-USD' 
    }
    PERIOD_CHART = "1y"
    EMA_PERIODS = [20, 50, 200]
    EMA_COLORS = {
        20: 'green',
        50: 'gold',
        200: 'red'
    }
    COMMODITY_COLORS = {
        'Gold': 'orange',   
        'Silver': '#FB6E51', 
        'Crude Oil': '#AAB2BD', 
    }

    @st.cache_data(ttl=3600)
    def get_commodity_current_data(tickers):
        data = {}
        ticker_list = list(tickers.values())
        df_all = yf.download(ticker_list, period="5d", interval="1d", progress=False)
        for name, ticker in tickers.items():
            try:
                if len(ticker_list) == 1 or not isinstance(df_all['Close'], pd.DataFrame):
                    df = df_all['Close']
                else:
                    df = df_all['Close'][ticker]
                
                if not df.empty and len(df) >= 2:
                    latest_close = df.iloc[-1]
                    previous_close = df.iloc[-2]
                    daily_change = latest_close - previous_close
                    daily_change_percent = (daily_change / previous_close) * 100
                    price_str = f"{latest_close:,.2f}"
                    data[name] = {
                        'price': price_str,
                        'delta': f'{daily_change:+.2f} ({daily_change_percent:+.2f}%)',
                        'raw_delta': daily_change
                    }
                else:
                    data[name] = {'price': "N/A", 'delta': "N/A", 'raw_delta': 0}
            except Exception as e:
                data[name] = {'price': "Error", 'delta': "Error", 'raw_delta': 0}
        return data

    @st.cache_data(ttl=3600)
    def get_commodity_historical_data_with_emas(ticker_name, ticker, period, ema_periods):
        end_date = datetime.datetime.today()
        start_date = end_date - relativedelta(days=365 * 2) 
        df_series = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
        if df_series.empty:
            return pd.DataFrame()
        df = pd.DataFrame(df_series)
        df.columns = ['Price']
        for p in ema_periods:
            df[f'EMA {p}'] = df['Price'].ewm(span=p, adjust=False).mean().fillna(method='bfill')
        if period.endswith('y'):
            years = int(period[:-1]) 
            data_start_date = end_date - relativedelta(days=365 * years)
            df_filtered = df[df.index >= data_start_date]
        else:
            df_filtered = df 
        return df_filtered

    @st.cache_data(ttl=3600)
    def fetch_all_correlation_data(tickers, period="1y"):
        data = yf.download(list(tickers.values()), period=period)['Close'] 
        ticker_to_name = {v: k for k, v in tickers.items()}
        data.columns = [ticker_to_name.get(col, col) for col in data.columns]
        return data

    @st.cache_data(ttl=3600)
    def create_correlation_heatmap(tickers, period="1y"):
        data_df = fetch_all_correlation_data(tickers, period=period)   
        if data_df.empty:
            return None
        returns_df = data_df.pct_change().dropna()
        correlation_matrix = returns_df.corr()
        fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',  # Red-Blue scale for correlation (negative to positive)
                zmin=-1, 
                zmax=1, 
                text=correlation_matrix.values.round(2), # Display text in cells
                texttemplate="%{text}",
                textfont={"size":10}
            ))
        fig.update_layout(
            title={"text": "Daily Return Correlation Matrix", "font": {"size": 25}},
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'), # Keep Y-axis order intuitive
            margin=dict(l=20, r=20, t=40, b=20),
            height=450,
            width=450 
        )
        return fig

    ##########################################################   ##########################################################

    def commodity_dashboard():  
        st.header("Commodity Market Overview", divider ="rainbow")
        st.subheader("Daily Performance")

        current_data = get_commodity_current_data(COMMODITY_TICKERS)
        cols = st.columns(len(COMMODITY_TICKERS))
        index_names = list(COMMODITY_TICKERS.keys())

        for i, name in enumerate(index_names):
            data = current_data[name]
            cols[i].metric(
                label=name.split('(')[0].strip(),
                value=f"${data['price']}",
                delta=data['delta'],
                delta_color="normal"
            )
        
        st.write("")
        st.write("")

        ########################################################## charts  ##########################################################
        
        st.write("")
        st.write("")
        
        chart_cols = st.columns(len(COMMODITY_TICKERS))
        
        for i, (name, ticker) in enumerate(COMMODITY_TICKERS.items()):
            with chart_cols[i]:
                simple_name = name.split('(')[0].strip()
                price_line_color = COMMODITY_COLORS.get(simple_name, 'orange')
                #st.subheader(f" {name.split('(')[0].strip()}")
                historical_df = get_commodity_historical_data_with_emas(name, ticker, PERIOD_CHART, EMA_PERIODS)
                if not historical_df.empty:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=historical_df.index, 
                            y=historical_df['Price'], 
                            mode='lines', 
                            name='Price',
                            line=dict(color=price_line_color, width=1)
                        )
                    )
                    for p in EMA_PERIODS:
                        line_color = EMA_COLORS.get(p, 'gray') 
                        
                        fig.add_trace(
                            go.Scatter(
                                x=historical_df.index, 
                                y=historical_df[f'EMA {p}'], 
                                mode='lines', 
                                name=f'EMA {p}',
                                line=dict(color=line_color, dash='dot', width=1.5)
                            )
                        )
                    fig.update_layout(
                        title={"text":f"{name.split('(')[0].strip()} Over {PERIOD_CHART}", "font": {"size": 25}},
                        yaxis_title='Price (USD)',
                        hovermode="x unified",
                        legend_title="Indicators",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=300, 
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No {PERIOD_CHART} data available for {name}.")
                    
        st.write("")
        st.write("")

        ########################################################## correlation  ##########################################################
        
        st.write("")
        st.write("")

        corr_col_1, corr_col_2 = st.columns([1, 2])
        with corr_col_1:
            correlation_heatmap_fig = create_correlation_heatmap(CORRELATION_PAIRS, period=PERIOD_CHART)
            if correlation_heatmap_fig:
                st.plotly_chart(correlation_heatmap_fig, use_container_width=True)
            else:
                st.error("Could not generate correlation heatmap.")  
        with corr_col_2:
            st.markdown(
                """
                ### Correlation Interpretation
                * **Value near +1.0**: The assets move in the **same direction** (Strong Positive Correlation).
                * **Value near -1.0**: The assets move in the **opposite direction** (Strong Negative Correlation).
                * **Value near 0.0**: The assets have **no predictable relationship** (No Correlation).
                
                Historically:
                * **Gold vs. S&P 500**: Often shows **low or negative correlation**, as gold is sometimes seen as a "safe-haven" asset when stocks fall.
                * **Gold vs. Bitcoin**: The relationship is **new and evolving**, often swinging from low correlation to moderate positive correlation.
                """
            )

    commodity_dashboard()

''
st.subheader("", divider ='gray')
iiqc1, iiqc2 = st.columns ([3,1])
with iiqc1:
    st.write("")
    st.markdown("**Disclaimer:**")
    st.write("This analysis dashboard is designed to enable beginner investors to analyze stocks effectively and with ease. Please note that the information in this page is intended for educational purposes only and it does not constitute investment advice or a recommendation to buy or sell any security. We are not responsible for any losses resulting from trading decisions based on this information.")
with iiqc2:
    invest_iq_central='./Image/InvestIQCentral.png'
    st.image(invest_iq_central,width=300)
''

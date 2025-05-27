import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import math
import numpy as np
import http.client
import json
import pandas as pd
import plotly.graph_objects as go
import datetime
import re
from dateutil.relativedelta import relativedelta
import pytz
from groq import Groq

st.set_page_config(page_title='US Stock Analysis Tool', layout='wide', page_icon="./Image/logo.png")

@st.cache_data(ttl=3600)
def get_stock_data(ticker1, ticker2):

    stock1 = yf.Ticker(ticker1)
    upper_ticker1 = ticker1.upper()
    lower_ticker1 = ticker1.lower()
    logo1 = f'https://logos.stockanalysis.com/{lower_ticker1}.svg'

    stock2 = yf.Ticker(ticker2)
    upper_ticker2 = ticker2.upper()
    lower_ticker2 = ticker2.lower()
    logo2 = f'https://logos.stockanalysis.com/{lower_ticker2}.svg'

    # Item Colors
    color1 = '#3BAFDA'
    color2 = '#E9573F'
    logo_background_color = 'rgba(197, 198, 199, 0.3)'

    # Chart Margin
    margin_top = 60
    margin_bottom = 40
    margin_left = 40
    margin_right = 30

    # Chart Legend
    legend_yanchor = "top"
    legend_y = 1.1
    legend_xanchor = "left"
    legend_x = 0.01
    legend_orientation = 'h'

    # Chart Height
    chart_height = 350

    # Title Font Size
    title_font_size = 20

    ##### Income Statement #####
    try:
        income_statement_tb1 = stock1.income_stmt
        quarterly_income_statement_tb1 = stock1.quarterly_income_stmt
    except: income_statement_tb1 = quarterly_income_statement_tb1 = ""
    try:
        income_statement_tb2 = stock2.income_stmt
        quarterly_income_statement_tb2 = stock2.quarterly_income_stmt
    except: income_statement_tb2 = quarterly_income_statement_tb2 = ""
    ##### Balance Sheet #####
    try:
        balance_sheet_tb1 = stock1.balance_sheet
        quarterly_balance_sheet_tb1 = stock1.quarterly_balance_sheet
    except: balance_sheet_tb1 = quarterly_balance_sheet_tb1 = ""
    try:
        balance_sheet_tb2 = stock2.balance_sheet
        quarterly_balance_sheet_tb2 = stock2.quarterly_balance_sheet
    except: balance_sheet_tb2 = quarterly_balance_sheet_tb2 = ""
    ##### Cashflow Statement #####
    try:
        cashflow_statement_tb1 = stock1.cashflow
        quarterly_cashflow_statement_tb1 = stock1.quarterly_cashflow
    except: cashflow_statement_tb1 = quarterly_cashflow_statement_tb1 = ""
    try:
        cashflow_statement_tb2 = stock2.cashflow
        quarterly_cashflow_statement_tb2 = stock2.quarterly_cashflow
    except: cashflow_statement_tb2 = quarterly_cashflow_statement_tb2 = ""
    ########################

    # Income Statement
    try: 
        income_statement1 = income_statement_tb1 
        quarterly_income_statement1 = quarterly_income_statement_tb1
        ttm1 = quarterly_income_statement1.iloc[:, :4].sum(axis=1)
        income_statement1.insert(0, 'TTM', ttm1)
        income_statement_flipped1 = income_statement1.iloc[::-1]
    except: income_statement_flipped1 =''
    try: 
        income_statement2 = income_statement_tb2 
        quarterly_income_statement2 = quarterly_income_statement_tb2
        ttm2 = quarterly_income_statement2.iloc[:, :4].sum(axis=1)
        income_statement2.insert(0, 'TTM', ttm2)
        income_statement_flipped2 = income_statement2.iloc[::-1]
    except: income_statement_flipped2 =''
    # Balance Sheet Statement
    try:
        balance_sheet1 = balance_sheet_tb1
        quarterly_balance_sheet1 = quarterly_balance_sheet_tb1
        ttm1 = quarterly_balance_sheet1.iloc[:, :4].sum(axis=1)
        balance_sheet1.insert(0, 'TTM', ttm1)
        balance_sheet_flipped1 = balance_sheet1.iloc[::-1]
    except: balance_sheet_flipped1 = ''
    try:
        balance_sheet2 = balance_sheet_tb2
        quarterly_balance_sheet2 = quarterly_balance_sheet_tb2
        ttm2 = quarterly_balance_sheet2.iloc[:, :4].sum(axis=1)
        balance_sheet2.insert(0, 'TTM', ttm2)
        balance_sheet_flipped2 = balance_sheet2.iloc[::-1]
    except: balance_sheet_flipped2 = ''
    # Cash Flow Statement
    try:
        cashflow_statement1 = cashflow_statement_tb1
        quarterly_cashflow_statement1 = quarterly_cashflow_statement_tb1
        ttm1 = quarterly_cashflow_statement1.iloc[:, :4].sum(axis=1)
        cashflow_statement1.insert(0, 'TTM', ttm1)
        cashflow_statement_flipped1 = cashflow_statement1.iloc[::-1]
    except: cashflow_statement_flipped1 = ''
    try:
        cashflow_statement2 = cashflow_statement_tb2
        quarterly_cashflow_statement2 = quarterly_cashflow_statement_tb2
        ttm2 = quarterly_cashflow_statement2.iloc[:, :4].sum(axis=1)
        cashflow_statement2.insert(0, 'TTM', ttm2)
        cashflow_statement_flipped2 = cashflow_statement2.iloc[::-1]
    except: cashflow_statement_flipped2 = ''
    ########################

    # Price Performance Comparison
    try:
        compare_tickers = (upper_ticker1, upper_ticker2)
        end = datetime.datetime.today()
        start = end - relativedelta(years=5)
        def relativereturn(df):
            rel = df.pct_change()
            cumret = (1+rel).cumprod()-1
            cumret = cumret.fillna(0)
            cumret = cumret*100
            return cumret
        price_performance_com = relativereturn(yf.download(compare_tickers, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))['Close'])
    except: price_performance_com =  ""
    ########################

    # StockAnalysis Metrics
    try:
        url = f'https://stockanalysis.com/stocks/{ticker1}/financials/ratios/'
        r = requests.get(url)
        soup = BeautifulSoup(r.text,"html.parser")
        table = soup.find("table",class_ = "w-full border-separate border-spacing-0 text-sm sm:text-base [&_tbody]:sm:whitespace-nowrap [&_thead]:whitespace-nowrap")
        rows = table.find_all("tr")
        headers = []
        data = []
        for row in rows:
                    cols = row.find_all(["th", "td"])
                    cols_text = [col.text.strip() for col in cols]
                    if not headers:
                        headers = cols_text
                    else:
                        data.append(cols_text)
        sa_metrics_df1 = pd.DataFrame(data, columns=headers)
        sa_metrics_df1 = sa_metrics_df1.iloc[1:, :-1].reset_index(drop=True)
    except: sa_metrics_df1 = ""
    try:
        url = f'https://stockanalysis.com/stocks/{ticker2}/financials/ratios/'
        r = requests.get(url)
        soup = BeautifulSoup(r.text,"html.parser")
        table = soup.find("table",class_ = "w-full border-separate border-spacing-0 text-sm sm:text-base [&_tbody]:sm:whitespace-nowrap [&_thead]:whitespace-nowrap")
        rows = table.find_all("tr")
        headers = []
        data = []
        for row in rows:
                    cols = row.find_all(["th", "td"])
                    cols_text = [col.text.strip() for col in cols]
                    if not headers:
                        headers = cols_text
                    else:
                        data.append(cols_text)
        sa_metrics_df2 = pd.DataFrame(data, columns=headers)
        sa_metrics_df2 = sa_metrics_df2.iloc[1:, :-1].reset_index(drop=True)
    except: sa_metrics_df2 = ""
    ########################

    ##### Profitability #####
    roe1 = stock1.info.get('returnOnEquity','N/A')
    roa1 = stock1.info.get('returnOnAssets','N/A')
    roe2 = stock2.info.get('returnOnEquity','N/A')
    roa2 = stock2.info.get('returnOnAssets','N/A')
    ##### Margin #####
    profitmargin1 = stock1.info.get('profitMargins','N/A')
    grossmargin1 = stock1.info.get('grossMargins','N/A')
    operatingmargin1 = stock1.info.get('operatingMargins','N/A')
    profitmargin2 = stock2.info.get('profitMargins','N/A')
    grossmargin2 = stock2.info.get('grossMargins','N/A')
    operatingmargin2 = stock2.info.get('operatingMargins','N/A')

    return ticker1, upper_ticker1, lower_ticker1, logo1, color1, \
        logo_background_color, \
        ticker2, upper_ticker2, lower_ticker2, logo2, color2, \
        margin_top, margin_bottom, margin_left, margin_right, \
        legend_yanchor, legend_y, legend_xanchor, legend_x, legend_orientation, \
        chart_height, title_font_size, \
        income_statement_flipped1, income_statement_flipped2, \
        balance_sheet_flipped1, balance_sheet_flipped2, \
        cashflow_statement_flipped1, cashflow_statement_flipped2, \
        price_performance_com, \
        sa_metrics_df1, sa_metrics_df2, \
        roe1, roe2, roa1, roa2, \
        profitmargin1, profitmargin2, grossmargin1, grossmargin2, operatingmargin1, operatingmargin2



main_col1, main_col2 = st.columns([3,1])
with main_col1:
    st.title("Face-off Analysis Tool (Beta)")
    input_col1, input_col2, input_col3 = st.columns([1, 1, 1])
    with input_col1:
        ticker1 = st.text_input("Ticker1:", "AAPL")
    with input_col2:
        ticker2 = st.text_input("Ticker2:", "MSFT")

""

if st.button("Get Data"):
    try:
        ticker1, upper_ticker1, lower_ticker1, logo1, color1, \
        logo_background_color, \
        ticker2, upper_ticker2, lower_ticker2, logo2, color2, \
        margin_top, margin_bottom, margin_left, margin_right, \
        legend_yanchor, legend_y, legend_xanchor, legend_x, legend_orientation, \
        chart_height, title_font_size, \
        income_statement_flipped1, income_statement_flipped2, \
        balance_sheet_flipped1, balance_sheet_flipped2, \
        cashflow_statement_flipped1, cashflow_statement_flipped2, \
        price_performance_com, \
        sa_metrics_df1, sa_metrics_df2, \
        roe1, roe2, roa1, roa2, \
        profitmargin1, profitmargin2, grossmargin1, grossmargin2, operatingmargin1, operatingmargin2 = get_stock_data(ticker1, ticker2)

        st.divider()
        
        col1,col2,col3,col4,col5 = st.columns([3,2,1,2,3])
        with col2:
            st.markdown(f"""
                <div style="display: flex; justify-content: center; background-color: {logo_background_color}; padding: 10px; border-radius: 10px;">
                    <img src="{logo1}" width="100">
                </div>
                """,unsafe_allow_html=True)
        # with col3:
        #     st.markdown(f"""
        #         <div style='display: flex; justify-content: center; align-items: center; height: 100%; padding: 30px;'>
        #             <h3 style='margin: 0;'>VS</h3>
        #         </div>
        #         """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div style="display: flex; justify-content: center; background-color: {logo_background_color}; padding: 10px; border-radius: 10px;">
                    <img src="{logo2}" width="100">
                </div>
                """,unsafe_allow_html=True)

        st.write("")
        st.write("")
        st.write("")

        # Price Performance Comparison
        col1, col2 = st.columns([3,1])
        try:
            with col1:
                price_performance_com_df = price_performance_com
                price_performance_com_df_melted = price_performance_com_df.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Relative Return')
                unique_years_sorted = price_performance_com_df_melted['Date'].dt.year.unique()
                custom_colors = {
                        upper_ticker1: color1,  
                        upper_ticker2: color2,
                }
                def plot_relative_return_chart(price_performance_com_df_melted, custom_colors):
                    df_plot = price_performance_com_df_melted.copy()
                    fig = go.Figure()
                    for ticker in df_plot['Ticker'].unique():
                        df_ticker = df_plot[df_plot['Ticker'] == ticker]
                        show_labels = True if ticker == df_plot['Ticker'].unique()[-1] else False
                        fig.add_trace(
                            go.Scatter(
                                x=df_ticker['Date'],
                                y=df_ticker['Relative Return'],
                                mode='lines',
                                name=ticker,
                                line=dict(color=custom_colors.get(ticker), shape='spline', smoothing=1.3),
                                showlegend=True,
                                hoverinfo="text",
                                text=[f"{date}: {ret:.2f}%" for date, ret in zip(df_ticker['Date'], df_ticker['Relative Return'])]
                            )
                        )
                    fig.update_layout(
                        title={"text":f'5-Year Price Performance', "font": {"size": title_font_size}},
                        title_y=1,  
                        title_x=0, 
                        margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                        xaxis=dict(title=None, showticklabels=show_labels, showgrid=True), 
                        yaxis=dict(title="Cumulative Relative Return (%)", showgrid=True),
                        legend=dict(yanchor=legend_yanchor,y=legend_y,xanchor=legend_xanchor,x=legend_x, orientation = legend_orientation),
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                plot_relative_return_chart(price_performance_com_df_melted, custom_colors)
            with col2:
                last_values = price_performance_com_df_melted.groupby('Ticker').last()
                st.metric(
                    label=upper_ticker1,
                    value=f"{last_values.loc[upper_ticker1, 'Relative Return']:.2f}%"
                )
                st.metric(
                    label=upper_ticker2,
                    value=f"{last_values.loc[upper_ticker2, 'Relative Return']:.2f}%"
                )
                best_performer = last_values['Relative Return'].idxmax()
                best_return = last_values.loc[best_performer, 'Relative Return']
                summary = f"{best_performer} showed the strongest performance with {best_return:.2f}% return."
                st.write(summary) 
        except Exception as e:
            st.warning(f'Error getting historical data.')

        st.divider()

        # Income Statement Comparison
        col1, col2, col3 = st.columns([3,3,3])
        # Revenue Comparison
        with col1:
            try:
                if not isinstance(income_statement_flipped1, str) and not isinstance(income_statement_flipped2, str):
                    revenue1 = income_statement_flipped1.loc['Total Revenue']
                    revenue2 = income_statement_flipped2.loc['Total Revenue']
                    df1 = pd.DataFrame({
                        'Date': list(revenue1.index),
                        'Revenue': revenue1.values.flatten(),
                        'Stock': [upper_ticker1] * len(revenue1)
                    })
                    df2 = pd.DataFrame({
                        'Date': list(revenue2.index),
                        'Revenue': revenue2.values.flatten(),
                        'Stock': [upper_ticker2] * len(revenue2)
                    })
                    combined_df = pd.concat([df1, df2])
                    combined_df['Revenue'] = combined_df['Revenue'] / 1e6
                    date_values = [x for x in combined_df['Date'].unique() if x != 'TTM']
                    formatted_dates = list(set([pd.to_datetime(date).strftime('%Y') for date in date_values]))
                    sorted_dates = sorted(formatted_dates) + ['TTM']
                    date_mapping = {date: pd.to_datetime(date).strftime('%Y') for date in date_values}
                    date_mapping['TTM'] = 'TTM'
                    combined_df['Date'] = combined_df['Date'].map(date_mapping)
                    combined_df['Date'] = pd.Categorical(combined_df['Date'], 
                                                        categories=sorted_dates,
                                                        ordered=True)
                    colors = {
                        upper_ticker1: color1,
                        upper_ticker2: color2
                    }
                    fig = go.Figure()
                    for stock in [upper_ticker1, upper_ticker2]:
                        stock_data = combined_df[combined_df['Stock'] == stock]
                        fig.add_trace(
                            go.Bar(
                                x=stock_data['Date'],
                                y=stock_data['Revenue'],
                                name=stock,
                                marker_color=colors[stock],
                                marker_line_width=0,
                                marker=dict(
                                    cornerradius=30
                                ),
                                hovertemplate=
                                "Ticker: %{customdata}<br>" +
                                "Year: %{x}<br>" +
                                "Value: $%{y:,.0f}M<br>" +
                                "<extra></extra>",
                                customdata=[stock] * len(stock_data)
                            )
                        )
                    fig.update_layout(
                        title={"text": f"Revenue", "font": {"size": title_font_size}},
                        title_y=1,
                        title_x=0,
                        margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                        barmode='group',
                        xaxis_title=None,
                        yaxis_title='USD in Million',
                        legend=dict(yanchor=legend_yanchor, y=legend_y, xanchor=legend_xanchor, x=legend_x, orientation = legend_orientation),
                        height=chart_height
                    )
                    fig.update_xaxes(
                        type='category',
                        categoryorder='array',
                        categoryarray=sorted_dates
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning('Income statement data not available for one or both stocks')
            except:
                st.warning('Income statement data not available for one or both stocks')
        ########################

        # Net Income Comparison
        with col2:
            try:
                if not isinstance(income_statement_flipped1, str) and not isinstance(income_statement_flipped2, str):
                    net_income1 = income_statement_flipped1.loc['Net Income']
                    net_income2 = income_statement_flipped2.loc['Net Income']
                    df1 = pd.DataFrame({
                        'Date': list(net_income1.index),
                        'Net Income': net_income1.values.flatten(),
                        'Stock': [upper_ticker1] * len(net_income1)
                    })
                    df2 = pd.DataFrame({
                        'Date': list(net_income2.index),
                        'Net Income': net_income2.values.flatten(),
                        'Stock': [upper_ticker2] * len(net_income2)
                    })
                    combined_df = pd.concat([df1, df2])
                    combined_df['Net Income'] = combined_df['Net Income'] / 1e6
                    date_values = [x for x in combined_df['Date'].unique() if x != 'TTM']
                    formatted_dates = list(set([pd.to_datetime(date).strftime('%Y') for date in date_values]))
                    sorted_dates = sorted(formatted_dates) + ['TTM']
                    date_mapping = {date: pd.to_datetime(date).strftime('%Y') for date in date_values}
                    date_mapping['TTM'] = 'TTM'
                    combined_df['Date'] = combined_df['Date'].map(date_mapping)
                    combined_df['Date'] = pd.Categorical(combined_df['Date'], 
                                                        categories=sorted_dates,
                                                        ordered=True)
                    colors = {
                        upper_ticker1: color1,
                        upper_ticker2: color2
                    }
                    fig = go.Figure()
                    for stock in [upper_ticker1, upper_ticker2]:
                        stock_data = combined_df[combined_df['Stock'] == stock]
                        fig.add_trace(
                            go.Bar(
                                x=stock_data['Date'],
                                y=stock_data['Net Income'],
                                name=stock,
                                marker_color=colors[stock],
                                marker_line_width=0,
                                marker=dict(
                                    cornerradius=30
                                ),
                                hovertemplate=
                                "Ticker: %{customdata}<br>" +
                                "Year: %{x}<br>" +
                                "Value: $%{y:,.0f}M<br>" +
                                "<extra></extra>",
                                customdata=[stock] * len(stock_data)
                            )
                        )
                    fig.update_layout(
                        title={"text": f"Net Income", "font": {"size": title_font_size}},
                        title_y=1,
                        title_x=0,
                        margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                        barmode='group',
                        xaxis_title=None,
                        yaxis_title='USD in Million',
                        legend=dict(yanchor=legend_yanchor, y=legend_y, xanchor=legend_xanchor, x=legend_x, orientation = legend_orientation),
                        height=chart_height
                    )
                    fig.update_xaxes(
                        type='category',
                        categoryorder='array',
                        categoryarray=sorted_dates
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning('Income statement data not available for one or both stocks')
            except:
                st.warning('Income statement data not available for one or both stocks')
        ########################

        # EPS Comparison
        with col3:
            try:
                if not isinstance(income_statement_flipped1, str) and not isinstance(income_statement_flipped2, str):
                    eps1 = income_statement_flipped1.loc['Diluted EPS']
                    eps2 = income_statement_flipped2.loc['Diluted EPS']
                    df1 = pd.DataFrame({
                        'Date': list(eps1.index),
                        'EPS': eps1.values.flatten(),
                        'Stock': [upper_ticker1] * len(eps1)
                    })
                    df2 = pd.DataFrame({
                        'Date': list(eps2.index),
                        'EPS': eps2.values.flatten(),
                        'Stock': [upper_ticker2] * len(eps2)
                    })
                    combined_df = pd.concat([df1, df2])
                    #combined_df['EPS'] = combined_df['EPS'] / 1e6
                    date_values = [x for x in combined_df['Date'].unique() if x != 'TTM']
                    formatted_dates = list(set([pd.to_datetime(date).strftime('%Y') for date in date_values]))
                    sorted_dates = sorted(formatted_dates) + ['TTM']
                    date_mapping = {date: pd.to_datetime(date).strftime('%Y') for date in date_values}
                    date_mapping['TTM'] = 'TTM'
                    combined_df['Date'] = combined_df['Date'].map(date_mapping)
                    combined_df['Date'] = pd.Categorical(combined_df['Date'], 
                                                        categories=sorted_dates,
                                                        ordered=True)
                    colors = {
                        upper_ticker1: color1,
                        upper_ticker2: color2
                    }
                    fig = go.Figure()
                    for stock in [upper_ticker1, upper_ticker2]:
                        stock_data = combined_df[combined_df['Stock'] == stock]
                        fig.add_trace(
                            go.Scatter(
                                x=stock_data['Date'],
                                y=stock_data['EPS'],
                                name=stock,
                                mode='lines+markers',
                                marker=dict(color=colors[stock], size=10),
                                hovertemplate=
                                "Ticker: %{customdata}<br>" +
                                "Year: %{x}<br>" +
                                "Value: $%{y:,.0f}<br>" +
                                "<extra></extra>",
                                customdata=[stock] * len(stock_data)
                            )
                        )
                    fig.update_layout(
                        title={"text": f"EPS Trend", "font": {"size": title_font_size}},
                        title_y=1,
                        title_x=0,
                        margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                        xaxis_title=None,
                        yaxis_title='USD',
                        legend=dict(yanchor=legend_yanchor, y=legend_y, xanchor=legend_xanchor, x=legend_x, orientation =legend_orientation),
                        height=chart_height
                    )
                    fig.update_xaxes(
                        type='category',
                        categoryorder='array',
                        categoryarray=sorted_dates
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning('Income statement data not available for one or both stocks')
            except Exception as e:
                st.warning('Income statement data not available for one or both stocks')
        ########################

        st.divider()

        col1, col2, col3 = st.columns([3,3,3])
        # Margin Comparison
        with col1:
            try:
                data = {
                    'Category': ['Profit Margin', 'Gross Margin', 'Operating Margin'] * 2,
                    'Value': [profitmargin1, grossmargin1, operatingmargin1,
                            profitmargin2, grossmargin2, operatingmargin2],
                    'Ticker': [upper_ticker1] * 3 + [upper_ticker2] * 3
                }
                df = pd.DataFrame(data)
                df['Value'] = df['Value'].apply(lambda x: x * 100 if x != 'N/A' else 0)
                fig = go.Figure()
                for ticker, color in [(upper_ticker1, color1), (upper_ticker2, color2)]:
                    mask = df['Ticker'] == ticker
                    fig.add_trace(go.Bar(
                        name=ticker,
                        x=df[mask]['Category'],
                        y=df[mask]['Value'],
                        marker=dict(
                            color=color,
                            cornerradius=15,
                            line_width=0
                        ),
                        hovertemplate=
                        'Ticker: %{data.name}<br>' +
                        'Category: %{x}<br>' +
                        'Value: %{y:.1f}%<br>' +
                        '<extra></extra>'
                    ))
                fig.update_layout(
                    title={"text": f"Margins", "font": {"size": title_font_size}},
                    title_y=1,
                    title_x=0,
                    yaxis_title='Percentage (%)',
                    barmode='group',
                    height=chart_height,
                    margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                    legend=dict(
                        yanchor=legend_yanchor,
                        y=legend_y,
                        xanchor=legend_xanchor,
                        x=legend_x,
                        orientation=legend_orientation
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning('Unable to display Margin comparison chart. Some data might be unavailable.')
        ########################

        # DE Comparison
        with col2:
            try:
                if not isinstance(sa_metrics_df1, str) and not isinstance(sa_metrics_df2, str):
                    sa_metrics_df1.columns = [col.replace('FY ', '') if 'FY' in col else 'TTM' if col == 'Current' else col for col in sa_metrics_df1.columns]
                    sa_metrics_df2.columns = [col.replace('FY ', '') if 'FY' in col else 'TTM' if col == 'Current' else col for col in sa_metrics_df2.columns]
                    years1 = [col for col in sa_metrics_df1.columns if col != 'Fiscal Year']
                    years2 = [col for col in sa_metrics_df2.columns if col != 'Fiscal Year']
                    common_years = sorted(list(set(years1) & set(years2)))
                    if 'TTM' in years1 and 'TTM' in years2:
                        common_years = [y for y in common_years if y != 'TTM'] + ['TTM']
                    final_cols = ['Fiscal Year'] + common_years
                    sa_metrics_df1 = sa_metrics_df1[final_cols]
                    sa_metrics_df2 = sa_metrics_df2[final_cols]
                    de_ratio1 = sa_metrics_df1[sa_metrics_df1['Fiscal Year'] == 'Debt / Equity Ratio'].iloc[:, 1:].values.flatten()
                    de_ratio2 = sa_metrics_df2[sa_metrics_df2['Fiscal Year'] == 'Debt / Equity Ratio'].iloc[:, 1:].values.flatten()
                    df1 = pd.DataFrame({
                        'Date': common_years,
                        'Debt/Equity': [float(x) for x in de_ratio1],
                        'Stock': [upper_ticker1] * len(common_years)
                    })
                    df2 = pd.DataFrame({
                        'Date': common_years,
                        'Debt/Equity': [float(x) for x in de_ratio2],
                        'Stock': [upper_ticker2] * len(common_years)
                    })
                    combined_df = pd.concat([df1, df2])
                    fig = go.Figure()
                    for stock in [upper_ticker1, upper_ticker2]:
                        stock_data = combined_df[combined_df['Stock'] == stock]
                        fig.add_trace(
                            go.Scatter(
                                x=stock_data['Date'],
                                y=stock_data['Debt/Equity'],
                                name=stock,
                                mode='lines+markers',
                                line=dict(color=color1 if stock == upper_ticker1 else color2),
                                marker=dict(
                                    size=10,
                                    color=color1 if stock == upper_ticker1 else color2
                                ),
                                hovertemplate=
                                "Ticker: %{customdata}<br>" +
                                "Year: %{x}<br>" +
                                "Value: %{y:.2f}<br>" +
                                "<extra></extra>",
                                customdata=[stock] * len(stock_data)
                            )
                        )
                    fig.update_layout(
                        title={"text": "Debt/Equity Ratio", "font": {"size": title_font_size}},
                        title_y=1,
                        title_x=0,
                        margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                        xaxis_title=None,
                        xaxis=dict(
                            type='category',
                            categoryorder='array',  
                            categoryarray=common_years
                        ),
                        yaxis_title='Ratio',
                        legend=dict(yanchor=legend_yanchor, y=legend_y, xanchor=legend_xanchor, x=legend_x, orientation=legend_orientation),
                        height=chart_height
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning('Metrics data not available for one or both stocks')
            except Exception as e:
                st.write(e)
                st.warning('Error plotting Debt/Equity ratio comparison')
        ########################

        # Profitability Comparison
        with col3:
            try:
                data = {
                    'Category': ['Return on Assets', 'Return on Equity'] * 2,
                    'Value': [roa1, roe1,
                            roa2, roe2],
                    'Ticker': [upper_ticker1] * 2 + [upper_ticker2] * 2
                }
                df = pd.DataFrame(data)
                df['Value'] = df['Value'].apply(lambda x: x * 100 if x != 'N/A' else 0)
                fig = go.Figure()
                for ticker, color in [(upper_ticker1, color1), (upper_ticker2, color2)]:
                    mask = df['Ticker'] == ticker
                    fig.add_trace(go.Bar(
                        name=ticker,
                        x=df[mask]['Category'],
                        y=df[mask]['Value'],
                        marker=dict(
                            color=color,
                            cornerradius=20,
                            line_width=0
                        ),
                        hovertemplate=
                        'Ticker: %{data.name}<br>' +
                        'Category: %{x}<br>' +
                        'Value: %{y:.1f}%<br>' +
                        '<extra></extra>'
                    ))
                fig.update_layout(
                    title={"text": f"Profitability", "font": {"size": title_font_size}},
                    title_y=1,
                    title_x=0,
                    yaxis_title='Percentage (%)',
                    barmode='group',
                    height=chart_height,
                    margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                    legend=dict(
                        yanchor=legend_yanchor,
                        y=legend_y,
                        xanchor=legend_xanchor,
                        x=legend_x,
                        orientation=legend_orientation
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning('Unable to display profitability comparison chart. Some data might be unavailable.')
        ########################

        st.divider()

        col1, col2, col3 = st.columns([3,3,3])
        # Cash Comparison
        with col1:
            try:
                cash1 = balance_sheet_flipped1.loc['Current Assets', 'TTM'] / 1e6  
                debt1 = balance_sheet_flipped1.loc['Total Debt', 'TTM'] / 1e6  
                cash2 = balance_sheet_flipped2.loc['Current Assets', 'TTM'] / 1e6  
                debt2 = balance_sheet_flipped2.loc['Total Debt', 'TTM'] / 1e6  
                data = {
                    'Category': ['Current Assets', 'Total Debt', 'Current Assets', 'Total Debt'],
                    'Value': [cash1, debt1, cash2, debt2],
                    'Ticker': [upper_ticker1, upper_ticker1, upper_ticker2, upper_ticker2]
                }
                df = pd.DataFrame(data)
                fig = go.Figure()
                for ticker, color in [(upper_ticker1, color1), (upper_ticker2, color2)]:
                    mask = df['Ticker'] == ticker
                    fig.add_trace(go.Bar(
                        name=ticker,
                        x=df[mask]['Category'],
                        y=df[mask]['Value'],
                        marker=dict(
                            color=color,
                            cornerradius=20,
                            line_width=0
                        ),
                        hovertemplate=
                        'Ticker: %{data.name}<br>' +
                        'Category: %{x}<br>' +
                        'Value: $%{y:,.0f}M<br>' +
                        '<extra></extra>'
                    ))
                fig.update_layout(
                    title={"text": f"Assets & Debt", "font": {"size": title_font_size}},
                    title_y=1,
                    title_x=0,
                    yaxis_title='USD in Million',
                    barmode='group',
                    height=chart_height,
                    margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                    legend=dict(
                        yanchor=legend_yanchor,
                        y=legend_y,
                        xanchor=legend_xanchor,
                        x=legend_x,
                        orientation=legend_orientation
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning('Unable to display Cash and Debt comparison chart. Some data might be unavailable.')
        ########################

        # Retained Earnings Comparison
        with col2:
            try:
                if not isinstance(balance_sheet_flipped1, str) and not isinstance(balance_sheet_flipped2, str):
                    retained1 = balance_sheet_flipped1.loc['Retained Earnings']
                    retained2 = balance_sheet_flipped2.loc['Retained Earnings']
                    df1 = pd.DataFrame({
                        'Date': list(retained1.index),
                        'Retained Earnings': retained1.values.flatten(),
                        'Stock': [upper_ticker1] * len(retained1)
                    })
                    df2 = pd.DataFrame({
                        'Date': list(retained2.index),
                        'Retained Earnings': retained2.values.flatten(),
                        'Stock': [upper_ticker2] * len(retained2)
                    })
                    combined_df = pd.concat([df1, df2])
                    combined_df['Retained Earnings'] = combined_df['Retained Earnings'] / 1e6
                    date_values = [x for x in combined_df['Date'].unique() if x != 'TTM']
                    formatted_dates = list(set([pd.to_datetime(date).strftime('%Y') for date in date_values]))
                    sorted_dates = sorted(formatted_dates) + ['TTM']
                    date_mapping = {date: pd.to_datetime(date).strftime('%Y') for date in date_values}
                    date_mapping['TTM'] = 'TTM'
                    combined_df['Date'] = combined_df['Date'].map(date_mapping)
                    combined_df['Date'] = pd.Categorical(combined_df['Date'], 
                                                        categories=sorted_dates,
                                                        ordered=True)
                    colors = {
                        upper_ticker1: color1,
                        upper_ticker2: color2
                    }
                    fig = go.Figure()
                    for stock in [upper_ticker1, upper_ticker2]:
                        stock_data = combined_df[combined_df['Stock'] == stock]
                        fig.add_trace(
                            go.Bar(
                                x=stock_data['Date'],
                                y=stock_data['Retained Earnings'],
                                name=stock,
                                marker_color=colors[stock],
                                marker_line_width=0,
                                marker=dict(
                                    cornerradius=30
                                ),
                                hovertemplate=
                                "Ticker: %{customdata}<br>" +
                                "Year: %{x}<br>" +
                                "Value: $%{y:,.0f}M<br>" +
                                "<extra></extra>",
                                customdata=[stock] * len(stock_data)
                            )
                        )
                    fig.update_layout(
                        title={"text": f"Retained Earnings", "font": {"size": title_font_size}},
                        title_y=1,
                        title_x=0,
                        margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                        barmode='group',
                        xaxis_title=None,
                        yaxis_title='USD in Million',
                        legend=dict(yanchor=legend_yanchor, y=legend_y, xanchor=legend_xanchor, x=legend_x, orientation = legend_orientation),
                        height=chart_height
                    )
                    fig.update_xaxes(
                        type='category',
                        categoryorder='array',
                        categoryarray=sorted_dates
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning('Balance Sheet data not available for one or both stocks')
            except:
                st.warning('Balance Sheet data not available for one or both stocks')
        ########################

        # Free Cash Flow Comparison
        with col3:
            try:
                if not isinstance(cashflow_statement_flipped1, str) and not isinstance(cashflow_statement_flipped2, str):
                    fcf1 = cashflow_statement_flipped1.loc['Free Cash Flow']
                    fcf2 = cashflow_statement_flipped2.loc['Free Cash Flow']
                    df1 = pd.DataFrame({
                        'Date': list(fcf1.index),
                        'Free Cash Flow': fcf1.values.flatten(),
                        'Stock': [upper_ticker1] * len(fcf1)
                    })
                    df2 = pd.DataFrame({
                        'Date': list(fcf2.index),
                        'Free Cash Flow': fcf2.values.flatten(),
                        'Stock': [upper_ticker2] * len(fcf2)
                    })
                    combined_df = pd.concat([df1, df2])
                    combined_df['Free Cash Flow'] = combined_df['Free Cash Flow'] / 1e6
                    date_values = [x for x in combined_df['Date'].unique() if x != 'TTM']
                    formatted_dates = list(set([pd.to_datetime(date).strftime('%Y') for date in date_values]))
                    sorted_dates = sorted(formatted_dates) + ['TTM']
                    date_mapping = {date: pd.to_datetime(date).strftime('%Y') for date in date_values}
                    date_mapping['TTM'] = 'TTM'
                    combined_df['Date'] = combined_df['Date'].map(date_mapping)
                    combined_df['Date'] = pd.Categorical(combined_df['Date'], 
                                                        categories=sorted_dates,
                                                        ordered=True)
                    colors = {
                        upper_ticker1: color1,
                        upper_ticker2: color2
                    }
                    fig = go.Figure()
                    for stock in [upper_ticker1, upper_ticker2]:
                        stock_data = combined_df[combined_df['Stock'] == stock]
                        fig.add_trace(
                            go.Bar(
                                x=stock_data['Date'],
                                y=stock_data['Free Cash Flow'],
                                name=stock,
                                marker_color=colors[stock],
                                marker_line_width=0,
                                marker=dict(
                                    cornerradius=30
                                ),
                                hovertemplate=
                                "Ticker: %{customdata}<br>" +
                                "Year: %{x}<br>" +
                                "Value: $%{y:,.0f}M<br>" +
                                "<extra></extra>",
                                customdata=[stock] * len(stock_data)
                            )
                        )
                    fig.update_layout(
                        title={"text": f"Free Cash Flow", "font": {"size": title_font_size}},
                        title_y=1,
                        title_x=0,
                        margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right),
                        barmode='group',
                        xaxis_title=None,
                        yaxis_title='USD in Million',
                        legend=dict(yanchor=legend_yanchor, y=legend_y, xanchor=legend_xanchor, x=legend_x, orientation = legend_orientation),
                        height=chart_height
                    )
                    fig.update_xaxes(
                        type='category',
                        categoryorder='array',
                        categoryarray=sorted_dates
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning('Cash Flow Statement data not available for one or both stocks')
            except:
                st.warning('Cash Flow Statement data not available for one or both stocks')
        ########################

    except Exception as e:
        st.write(e)
        st.error(f"Failed to fetch data. Please check your ticker again.")
        st.warning("This tool supports only tickers from the U.S. stock market. Please note that ETFs and cryptocurrencies are not available for analysis. If the entered ticker is valid but the tool does not display results, it may be due to missing data or a technical issue. Kindly try again later. If the issue persists, please contact the developer for further assistance.")
        
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

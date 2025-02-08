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
def get_stock_data(ticker, apiKey=None):

    stock = yf.Ticker(ticker)
    lowercase_ticker = ticker.lower()
    upper_ticker = ticker.upper()
    price = stock.info.get('currentPrice', 'N/A')
    picture_url = f'https://logos.stockanalysis.com/{lowercase_ticker}.svg'
    exchange = stock.info.get('exchange', 'N/A')
    if exchange == 'NYQ':
        exchange_value = "NYSE"
    elif exchange == 'NMS':
        exchange_value = "NASDAQ"
    else:
        exchange_value = "N/A"
    lower_exchange = exchange_value.lower()

    ##### Morning Star #####
    fair_value = fvDate = moat = moatDate = starRating = assessment = 'N/A'
    performance_id = None
    if apiKey:
                    try:
                        conn = http.client.HTTPSConnection("morning-star.p.rapidapi.com")
                        headers = {
                            'x-rapidapi-key': apiKey,
                            'x-rapidapi-host': "morning-star.p.rapidapi.com"
                        }
                        conn.request("GET", "/market/v2/auto-complete?q=" + ticker, headers=headers)
                        res = conn.getresponse()
                        data = res.read()
                        json_data = json.loads(data.decode("utf-8"))
                        for item in json_data.get('results', []):
                            if item.get('ticker', '').upper() == ticker.upper():
                                performance_id = item.get('performanceId')
                                break
                    except Exception as e:
                        print(f"APIkey: Morningstar API request failed.")

    if performance_id:
                    try:
                        conn = http.client.HTTPSConnection("morning-star.p.rapidapi.com")
                        headers = {
                            'x-rapidapi-key': apiKey,
                            'x-rapidapi-host': "morning-star.p.rapidapi.com"
                        }
                        conn.request("GET", "/stock/v2/get-analysis-data?performanceId="+ performance_id, headers=headers)
                        res = conn.getresponse()
                        data = res.read()
                        json_data = json.loads(data.decode("utf-8"))
                        fair_value = json_data['valuation']['fairValue']
                        fvDate = json_data['valuation']['fairValueDate']
                        moat = json_data['valuation']['moat']
                        moatDate = json_data['valuation']['moatDate']
                        starRating = json_data['valuation']['startRating']
                        assessment = json_data['valuation']['assessment']
                    except Exception as e:
                        print("Performance ID: Morningstar API request failed.")

    #### Seeking Alpha ####
    authors_strongsell_count = authors_strongbuy_count = authors_sell_count = authors_hold_count = authors_buy_count = authors_rating = authors_count = epsRevisionsGrade = dpsRevisionsGrade = dividendYieldGrade = divSafetyCategoryGrade = divGrowthCategoryGrade = divConsistencyCategoryGrade = sellSideRating = ticker_id = quant_rating = growth_grade = momentum_grade = profitability_grade = value_grade = yield_on_cost_grade = 'N/A'
    sk_targetprice = 'N/A'
    if apiKey:
                    try:
                        conn = http.client.HTTPSConnection("seeking-alpha.p.rapidapi.com")
                        headers = {
                            'x-rapidapi-key': apiKey,
                            'x-rapidapi-host': "seeking-alpha.p.rapidapi.com"
                        }
                        conn.request("GET", "/symbols/get-ratings?symbol=" + ticker, headers=headers)
                        res = conn.getresponse()
                        data = res.read()
                        json_data = json.loads(data.decode("utf-8"))
                        first_data = json_data['data'][0]['attributes']['ratings']
                        ticker_id = json_data['data'][0]['attributes']['tickerId']
                        #
                        quant_rating = first_data['quantRating']
                        growth_grade = first_data['growthGrade']
                        momentum_grade = first_data['momentumGrade']
                        profitability_grade = first_data['profitabilityGrade']
                        value_grade = first_data['valueGrade']
                        yield_on_cost_grade = first_data['yieldOnCostGrade']
                        epsRevisionsGrade = first_data['epsRevisionsGrade']
                        dpsRevisionsGrade = first_data['dpsRevisionsGrade']
                        dividendYieldGrade = first_data['dividendYieldGrade']
                        divSafetyCategoryGrade = first_data['divSafetyCategoryGrade']
                        divGrowthCategoryGrade = first_data['divGrowthCategoryGrade']
                        divConsistencyCategoryGrade = first_data['divConsistencyCategoryGrade']
                        sellSideRating = first_data['sellSideRating']
                        #
                        authors_count = first_data['authorsCount']
                        authors_rating = first_data['authorsRating']
                        authors_buy_count = first_data['authorsRatingBuyCount']
                        authors_hold_count = first_data['authorsRatingHoldCount']
                        authors_sell_count = first_data['authorsRatingSellCount']
                        authors_strongbuy_count = first_data['authorsRatingStrongBuyCount']
                        authors_strongsell_count = first_data['authorsRatingStrongSellCount']
                    except Exception as e:
                        print("Analysts Data: Seeking Alpha API request failed.")

    if apiKey and ticker_id and ticker_id != 'N/A':
                    ticker_id_str = str(ticker_id)
                    try:
                        conn = http.client.HTTPSConnection("seeking-alpha.p.rapidapi.com")
                        headers = {
                            'x-rapidapi-key': apiKey,
                            'x-rapidapi-host': "seeking-alpha.p.rapidapi.com"
                        }
                        conn.request("GET", "/symbols/get-analyst-price-target?ticker_ids=" + ticker_id_str + "&return_window=1&group_by_month=false", headers=headers)
                        res = conn.getresponse()
                        data = res.read()
                        json_data = json.loads(data.decode("utf-8"))
                        get_sk_data = json_data['estimates'][f'{ticker_id}']['target_price']['0'][0]
                        sk_targetprice = get_sk_data['dataitemvalue']
                    except Exception as e:
                        print("Price Data: Seeking Alpha API request failed.")
    
    ##### SA forecasts #####
    try:
        url = f'https://stockanalysis.com/stocks/{ticker}/forecast/'
        r = requests.get(url)
        soup = BeautifulSoup(r.text,"lxml")
        table = soup.find("table",class_ = "w-full whitespace-nowrap border border-gray-200 text-right text-sm dark:border-dark-700 sm:text-base")
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
        sa_growth_df = pd.DataFrame(data, columns=headers)
        sa_growth_df = sa_growth_df.iloc[1:, :-1].reset_index(drop=True)
    except: sa_growth_df = ""
    
    ##### SA scores #####
    try:
                sa_score_url = f'https://stockanalysis.com/stocks/{ticker}/statistics/'
                sa_score_response = requests.get(sa_score_url)
                sa_score_soup = BeautifulSoup(sa_score_response.content, "html.parser")
                sa_score_table = sa_score_soup.find_all('table')[17]
                sa_score_data = {}
                #sa_altmanz = "N/A"
                #sa_piotroski = "N/A"
                if sa_score_table:
                    rows = sa_score_table.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) == 2:  
                            key = cols[0].text.strip()
                            value = cols[1].text.strip()
                            sa_score_data[key] = value
                    sa_altmanz = sa_score_data.get("Altman Z-Score", "N/A")
                    sa_piotroski = sa_score_data.get("Piotroski F-Score", "N/A")
    except Exception as e:
                print(f"SA scores fetching failed")

    ##### SA analysts rating #####
    try:
                sa_statistics_url = f'https://stockanalysis.com/stocks/{ticker}/statistics/'
                sa_statistics_response = requests.get(sa_statistics_url)
                sa_statistics_soup = BeautifulSoup(sa_statistics_response.content, "html.parser")
                sa_analyst_table = sa_statistics_soup.find_all('table')[15]
                sa_analysts_data = {}
                sa_analysts_consensus = "N/A"
                sa_analysts_targetprice = "N/A"
                sa_analysts_count = "N/A"
                if sa_analyst_table:
                    rows = sa_analyst_table.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) == 2:  
                            key = cols[0].text.strip()
                            value = cols[1].text.strip()
                            sa_analysts_data[key] = value

                    sa_analysts_consensus = sa_analysts_data.get("Analyst Consensus", "N/A")
                    sa_analysts_targetprice = sa_analysts_data.get("Price Target", "N/A")
                    sa_analysts_count = sa_analysts_data.get("Analyst Count", "N/A")
    except Exception as e:
                print("SA analysts data fetching failed.")
    
    ##### Market Beat forecast #####
    try:
                mb_url = f'https://www.marketbeat.com/stocks/{exchange_value}/{upper_ticker}/forecast/'
                mb_response = requests.get(mb_url)
                mb_soup = BeautifulSoup(mb_response.content, "html.parser")
                mb_table = mb_soup.find_all('table')[1]
                mb_data = {}
                mb_consensus_rating = "N/A"
                mb_predicted_upside = "N/A"
                mb_rating_score = "N/A"
                if mb_table:
                    rows = mb_table.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 2:  
                            key = cols[0].text.strip()
                            value = cols[1].text.strip()
                            mb_data[key] = value
                    mb_consensus_rating = mb_data.get("Consensus Rating", "N/A")
                    mb_predicted_upside = mb_data.get("Predicted Upside", "N/A")
                    mb_rating_score = mb_data.get("Consensus Rating Score", "N/A")
                    if mb_predicted_upside != "N/A":
                        match = re.search(r"([-+]?\d*\.?\d+)", mb_predicted_upside)
                    if match:
                        mb_predicted_upside = float(match.group(0))
                        mb_targetprice = 'N/A' if mb_predicted_upside == 'N/A' else (price * (mb_predicted_upside + 100)) / 100
                        mb_targetprice_value = 'N/A' if mb_targetprice == 'N/A' else f'${mb_targetprice:.2f}'
    except Exception as e:
                mb_targetprice_value = mb_predicted_upside = mb_consensus_rating = mb_rating_score = 'N/A'
    
    ##### Market Beat sector competitors #####
    try:
        mb_com_url = f'https://www.marketbeat.com/stocks/{exchange_value}/{upper_ticker}/competitors-and-alternatives/'
        mb_com_response = requests.get(mb_com_url)
        mb_com_soup = BeautifulSoup(mb_com_response.content, "html.parser")
        mb_com_table = mb_com_soup.find_all('table')[5]
        headers = [header.get_text(strip=True) for header in mb_com_table.find_all('th')]
        headers[2] = "Stock's Industry"
        rows = []
        for row in mb_com_table.find_all('tr')[1:]:
            row_data = [cell.get_text(strip=True) for cell in row.find_all('td')]
            rows.append(row_data)
        mb_com_df = pd.DataFrame(rows, columns=headers)
    except: mb_com_df = ""

    ##### Market Beat dividend comparison #####
    try:
        mb_com_url = f'https://www.marketbeat.com/stocks/{exchange_value}/{upper_ticker}/dividend/'
        mb_com_response = requests.get(mb_com_url)
        mb_com_soup = BeautifulSoup(mb_com_response.content, "html.parser")
        mb_com_table = mb_com_soup.find_all('table')[0]
        headers = [header.get_text(strip=True) for header in mb_com_table.find_all('th')]
        rows = []
        for row in mb_com_table.find_all('tr')[1:]:
            row_data = [cell.get_text(strip=True) for cell in row.find_all('td')]
            rows.append(row_data)
        mb_div_df = pd.DataFrame(rows, columns=headers)
        if mb_div_df.iloc[0, 0] == 'Annual Dividend':
            mb_div_df = mb_div_df
        else:
            mb_div_df = ""
    except: mb_div_df = ""

    ##### Market Beat competitors #####
    try:
        mb_com_url = f'https://www.marketbeat.com/stocks/{exchange_value}/{upper_ticker}/competitors-and-alternatives/'
        mb_com_response = requests.get(mb_com_url)
        mb_com_soup = BeautifulSoup(mb_com_response.content, "html.parser")
        mb_com_table = mb_com_soup.find_all('table')[6]
        mb_alt_headers = [mb_alt_header.get_text(strip=True) for mb_alt_header in mb_com_table.find_all('th')]
        rows = []
        for row in mb_com_table.find_all('tr')[1:]:
            row_data = [cell.get_text(strip=True) for cell in row.find_all('td')]
            rows.append(row_data)
        mb_alt_df = pd.DataFrame(rows, columns=mb_alt_headers)
    except: mb_alt_df = mb_alt_headers = ""
    
    ##### SA metric table #####
    try:
        url = f'https://stockanalysis.com/stocks/{ticker}/financials/ratios/'
        r = requests.get(url)
        soup = BeautifulSoup(r.text,"lxml")
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
        sa_metrics_df = pd.DataFrame(data, columns=headers)
        sa_metrics_df = sa_metrics_df.iloc[1:, :-1].reset_index(drop=True)
    except: sa_metrics_df = ""

    ##### SA metric table2 #####
    try:
        url2 = f'https://stockanalysis.com/stocks/{ticker}/financials/'
        r2 = requests.get(url2)
        soup2 = BeautifulSoup(r2.text,"lxml")
        table2 = soup2.find("table",class_ = "w-full border-separate border-spacing-0 text-sm sm:text-base [&_tbody]:sm:whitespace-nowrap [&_thead]:whitespace-nowrap")
        rows2 = table2.find_all("tr")
        headers2 = []
        data2 = []
        for row2 in rows2:
            cols2 = row2.find_all(["th", "td"])
            cols2_text = [col2.text.strip() for col2 in cols2]
            if not headers2:
                headers2 = cols2_text
            else:
                data2.append(cols2_text)
        sa_metrics_df2 = pd.DataFrame(data2, columns=headers2)
        sa_metrics_df2 = sa_metrics_df2.iloc[1:, :-1].reset_index(drop=True)
    except: sa_metrics_df2 = ""

    ##### Market Beat insider trades #####
    try:
        insider_mb_url = f'https://www.marketbeat.com/stocks/{exchange_value}/{upper_ticker}/insider-trades/'
        response = requests.get(insider_mb_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        if len(tables) >= 0:
            insider_mb = pd.read_html(str(tables[0]))[0]
        else:    
            insider_mb = ""
    except: insider_mb = ""
    
    name = stock.info.get('longName', 'N/A')
    sector = stock.info.get('sector', 'N/A')
    industry = stock.info.get('industry', 'N/A')
    employee = stock.info.get('fullTimeEmployees', 'N/A')
    marketCap = stock.info.get('marketCap', 'N/A')
    beta = stock.info.get('beta', 'N/A')
    longProfile = stock.info.get('longBusinessSummary', 'N/A')
    eps = stock.info.get('trailingEps', 'N/A')
    pegRatio = stock.info.get('pegRatio', stock.info.get('trailingPegRatio', 'N/A'))
    country = stock.info.get('country', 'N/A')
    yf_targetprice = stock.info.get('targetMeanPrice', 'N/A')
    yf_consensus = stock.info.get('recommendationKey', 'N/A')
    yf_analysts_count = stock.info.get('numberOfAnalystOpinions', 'N/A')
    website = stock.info.get('website', 'N/A')
    peRatio = stock.info.get('trailingPE', 'N/A')
    forwardPe = stock.info.get('forwardPE', 'N/A')
    dividendYield = stock.info.get('dividendYield', 'N/A')
    payoutRatio = stock.info.get('payoutRatio', 'N/A')
    sharesOutstanding = stock.info.get('sharesOutstanding', 'N/A')
    pbRatio = stock.info.get('priceToBook','N/A')
    deRatio = stock.info.get('debtToEquity','N/A')
    dividends = stock.info.get('dividendRate','N/A')
    exDividendDate = stock.info.get('exDividendDate','N/A')
    roe = stock.info.get('returnOnEquity','N/A')
    revenue_growth_current = stock.info.get('revenueGrowth','N/A')
    profitmargin = stock.info.get('profitMargins','N/A')
    grossmargin = stock.info.get('grossMargins','N/A')
    operatingmargin = stock.info.get('operatingMargins','N/A')
    ebitdamargin = stock.info.get('ebitdaMargins','N/A')
    fcf = stock.info.get('freeCashflow','N/A')
    revenue = stock.info.get('totalRevenue', 'N/A')
    roa = stock.info.get('returnOnAssets','N/A')
    current_ratio = stock.info.get('currentRatio','N/A')
    quick_ratio = stock.info.get('quickRatio','N/A')
    revenue_growth = stock.info.get('revenueGrowth', 'N/A')
    earnings_growth = stock.info.get('earningsGrowth', 'N/A')
    ev_to_ebitda = stock.info.get('enterpriseToEbitda', 'N/A')
    news = stock.news
    try:
        totalEsg = stock.sustainability.loc['totalEsg', 'esgScores']
        enviScore = stock.sustainability.loc['environmentScore', 'esgScores']
        socialScore = stock.sustainability.loc['socialScore', 'esgScores']
        governScore = stock.sustainability.loc['governanceScore', 'esgScores']
        percentile = stock.sustainability.loc['percentile', 'esgScores']
        insiderPct = stock.major_holders.loc['insidersPercentHeld', 'Value']
        institutionsPct = stock.major_holders.loc['institutionsPercentHeld', 'Value']
    except:
        totalEsg = enviScore = socialScore = governScore = percentile = insiderPct = institutionsPct = "N/A"
    try:
        hdata = stock.history(period='max')
        previous_close = hdata['Close'].iloc[-2]
    except: previous_close = 'N/A'
    try:
        get_earningsDate = stock.calendar['Earnings Date']
        if get_earningsDate:
            earningsDate = get_earningsDate[0].strftime('%Y-%m-%d')
        else:
            earningsDate = 'N/A'
    except: earningsDate = 'N/A'
    try: dividend_history = stock.dividends
    except: dividend_history = ""
    try: earnings_history = stock.earnings_history
    except: earnings_history = ""
    try: eps_trend = stock.eps_trend
    except: eps_trend = ""
    try:
        income_statement_tb = stock.income_stmt
        quarterly_income_statement_tb = stock.quarterly_income_stmt
    except: income_statement_tb = quarterly_income_statement_tb = ""
    try:
        balance_sheet_tb = stock.balance_sheet
        quarterly_balance_sheet_tb = stock.quarterly_balance_sheet
    except: balance_sheet_tb = quarterly_balance_sheet_tb = ""
    try:
        cashflow_statement_tb = stock.cashflow
        quarterly_cashflow_statement_tb = stock.quarterly_cashflow
    except: cashflow_statement_tb = quarterly_cashflow_statement_tb = ""
    try:
        sector_etf_mapping = {
                    "Consumer Cyclical": "XLY",
                    "Consumer Defensive": "XLP",
                    "Energy": "XLE",
                    "Financial Services": "XLF",
                    "Healthcare": "XLV",
                    "Industrials": "XLI",
                    "Basic Materials": "XLB",
                    "Real Estate": "XLR",
                    "Technology": "XLK",
                    "Utilities": "XLU",
                    "Communication Services": "XLC"
                    }
        matching_etf = sector_etf_mapping.get(sector)
        compare_tickers = (upper_ticker, '^GSPC', matching_etf)
        end = datetime.datetime.today()
        start = end - relativedelta(years=5)
        def relativereturn(df):
            rel = df.pct_change()
            cumret = (1+rel).cumprod()-1
            cumret = cumret.fillna(0)
            return cumret
        yf_com = relativereturn(yf.download(compare_tickers, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))['Close'])
    except: yf_com = matching_etf = ""
    try:
        end_date = datetime.datetime.today()
        start_date = (end_date - datetime.timedelta(days=int(2 * 365)))
        start_date_1y = (end_date - datetime.timedelta(days=int(1 * 365)))
        extended_data_r = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
        extended_data_r.columns = extended_data_r.columns.map('_'.join)
        extended_data_r.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        macd_data_r = yf.download(ticker, start=start_date_1y.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
        macd_data_r.columns = macd_data_r.columns.map('_'.join)
        macd_data_r.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        rsi_data_r = yf.download(ticker, start=start_date_1y.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
        rsi_data_r.columns = rsi_data_r.columns.map('_'.join)
        rsi_data_r.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        ta_data_r = yf.download(ticker, start=start_date_1y.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
        ta_data_r.columns = ta_data_r.columns.map('_'.join)
        ta_data_r.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    except: end_date = extended_data_r = macd_data_r = rsi_data_r = ta_data_r = ""

    try:
        eps_yield = eps/price
    except: eps_yield = "N/A"

    try:
        api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=api_key)
        def analyze_stock():
            prompt = f"""
            Analyze the stock {upper_ticker} for both long-term and short-term investment potential. Use the following financial data:
            - Historical price data: {extended_data_r}
            - Key financial metrics: 
                - Valuation: P/E Ratio = {peRatio}, P/B Ratio = {pbRatio}, EV/EBITDA = {ev_to_ebitda}
                - Profitability: Net profit margin = {profitmargin}, ROE = {roe}, ROA = {roa}, Gross margin = {grossmargin}
                - Growth: Revenue growth = {revenue_growth}, Earnings growth = {earnings_growth}
                - Financial health: Debt-to-equity = {deRatio}, Current ratio = {current_ratio}, Quick ratio = {quick_ratio}
                - Cash flow: Free cash flow = {fcf}, Operating cash flow margin = {operatingmargin}
                - Dividends: Dividend yield = {dividendYield}, Dividend payout ratio = {payoutRatio}
            - Income Statement data: {income_statement_tb}
            - Balance Sheet data: {balance_sheet_tb}
            - Cashflow Statement data: {cashflow_statement_tb}
                    
            Provide:
            1. A summary of whether the stock is good to invest in or not.
            2. Key fundamental analysis metrics (e.g., P/E ratio, revenue growth, debt-to-equity).
            3. Key technical analysis insights (e.g., moving averages, RSI, support/resistance levels).
            4. Sentiment analysis based on news and social media.
            5. Recommendations for when to buy (e.g., based on technical indicators or valuation).
            6. Separate conclusions for long-term and short-term investment strategies.
            """
                    
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an experienced financial analyst with expertise in both fundamental and technical analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.7
            )
                    
            return response.choices[0].message.content
        analysis = analyze_stock()
    except Exception as e:
        analysis = ""
    
    return analysis, ev_to_ebitda, earnings_growth, revenue_growth, quick_ratio, news, insider_mb, sa_growth_df, eps_yield, end_date, extended_data_r, macd_data_r, rsi_data_r, ta_data_r, matching_etf, yf_com, mb_alt_headers, sa_metrics_df2, sa_metrics_df, cashflow_statement_tb, quarterly_cashflow_statement_tb, balance_sheet_tb, quarterly_balance_sheet_tb, income_statement_tb, quarterly_income_statement_tb, mb_alt_df, mb_div_df, mb_com_df, mb_targetprice_value, mb_predicted_upside, mb_consensus_rating, mb_rating_score, sa_analysts_count, sa_analysts_consensus, sa_analysts_targetprice, sa_altmanz, sa_piotroski, sk_targetprice, authors_strongsell_count, authors_strongbuy_count, authors_sell_count, authors_hold_count, authors_buy_count, authors_rating, authors_count, epsRevisionsGrade, dpsRevisionsGrade, dividendYieldGrade, divSafetyCategoryGrade, divGrowthCategoryGrade, divConsistencyCategoryGrade, sellSideRating, ticker_id, quant_rating, growth_grade, momentum_grade, profitability_grade, value_grade, yield_on_cost_grade, performance_id, fair_value, fvDate, moat, moatDate, starRating, assessment, eps_trend, earnings_history, dividend_history, earningsDate, previous_close, current_ratio, fcf, revenue, exchange_value, upper_ticker, roa, ebitdamargin, operatingmargin, grossmargin, profitmargin, roe, revenue_growth_current, exDividendDate, pbRatio, deRatio, dividends, ticker, sharesOutstanding, institutionsPct, insiderPct, totalEsg, enviScore, socialScore, governScore, percentile, price, beta, name, sector, industry, employee, marketCap, longProfile, eps, pegRatio, picture_url, country, yf_targetprice, yf_consensus, yf_analysts_count, website, peRatio, forwardPe, dividendYield, payoutRatio, apiKey

''
''
#############################################        #############################################
############################################# Inputs #############################################
#############################################        #############################################

main_col1, main_col2 = st.columns([3,1])
with main_col1:
    st.title("US Stock Analysis Tool")
    input_col1, input_col2, input_col3 = st.columns([1, 3, 1])
    with input_col1:
        ticker = st.text_input("Enter US Stock Ticker:", "AAPL")
    with input_col2:
        apiKey = st.text_input("Enter your RapidAPI Key (optional):", "")

st.write("This analysis dashboard is designed to enable beginner investors to analyze stocks effectively and with ease. Please note that the information in this page is intended for educational purposes only and it does not constitute investment advice or a recommendation to buy or sell any security. We are not responsible for any losses resulting from trading decisions based on this information.")
st.info('Data is sourced from Yahoo Finance, Morningstar, Seeking Alpha, Market Beat, Stockanalysis.com and Alpha Spread. Certain sections require API keys to operate. Users are advised to subscribe to the Morningstar and Seeking Alpha APIs provided by Api Dojo through rapidapi.com.')

if st.button("Get Data"):
    try:
        analysis, ev_to_ebitda, earnings_growth, revenue_growth, quick_ratio, news, insider_mb, sa_growth_df, eps_yield, end_date, extended_data_r, macd_data_r, rsi_data_r, ta_data_r, matching_etf, yf_com, mb_alt_headers, sa_metrics_df2, sa_metrics_df, cashflow_statement_tb, quarterly_cashflow_statement_tb, balance_sheet_tb, quarterly_balance_sheet_tb, income_statement_tb, quarterly_income_statement_tb, mb_alt_df, mb_div_df, mb_com_df, mb_targetprice_value, mb_predicted_upside, mb_consensus_rating, mb_rating_score, sa_analysts_count, sa_analysts_consensus, sa_analysts_targetprice, sa_altmanz, sa_piotroski, sk_targetprice, authors_strongsell_count, authors_strongbuy_count, authors_sell_count, authors_hold_count, authors_buy_count, authors_rating, authors_count, epsRevisionsGrade, dpsRevisionsGrade, dividendYieldGrade, divSafetyCategoryGrade, divGrowthCategoryGrade, divConsistencyCategoryGrade, sellSideRating, ticker_id, quant_rating, growth_grade, momentum_grade, profitability_grade, value_grade, yield_on_cost_grade, performance_id, fair_value, fvDate, moat, moatDate, starRating, assessment, eps_trend, earnings_history, dividend_history, earningsDate, previous_close, current_ratio, fcf, revenue, exchange_value, upper_ticker, roa, ebitdamargin, operatingmargin, grossmargin, profitmargin, roe, revenue_growth_current, exDividendDate, pbRatio, deRatio, dividends, ticker, sharesOutstanding, institutionsPct, insiderPct, totalEsg, enviScore, socialScore, governScore, percentile, price, beta, name, sector, industry, employee, marketCap, longProfile, eps, pegRatio, picture_url, country, yf_targetprice, yf_consensus, yf_analysts_count, website, peRatio, forwardPe, dividendYield, payoutRatio, apiKey = get_stock_data(ticker, apiKey if apiKey.strip() else None)
     
#############################################         #############################################
############################################# Profile #############################################
#############################################         #############################################

        st.header(f'{name}', divider='gray')
        
        try: change_dollar = price - previous_close
        except: change_dollar = 'N/A'
        try: change_percent = (change_dollar / previous_close) * 100
        except: change_percent = 'N/A'
        try: yf_mos = ((yf_targetprice - price)/yf_targetprice) * 100
        except: yf_mos = 'N/A'
        ''
        ''
        employee_value = 'N/A' if employee == 'N/A' else f'{employee:,}'
        marketCap_value = 'N/A' if marketCap == 'N/A' else f'${marketCap/1000000:,.2f}'
        col1, col2, col3, col4 = st.columns([2, 1, 1, 3])
        with col1:
            #st.image(picture_url, width= 250)
            background_color = '#C5C6C7'
            st.markdown(f"""
                <div style="display: flex; justify-content: center; background-color: {background_color}; padding: 10px; border-radius: 10px;">
                    <img src="{picture_url}" width="250">
                </div>
                """,unsafe_allow_html=True)

        with col3:
            sharesOutstanding_value = 'N/A' if sharesOutstanding == 'N/A' else f'{sharesOutstanding/1000000000:,.2f}B'
            st.metric(label='Shares Outstanding', value=sharesOutstanding_value)

            insiderPct_value = 'N/A' if insiderPct == 'N/A' else f'{insiderPct*100:,.2f}%'
            st.metric(label='Owned by Insiders', value=insiderPct_value)

            institutionsPct_value = 'N/A' if institutionsPct == 'N/A' else f'{institutionsPct*100:,.2f}%'
            st.metric(label='Owned by Institutions', value=institutionsPct_value)

        st.markdown(f"<div style='text-align: justify;'>{longProfile}</div>", unsafe_allow_html=True)
        ''
        st.caption("Data source: Yahoo Finance")
        st.caption("Company logo source: Stockanalysis.com")

        with col4:
             st.markdown(f"""
             <div style='float: left; width: 100%;'>
                 <table style='width: 100%;'>
                     <tr><td><strong>Sector</strong></td><td>{sector}</td></tr>
                     <tr><td><strong>Industry</strong></td><td>{industry}</td></tr>
                     <tr><td><strong>Employees</strong></td><td>{employee_value}</td></tr>
                     <tr><td><strong>Market Cap</strong></td><td>{marketCap_value}M</td></tr>
                     <tr><td><strong>Country</strong></td><td>{country}</td></tr>
                     <tr><td><strong>Website</strong></td><td>{website}</td></tr>
                     <tr><td><strong>Earnings Date</strong></td><td>{earningsDate}</td></tr>
                 </table>
             </div>
             """, unsafe_allow_html=True)
        ''
        ''
#############################################      #############################################
############################################# Tabs #############################################
#############################################      #############################################

        overview_data, comparison_data, statements_data, guru_checklist, insider_trades, technicalAnalysis_data, news_data = st.tabs (["Overview","Comparisons","Financial Statements","Guru Checklist","Insider Trades","Technical Analysis","Top News"])

#############################################               #############################################
############################################# Overview Data #############################################
#############################################               #############################################

        with overview_data:

#Stock Performance
            st.subheader('Stock Performance', divider='gray')
            cols = st.columns(5)
            cols[0].metric(label='Current Price',value=f'${price:,.2f}',delta=f'{change_dollar:,.2f} ({change_percent:.2f}%)',delta_color='normal' )
            
            eps_value = 'N/A' if eps == 'N/A' else f'{eps:,.2f}'
            cols[1].metric(label='EPS (ttm)',value=eps_value)
            
            try:
                pegRatio_value = 'N/A' if pegRatio == 'N/A' else f'{pegRatio:,.2f}'
            except: 
                pegRatio_value = 'N/A'
            cols[2].metric(label='PEG Ratio',value=pegRatio_value)
            
            beta_value = 'N/A' if beta == 'N/A' else f'{beta:.2f}'
            cols[3].metric(label='Beta',value=beta_value)

            roe_value = 'N/A' if roe == 'N/A' else f'{roe*100:.2f}%'
            cols[4].metric(label='ROE',value=roe_value)

            cols1 = st.columns(5)
            pe_value = 'N/A' if peRatio == 'N/A' else f'{peRatio:.2f}'
            cols1[0].metric(label='PE Ratio',value=pe_value)
            
            forwardPe_value = 'N/A' if forwardPe == 'N/A' else f'{forwardPe:.2f}'
            cols1[1].metric(label='Forward PE',value=forwardPe_value)
            
            pbRatio_value = 'N/A' if pbRatio == 'N/A' else f'{pbRatio:.2f}'
            cols1[2].metric(label='PB Ratio',value=pbRatio_value)
            
            deRatio_value = 'N/A' if deRatio == 'N/A' else f'{deRatio/100:.2f}'
            cols1[3].metric(label='DE Ratio',value=deRatio_value)

            revenue_growth_current_value = 'N/A' if revenue_growth_current == 'N/A' else f'{revenue_growth_current*100:.2f}%'
            cols1[4].metric(label='Revenue Growth',value=revenue_growth_current_value)

            st.caption("Data source: Yahoo Finance")
            ''
 #Morning Star Research           
            
            if apiKey is None:
                #st.markdown("---")
                st.warning('Certain information will be hidden due to unavailability of API key. Please input your API key to access the full data.')
                #st.markdown("---")
            else:
                st.subheader('Morningstar Research', divider='gray')
                st.caption("This section only works with RapidAPI key.")
                
                starRating_value = 0 if starRating == 'N/A' else int(starRating)
                star_rating = ":star:" * int(round(starRating_value, 0))
                column1, column2, column3 = st.columns(3)
                with column1:
                    st.write("Economic Moat")
                    st.subheader(f'{moat}')
                fair_value_mos = 'N/A' if fair_value == 'N/A' else f'{((float(fair_value) - price)/float(fair_value)) * 100:.2f}%'
                fair_value_fix = 'N/A' if fair_value == 'N/A' else f'${float(fair_value):.2f}'
                with column2:
                    st.write("Fair Value")
                    st.subheader(f'{fair_value_fix}')  
                    if fair_value != 'N/A':  
                        mos_value = ((float(fair_value) - price)/float(fair_value)) * 100
                        if mos_value < 0:
                            st.markdown(f'<p style="color:red;">{fair_value_mos}</p>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<p style="color:green;">{fair_value_mos}</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p style="color:gray;">N/A</p>', unsafe_allow_html=True)  
                with column3:
                    st.write("Rating")
                    st.subheader(f'{star_rating}')
                ''
                #st.markdown(f'Current price of the stock is <span style="color:blue;">{assessment}</span>.', unsafe_allow_html=True)
                st.write(f'Morningstar Current Assessment: {assessment}')
                ''
                try:
                    formatted_moat_date = datetime.datetime.strptime(moatDate, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d")
                    formatted_fv_date = datetime.datetime.strptime(fvDate, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d")
                    st.caption(f"An economic moat refers to a company's ability to maintain competitive advantages to protect its long-term profits and market share from competitors.<br>Moat Assessment Date: {formatted_moat_date}", unsafe_allow_html=True)
                    st.caption(f"The Star Rating is determined by three factors: a stock's current price, Morningstar's estimate of the stock's fair value, and the uncertainty rating of the fair value. The bigger the discount, the higher the star rating. Four- and 5-star ratings mean the stock is undervalued, while a 3-star rating means it's fairly valued, and 1- and 2-star stocks are overvalued. When looking for investments, a 5-star stock is generally a better opportunity than a 1-star stock.<br>Fair Value Assessment Date: {formatted_fv_date}", unsafe_allow_html=True)
                    st.caption("Data source: Morning Star")
                except Exception as e:
                    st.write("")
                ''

#Quant Rating
                st.subheader('Seeking Alpha Quantitative Analysis', divider = 'gray')
                st.caption("This section only works with RapidAPI key.")

                cols = st.columns(3)
                quant_rating_value = 'N/A' if quant_rating == 'N/A' else f'{quant_rating:.2f}'
                cols[0].metric(label='Quant Rating',value=quant_rating_value)
                
                cols[1].metric(label='Growth Grade',value=growth_grade)
                
                cols[2].metric(label='Momentum Grade',value=momentum_grade)

                cols = st.columns(3)
                cols[0].metric(label='Profitability Grade',value=profitability_grade)
                
                cols[1].metric(label='Value Grade',value=value_grade)
                
                cols[2].metric(label='Yield on Cost Grade',value=yield_on_cost_grade)
                ''
                st.caption("Quant rating is a score from 1.0 to 5.0, where 1.0 is Strong Sell and 5.0 is Strong Buy.")
                st.caption("Grades refer to a system where a higher number indicates a better result.")
                st.caption("Data source: Seeking Alpha")

#Margins data
            st.subheader('Margins', divider='gray')
            if fcf == 'N/A' or revenue == 'N/A':
                fcf_margin = 'N/A'
            else: fcf_margin = (fcf/revenue)
            mgcol1, mgcol2, mgcol3, mgcol4 = st.columns([2,2,2,2])
            with mgcol1:
                try:
                    if grossmargin is None or grossmargin == 'N/A':
                        grossmargin_value = 'N/A'
                    else:
                        try:
                            grossmargin_value = float(grossmargin)
                        except ValueError:
                            grossmargin_value = 'N/A'
                    if grossmargin_value == 'N/A' or grossmargin_value <= 0:
                        rotate = 0
                        pie_values = [0, 1]
                        annotation_text = "No Data"
                    else:
                        pie_values = [grossmargin_value, 1 - grossmargin_value]
                        rotate = 0 if grossmargin_value > 0.5 else (grossmargin_value * 360) + 360
                        annotation_text = f"{grossmargin_value * 100:.1f}%"
                    fig = go.Figure(go.Pie(
                        values=pie_values,
                        labels=["Operating Margin", "Remaining"],
                        hole=0.7,
                        marker=dict(colors=["#4FC1E9", "#d3d3d3"]),
                        textinfo="none",
                        hoverinfo="none",
                        rotation=rotate
                    ))
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        title={"text": 'Gross Margin', "font": {"size": 22}},
                        title_y=0.95,
                        margin=dict(t=40, b=0, l=30, r=30),
                        annotations=[{
                            "text": annotation_text,
                            "showarrow": False,
                            "font_size": 22,
                            "x": 0.5,
                            "y": 0.5
                        }]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.write("Failed to get data.")
            with mgcol2:
                try:
                    if operatingmargin is None or operatingmargin == 'N/A':
                        operatingmargin_value = 'N/A'
                    else:
                        try:
                            operatingmargin_value = float(operatingmargin)
                        except ValueError:
                            operatingmargin_value = 'N/A'
                    if operatingmargin_value == 'N/A' or operatingmargin_value <= 0:
                        rotate = 0
                        pie_values = [0, 1]
                        annotation_text = "No Data"
                    else:
                        pie_values = [operatingmargin_value, 1 - operatingmargin_value]
                        rotate = 0 if operatingmargin_value > 0.5 else (operatingmargin_value * 360) + 360
                        annotation_text = f"{operatingmargin_value * 100:.1f}%"
                    fig = go.Figure(go.Pie(
                        values=pie_values,
                        labels=["Operating Margin", "Remaining"],
                        hole=0.7,
                        marker=dict(colors=["#48CFAD", "#d3d3d3"]),
                        textinfo="none",
                        hoverinfo="none",
                        rotation=rotate
                    ))
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        title={"text": 'Operating Margin', "font": {"size": 22}},
                        title_y=0.95,
                        margin=dict(t=40, b=0, l=30, r=30),
                        annotations=[{
                            "text": annotation_text,
                            "showarrow": False,
                            "font_size": 22,
                            "x": 0.5,
                            "y": 0.5
                        }]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.write("Failed to get data.")
            with mgcol3:
                try:
                    if profitmargin is None or profitmargin == 'N/A':
                        profitmargin_value = 'N/A'
                    else:
                        try:
                            profitmargin_value = float(profitmargin)
                        except ValueError:
                            profitmargin_value = 'N/A'
                    if profitmargin_value == 'N/A' or profitmargin_value <= 0:
                        rotate = 0
                        pie_values = [0, 1]
                        annotation_text = "No Data"
                    else:
                        pie_values = [profitmargin_value, 1 - profitmargin_value]
                        rotate = 0 if profitmargin_value > 0.5 else (profitmargin_value * 360) + 360
                        annotation_text = f"{profitmargin_value * 100:.1f}%"
                    fig = go.Figure(go.Pie(
                        values=pie_values,
                        labels=["Operating Margin", "Remaining"],
                        hole=0.7,
                        marker=dict(colors=["#FFCE54", "#d3d3d3"]),
                        textinfo="none",
                        hoverinfo="none",
                        rotation=rotate
                    ))
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        title={"text": 'Profit Margin', "font": {"size": 22}},
                        title_y=0.95,
                        margin=dict(t=40, b=0, l=30, r=30),
                        annotations=[{
                            "text": annotation_text,
                            "showarrow": False,
                            "font_size": 22,
                            "x": 0.5,
                            "y": 0.5
                        }]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.write("Failed to get data.")
            with mgcol4:
                try:
                    if fcf_margin is None or fcf_margin == 'N/A':
                        fcfmargin_value = 'N/A'
                    else:
                        try:
                            fcfmargin_value = float(fcf_margin)
                        except ValueError:
                            fcfmargin_value = 'N/A'
                    if fcfmargin_value == 'N/A' or fcfmargin_value <= 0:
                        rotate = 0
                        pie_values = [0, 1]
                        annotation_text = "No Data"
                    else:
                        pie_values = [fcfmargin_value, 1 - fcfmargin_value]
                        rotate = 0 if fcfmargin_value > 0.5 else (fcfmargin_value * 360) + 360
                        annotation_text = f"{fcfmargin_value * 100:.1f}%"
                    fig = go.Figure(go.Pie(
                        values=pie_values,
                        labels=["Operating Margin", "Remaining"],
                        hole=0.7,
                        marker=dict(colors=["#ED5565", "#d3d3d3"]),
                        textinfo="none",
                        hoverinfo="none",
                        rotation=rotate
                    ))
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        title={"text": 'FCF Margin', "font": {"size": 22}},
                        title_y=0.95,
                        margin=dict(t=40, b=0, l=30, r=30),
                        annotations=[{
                            "text": annotation_text,
                            "showarrow": False,
                            "font_size": 22,
                            "x": 0.5,
                            "y": 0.5
                        }]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.write("Failed to get data.")
            st.caption("Data source: Yahoo Finance")
            ''

#Dividend data
            st.subheader('Dividends & Yields', divider='gray')
            if dividendYield == 'N/A':
                st.write(f'{name} has no dividend data.')
            else:
                col1, col2 = st.columns([1, 3])
                with col1:
                    dividends_value = 'N/A' if dividends == 'N/A' else f'${dividends:,.2f}'
                    st.metric(label='Dividend per share',value=dividends_value)
                    
                    dividendYield_value = 'N/A' if dividendYield == 'N/A' else f'{dividendYield*100:.2f}%'
                    st.metric(label='Dividend Yield',value=dividendYield_value)
                    
                    payoutRatio_value = 'N/A' if payoutRatio == 'N/A' else f'{payoutRatio:.2f}'
                    st.metric(label='Payout Ratio',value=payoutRatio_value)

                    if exDividendDate == 'N/A':
                        exDividendDate_value = 'N/A'
                    else:
                        exDate = datetime.datetime.fromtimestamp(exDividendDate)
                        exDividendDate_value = exDate.strftime('%Y-%m-%d')
                    st.metric(label='Ex-Dividend Date',value=exDividendDate_value)

                    eps_yield_value = 'N/A' if eps_yield == 'N/A' else f'{eps_yield * 100:.2f}%'
                    st.metric(label='Earnings Yield',value=eps_yield_value)

                with col2:
                    try:
                        data_yearly = dividend_history.resample('YE').sum().reset_index()
                        data_yearly['Year'] = data_yearly['Date'].dt.year
                        data_yearly = data_yearly[['Year', 'Dividends']]
                        if dividends != 'N/A':
                            data_yearly.loc[data_yearly.index[-1], 'Dividends'] = dividends
                        tick_angle = 0 if len(data_yearly) < 20 else -90
                        fig = go.Figure()
                        fig.add_trace(
                            go.Bar(
                                x=data_yearly['Year'],
                                y=data_yearly['Dividends'],
                                name="Dividends",
                                marker=dict(color='#AC92EC', cornerradius=30),
                            )
                        )
                        fig.update_layout(
                            title={"text":'Dividends History', "font": {"size": 22}},
                            title_y=1,  
                            title_x=0.75, 
                            margin=dict(t=30, b=40, l=40, r=30),
                            xaxis_title=None,
                            yaxis_title="Dividends (USD)",
                            xaxis=dict(type='category',tickangle=tick_angle),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except: st.write ("Failed to get dividend history.")
            st.caption("Data source: Yahoo Finance")

#Earnings History
            st.subheader('Earnings History', divider='gray')
            ecol1, ecol2, ecol3 = st.columns([3, 1, 3])
            try:
                earnings_data = pd.DataFrame(earnings_history)
                earnings_data = earnings_data[~earnings_data.index.duplicated(keep='first')]
                with ecol1:
                    if 'epsEstimate' in earnings_data.columns and 'epsActual' in earnings_data.columns:
                        df = earnings_data.reset_index().melt(id_vars=['index'], value_vars=['epsEstimate', 'epsActual'], var_name='variable', value_name='value')
                        df['index'] = df['index'].dt.strftime('%Y-%m-%d')
                        actual_data = df[df['variable'] == 'epsActual']
                        estimate_data = df[df['variable'] == 'epsEstimate']
                        bar = go.Bar(
                            x=actual_data['index'],  
                            y=actual_data['value'],
                            name='Actual',
                            marker=dict(color='#FFCE54'),
                        )
                        estimate = go.Scatter(
                            x=estimate_data['index'],
                            y=estimate_data['value'],
                            mode='markers+lines',
                            name='Estimate',
                            marker=dict(color='red', size=10), 
                            line=dict(width=3) 
                        )
                        fig = go.Figure(data=[bar, estimate])
                        fig.update_layout(
                            title='Earnings Estimate vs Actual',
                            title_y=1,  
                            title_x=0, 
                            margin=dict(t=30, b=40, l=40, r=30),
                            xaxis_title=None,
                            yaxis_title='Earnings',
                            xaxis=dict(type='category',showgrid=True),
                            yaxis=dict(showgrid=True),
                            barmode='group',
                            height=400,
                            width=600,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write(f'{name} has no earning estimates data.')
                
                with ecol2:
                    last_eps_difference = earnings_data['epsDifference'].iloc[-1]
                    last_eps_actual = earnings_data['epsActual'].iloc[-1]
                    last_eps_estimate = earnings_data['epsEstimate'].iloc[-1]
                    st.metric(label='EPS Actual (last value)', value=f'${last_eps_actual}')
                    st.metric(label='EPS Estimate (last value)', value=f'${last_eps_estimate}')
                    if last_eps_difference < 0:
                        st.markdown(''':red[Last Earnings Missed!]''')
                        st.caption("The last actual EPS data missed the estimate EPS data.")
                    elif last_eps_difference == 0:
                        st.write("Last Earnings hit!")
                        st.caption("The last actual EPS data hit the estimate EPS data.")
                    elif last_eps_difference > 0:
                        st.markdown(''':green[Last Earnings Beat!]''')
                        st.caption("The last actual EPS data exceeded the estimate EPS data.")
                    else:
                        st.caption("No EPS data.")
                    st.caption(f"Next Earning - {earningsDate}")
            
                with ecol3:
                    try:
                        eps_data = eps_trend.loc[["0y", "+1y"], ["current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo"]]
                        eps_data = eps_data.T.reset_index()
                        eps_data.columns = ['TimePeriod', 'CurrentYear', 'NextYear']
                        label_map = {
                                'current': 'Current',
                                '7daysAgo': '7 Days Ago',
                                '30daysAgo': '30 Days Ago',
                                '60daysAgo': '60 Days Ago',
                                '90daysAgo': '90 Days Ago'
                        }
                        eps_data['TimePeriod'] = eps_data['TimePeriod'].map(label_map)
                        eps_melted = pd.melt(eps_data, id_vars=['TimePeriod'], value_vars=['CurrentYear', 'NextYear'],
                                                var_name='Year', value_name='EPS')
                        current_year_data = eps_melted[eps_melted['Year'] == 'CurrentYear']
                        next_year_data = eps_melted[eps_melted['Year'] == 'NextYear']
                        color_map = {'CurrentYear': '#9678DC', 'NextYear': '#D772AD'}
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                                x=current_year_data['TimePeriod'],
                                y=current_year_data['EPS'],
                                mode='lines+markers',
                                name='Current Year',
                                line=dict(color=color_map['CurrentYear']),
                                marker=dict(color=color_map['CurrentYear'])
                        ))
                        fig.add_trace(go.Scatter(
                                x=next_year_data['TimePeriod'],
                                y=next_year_data['EPS'],
                                mode='lines+markers',
                                name='Next Year',
                                line=dict(color=color_map['NextYear']),
                                marker=dict(color=color_map['NextYear'])
                        ))
                        fig.update_layout(
                                title='EPS Trend',
                                title_y=1,  
                                title_x=0, 
                                margin=dict(t=30, b=40, l=40, r=30),
                                xaxis=dict(
                                    title=None,
                                    categoryorder='array',
                                    showgrid=True,  
                                    categoryarray=['90 Days Ago', '60 Days Ago', '30 Days Ago', '7 Days Ago', 'Current'],
                                ),
                                yaxis=dict(
                                    title='EPS',
                                    showgrid=True
                                ),
                                height=400,
                                legend=dict(title_text=None),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        try:
                            epsmetrics = ['EPS (Diluted)']
                            sa_eps_df_filtered = sa_metrics_df2[sa_metrics_df2['Fiscal Year'].isin(epsmetrics)]
                            sa_eps_df_melted = sa_eps_df_filtered.melt(id_vars=['Fiscal Year'], 
                                                        var_name='Year', 
                                                        value_name='Value')
                            eps_unique_years = sa_eps_df_melted['Year'].unique()
                            eps_unique_years_sorted = sorted([year for year in eps_unique_years if year != 'TTM'])
                            if 'TTM' in eps_unique_years:
                                eps_unique_years_sorted.append('TTM')
                            figg = go.Figure()
                            for fiscal_year in sa_eps_df_melted['Fiscal Year'].unique():
                                filtered_data = sa_eps_df_melted[sa_eps_df_melted['Fiscal Year'] == fiscal_year]
                                figg.add_trace(go.Scatter(
                                    x=filtered_data['Year'],
                                    y=filtered_data['Value'],
                                    mode='lines+markers',
                                    name=str(fiscal_year)
                                ))
                            figg.update_layout(
                                title={"text":"EPS Trend", "font": {"size": 20}},
                                title_y=1,  
                                title_x=0, 
                                margin=dict(t=30, b=30, l=40, r=30),
                                xaxis_title=None,
                                yaxis_title='Value',
                                xaxis=dict(tickmode='array', tickvals=eps_unique_years_sorted, autorange='reversed',showgrid=True),
                                yaxis=dict(showgrid=True),
                                height=400
                            )
                            st.plotly_chart(figg, use_container_width=True)
                        except: st.write("Failed to get EPS trend.")
            except: st.write("Failed to get earnings data.")
            st.caption("Data source: Yahoo Finance")

#Estimate Data
            st.subheader('Growth Estimation', divider='gray')
            gcol1, gcol2= st.columns([3, 2])
            with gcol1:
                try:
                        growth_metrics = ['Revenue Growth', 'EPS Growth']
                        sa_growth_df_filtered = sa_growth_df[sa_growth_df['Fiscal Year'].isin(growth_metrics)]
                        sa_growth_metrics_df_melted = sa_growth_df_filtered.melt(id_vars=['Fiscal Year'], var_name='Year', value_name='Value')
                        growth_unique_years = sa_growth_metrics_df_melted['Year'].unique()
                        growth_unique_years_sorted = sorted([year for year in growth_unique_years if year != 'Current'])
                        if 'Current' in growth_unique_years:
                            growth_unique_years_sorted.append('Current')
                        fig_growth = go.Figure()
                        for fiscal_year in sa_growth_metrics_df_melted['Fiscal Year'].unique():
                            filtered_data = sa_growth_metrics_df_melted[sa_growth_metrics_df_melted['Fiscal Year'] == fiscal_year]
                            fig_growth.add_trace(go.Scatter(
                                x=filtered_data['Year'],
                                y=filtered_data['Value'],
                                mode='lines+markers',
                                name=str(fiscal_year)
                            ))
                        fig_growth.update_layout(
                            title={"text":"Growth Data", "font": {"size": 20}},
                            title_y=1,  
                            title_x=0, 
                            margin=dict(t=30, b=30, l=40, r=30),
                            xaxis_title=None,
                            yaxis_title='Value (%)',
                            xaxis=dict(tickmode='array', tickvals=growth_unique_years_sorted,showgrid=True),
                            yaxis=dict(showgrid=True),
                            legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.010),
                            height=400
                        )
                        st.plotly_chart(fig_growth, use_container_width=True)
                except Exception as e:
                        st.warning(f'{name} has no growth estimates data.')
            
            with gcol2:
                try:
                    sub_gcol1 = st.columns(2)
                    current_year = datetime.datetime.now().year
                    one_yr_header = f'FY {current_year + 1}'
                    two_yr_header = f'FY {current_year + 2}'
                    three_yr_header = f'FY {current_year + 3}'
                    try:
                        one_yr_revenue = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'Revenue Growth', one_yr_header].values[0]
                        one_yr_revenue_value = 'N/A' if one_yr_revenue == 'Upgrade' else one_yr_revenue
                        #one_yr_revenue = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'Revenue Growth', sa_growth_df.columns[6]].values[0]
                    except: one_yr_revenue_value = 'N/A'
                    sub_gcol1[0].metric(label='+1Y Revenue Growth',value=one_yr_revenue_value)

                    try:
                        one_yr_earnings = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'EPS Growth', one_yr_header].values[0]
                        one_yr_earnings_value = 'N/A' if one_yr_earnings == 'Upgrade' else one_yr_earnings
                        #one_yr_earnings = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'EPS Growth', sa_growth_df.columns[6]].values[0]
                    except: one_yr_earnings_value = 'N/A'
                    sub_gcol1[1].metric(label='+1Y EPS Growth',value=one_yr_earnings_value)

                    sub_gcol2 = st.columns(2)
                    try:
                        two_yr_revenue = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'Revenue Growth', two_yr_header].values[0]
                        two_yr_revenue_value = 'N/A' if two_yr_revenue == 'Upgrade' else two_yr_revenue
                        #two_yr_revenue = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'Revenue Growth', sa_growth_df.columns[7]].values[0]
                    except: two_yr_revenue_value = 'N/A'
                    sub_gcol2[0].metric(label='+2Y Revenue Growth',value=two_yr_revenue_value)

                    try:
                        two_yr_earnings = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'EPS Growth', two_yr_header].values[0]
                        two_yr_earnings_value = 'N/A' if two_yr_earnings == 'Upgrade' else two_yr_earnings
                        #two_yr_earnings = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'EPS Growth', sa_growth_df.columns[7]].values[0]
                    except: two_yr_earnings_value = 'N/A'
                    sub_gcol2[1].metric(label='+2Y EPS Growth',value=two_yr_earnings_value)

                    sub_gcol3 = st.columns(2)
                    try:
                        three_yr_revenue = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'Revenue Growth', three_yr_header].values[0]
                        three_yr_revenue_value = 'N/A' if three_yr_revenue == 'Upgrade' else three_yr_revenue 
                        #three_yr_revenue = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'Revenue Growth', sa_growth_df.columns[8]].values[0]
                    except: three_yr_revenue_value = 'N/A'
                    sub_gcol3[0].metric(label='+3Y Revenue Growth',value=three_yr_revenue_value)

                    try:
                        three_yr_earnings = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'EPS Growth', three_yr_header].values[0]
                        three_yr_earnings_value = 'N/A' if three_yr_earnings == 'Upgrade' else three_yr_earnings
                        #three_yr_earnings = sa_growth_df.loc[sa_growth_df.iloc[:, 0] == 'EPS Growth', sa_growth_df.columns[8]].values[0]
                    except: three_yr_earnings_value = 'N/A'
                    sub_gcol3[1].metric(label='+3Y EPS Growth',value=three_yr_earnings_value)

                    st.caption("The growth estimation data is sourced from Stockanalysis.com.")
                    st.caption("Please note that estimated data may not always be accurate and should not be solely relied upon for making investment decisions.")
                except Exception as e: 
                    st.write(f'{name} has no other estimates data. {e}')

# Scores
            st.subheader('Scores', divider='gray')
            try:
                score_col1, score_col2 = st.columns([2,3])
                with score_col1:
                    sa_piotroski_value = 'N/A' if sa_piotroski == 'N/A' else float(sa_piotroski)
                    if sa_piotroski_value != 'N/A':
                        fig = go.Figure(go.Indicator(
                            mode="gauge",
                            gauge={
                                'shape': "bullet",
                                'axis': {
                                    'range': [0, 9], 
                                    'visible': True,
                                    'tickcolor': 'white',
                                    'ticklen': 0,
                                    'showticklabels': True,
                                    'tickwidth': 0
                                },
                                'bar': {'color': "#5F9BEB", 'thickness':0.5},
                                'threshold': {
                                    'line': {'color': "blue", 'width': 0},
                                    'thickness': 0.5,
                                    'value': sa_piotroski_value
                                },
                                'steps': [
                                    {'range': [0, 1.8], 'color': "#ED2C0A"},
                                    {'range': [1.8, 3.6], 'color': "#F0702C"},
                                    {'range': [3.6, 5.4], 'color': "#FBB907"},
                                    {'range': [5.4, 7.2], 'color': "#B0B431"},
                                    {'range': [7.2, 9], 'color': "#88B03E"}
                                ],
                            },
                            value=sa_piotroski_value,
                            domain={'x': [0.1, 1], 'y': [0, 1]},
                            title={'text': f"{sa_piotroski_value:,.0f}/9"}
                        ))
            
                        fig.update_layout(
                            height=100,
                            margin=dict(l=30, r=30, t=30, b=30),
                            title={
                                'text': "Piotroski F-Score",
                                'y':1,
                                'x':0,
                                'xanchor':'left',
                                'yanchor':'top',
                                'font':{'size':20}
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Piotroski F-Score data is not available.")
                with score_col2:
                    st.caption("A company with a high Piotroski F-Score (say, 7 or above) is likely to be in good financial health, and may therefore be a good investment. Conversely, a company with a low score (say, 3 or below) may be in poor financial health, and may therefore be a risky investment.")
            except Exception as e:
                st.warning("Piotroski F-Score data is not available.")
            
            try:
                score_col3, score_col4 = st.columns([2,3])
                with score_col3:
                    sa_altmanz_value = 'N/A' if sa_altmanz == 'N/A' else float(sa_altmanz)
                    if sa_altmanz_value != 'N/A':
                        fig = go.Figure(go.Indicator(
                            mode="gauge",
                            gauge={
                                'shape': "bullet",
                                'axis': {
                                    'range': [0, 4], 
                                    'visible': True,
                                    'tickcolor': 'white',
                                    'ticklen': 0,
                                    'showticklabels': True,
                                    'tickwidth': 0
                                },
                                'bar': {'color': "#5F9BEB", 'thickness':0.5},
                                'threshold': {
                                    'line': {'color': "blue", 'width': 0},
                                    'thickness': 0.5,
                                    'value': sa_altmanz_value
                                },
                                'steps': [
                                    {'range': [0, 1.8], 'color': "#ED2C0A"},
                                    {'range': [1.8, 3.0], 'color': "#FBB907"},
                                    {'range': [3.0, 4.0], 'color': "#88B03E"}
                                ],
                            },
                            value=sa_altmanz_value,
                            domain={'x': [0.1, 1], 'y': [0, 1]},
                            title={'text': f"{sa_altmanz_value}"}
                        ))
            
                        fig.update_layout(
                            height=100,
                            margin=dict(l=30, r=30, t=30, b=30),
                            title={
                                'text': "Altman Z-Score",
                                'y':1,
                                'x':0,
                                'xanchor':'left',
                                'yanchor':'top',
                                'font':{'size':20}
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Altman Z-Score data is not available.")
                with score_col4:
                    st.caption("A score below 1.8 signals the company is likely headed for bankruptcy, while companies with scores above 3 are not likely to go bankrupt. Investors may consider purchasing a stock if its Altman Z-Score value is closer to 3 and selling, or shorting, a stock if the value is closer to 1.8.")
            except Exception as e:
                st.warning("Altman Z-Score data is not available.")
            st.caption("Data source: Stockanalysis.com")

#Analysts Ratings
            st.subheader('Analysts Ratings', divider='gray')
            try: sa_price_float = float(sa_analysts_targetprice.replace('$', ''))
            except: sa_price_float = 'N/A'
            try: sa_mos = ((sa_price_float - price)/sa_price_float) * 100
            except: sa_mos = 'N/A'
            try:
                counts = {
                'Buy': authors_buy_count,
                'Sell': authors_sell_count,
                'Hold': authors_hold_count,
                'Strong Buy': authors_strongbuy_count,
                'Strong Sell': authors_strongsell_count
                }
                largest_count_type = max(counts, key=counts.get)
                largest_value = round(counts[largest_count_type])
            except Exception as e:
                largest_count_type = 'N/A'
                largest_value = 'N/A'
            col1, col2, col3, col4 = st.columns([3, 3, 3, 3])
            try:
                yf_targetprice_value = 'N/A' if yf_targetprice == 'N/A' else f'${yf_targetprice:.2f}'
            except: yf_targetprice_value = 'N/A'
            yf_mos_value = 'N/A' if yf_mos == 'N/A' else f'{yf_mos:.2f}%'
            yf_consensus_value = 'N/A' if yf_consensus == 'none' else yf_consensus
            with col1:
                st.markdown(''':violet-background[Yahoo Finance]''')
                st.write(f'Price Target: {yf_targetprice_value}')
                st.write(f'Forecasted Difference: {yf_mos_value}')
                st.write(f'Analyst Consensus: {yf_consensus_value}')
                st.write(f'Analyst Count: {yf_analysts_count}')
                ''

            with col3:
                st.markdown(''':blue-background[MarketBeat]''')
                st.write(f'Price Target: {mb_targetprice_value}')
                st.write(f'Forecasted Difference: {mb_predicted_upside}%')
                st.write(f'Analyst Consensus: {mb_consensus_rating}')
                st.write(f'Rating Score: {mb_rating_score}')
                ''
            
            sk_targetprice_fix = 'N/A' if sk_targetprice == 'N/A' else f'${float(sk_targetprice):.2f}'
            sk_targetprice_mos ='N/A' if sk_targetprice =='N/A' else f'{((float(sk_targetprice) - price)/float(sk_targetprice)) * 100:.2f}%'
            with col4:
                st.markdown(''':orange-background[Seeking Alpha]''')
                st.write(f'Price Target: {sk_targetprice_fix}')
                st.write(f'Forecasted Difference: {sk_targetprice_mos}')
                st.write(f'Analyst Consensus: {largest_count_type}')
                st.write(f'Analyst Count: {largest_value}')
                ''

            sa_mos_value = 'N/A' if sa_mos == 'N/A' else f'{sa_mos:.2f}%'
            with col2:
                st.markdown(''':blue-background[Stockanalysis.com]''')
                st.write(f'Price Target: {sa_analysts_targetprice}')
                st.write(f'Forecasted Difference: {sa_mos_value}')
                st.write(f'Analyst Consensus: {sa_analysts_consensus}')
                st.write(f'Analyst Count: {sa_analysts_count}')
                ''
            ''
            st.subheader("Sustainability", divider = 'gray')
            #st.caption("This section shows the ESG risk ratings of '" + name + "' .")

#Risk gauge
#Gauge Plot
            def plot_gauge():
                totalEsg_value = 0.00 if totalEsg == 'N/A' else totalEsg
                max_value = 100
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=totalEsg_value,
                    title={'text': "Total ESG Risk"},
                    gauge={
                        'axis': {'range': [None, max_value]},
                        'bar': {'color': "#5F9BEB"},
                        'steps': [
                            {'range': [0, max_value * 0.25], 'color': "#CCD1D9"},
                            {'range': [max_value * 0.25, max_value * 0.5], 'color': "#FFCE54"},
                            {'range': [max_value * 0.5, max_value * 0.75], 'color': "#FB6E51"},
                            {'range': [max_value * 0.75, max_value], 'color': "#ED5565"},
                        ],
                    }
                ))
                gauge.update_layout(
                    autosize=False,
                    width=400,  
                    height=350, 
                    margin={'l': 50, 'r': 50, 't': 50, 'b': 50} 
                )
                st.plotly_chart(gauge,use_container_width=True)
            gauge_pcol1, gauge_pcol2, gauge_pcol3= st.columns ([3,1,3])
            with gauge_pcol1:
                plot_gauge()
#Risk Scores
            with gauge_pcol2:
                st.metric(label='Environmental Risk',value=enviScore)
                st.metric(label='Social Risk',value=socialScore)
                st.metric(label='Governance Risk',value=governScore)
            if enviScore == socialScore == governScore == 'N/A':
                    st.markdown(''':red[No ESG Data.]''')
            else:
                ''
#Descriptions
            with gauge_pcol3:
                st.caption("Total ESG Risk: Companies with ESG scores closer to 100 are considered to have significant ESG-related risks and challenges that could potentially harm their long-term sustainability.")
                st.caption("Environmental Risk: This reflects the company’s impact on the environment. e.g. carbon emissions, waste management, energy efficiency.")
                st.caption("Social Risk: This measures the company’s relationships with employees, suppliers, customers, and the community. e.g. human rights, labor practices, diversity, and community engagement.")
                st.caption("Governance Risk: this focuses on the company’s leadership, audit practices, internal controls, and shareholder rights. e.g. transparent financial reporting and strong board oversight.")
            st.caption("Data source: Yahoo Finance")

#############################################            #############################################
############################################# Comparison #############################################
#############################################            #############################################

        with comparison_data:
            compcol1,compcol2 = st.columns([3,1])
            with compcol1:
                try:
                    yf_com_df = yf_com
                    yf_com_df_melted = yf_com_df.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Relative Return')
                    yf_com_df_melted['Ticker'] = yf_com_df_melted['Ticker'].replace({'^GSPC': 'S&P500', matching_etf: 'Sector'})
                    unique_years_sorted = yf_com_df_melted['Date'].dt.year.unique()
                    custom_colors = {
                            upper_ticker: '#DA4453',  
                            'S&P500': '#5E9BEB',
                            'Sector': '#FFCE54'
                    }
                    def plot_relative_return_chart(yf_com_df_melted, custom_colors, upper_ticker):
                        df_plot = yf_com_df_melted.copy()
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
                                    line=dict(color=custom_colors.get(ticker, '#1f77b4'), shape='spline', smoothing=1.3),
                                    showlegend=True,
                                    hoverinfo="text",
                                    text=[f"{date}: {ret:.2f}%" for date, ret in zip(df_ticker['Date'], df_ticker['Relative Return'])]
                                )
                            )
                        fig.update_layout(
                            title={"text":f'{upper_ticker} - 5 Years Price Performance Comparison With Indices', "font": {"size": 22}},
                            title_y=1,  
                            title_x=0, 
                            margin=dict(t=30, b=40, l=40, r=30),
                            xaxis=dict(title=None, showticklabels=show_labels, showgrid=True), 
                            yaxis=dict(title="Cumulative Relative Return", showgrid=True),
                            legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.010),
                            height=500,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    plot_relative_return_chart(yf_com_df_melted, custom_colors, upper_ticker)
                except Exception as e:
                    st.warning(f'Error getting historical data. {e}')

            with compcol2:
                try:
                    st.subheader("5 Years Performance Summary")
                    last_values = yf_com_df_melted.groupby('Ticker').last()
                    st.metric(
                        label=upper_ticker,
                        value=f"{last_values.loc[upper_ticker, 'Relative Return']:.2f}%"
                    )
                    st.metric(
                        label="Sector",
                        value=f"{last_values.loc['Sector', 'Relative Return']:.2f}%"
                    )
                    st.metric(
                        label="S&P500",
                        value=f"{last_values.loc['S&P500', 'Relative Return']:.2f}%"
                    )
                    stock_return = last_values.loc[upper_ticker, 'Relative Return']
                    sector_return = last_values.loc['Sector', 'Relative Return']
                    sp500_return = last_values.loc['S&P500', 'Relative Return']
                    
                    performance_text = f"{upper_ticker} has "
                    if stock_return > sector_return and stock_return > sp500_return:
                        performance_text += f"outperformed both its sector ({sector_return:.2f}%) and S&P500 ({sp500_return:.2f}%) with a return of {stock_return:.2f}%"
                    elif stock_return < sector_return and stock_return < sp500_return:
                        performance_text += f"underperformed both its sector ({sector_return:.2f}%) and S&P500 ({sp500_return:.2f}%) with a return of {stock_return:.2f}%"
                    else:
                        performance_text += f"shown mixed performance with a return of {stock_return:.2f}% compared to its sector ({sector_return:.2f}%) and S&P500 ({sp500_return:.2f}%)"
                    
                    st.write("")
                    st.caption(performance_text)
                except Exception as e:
                    st.write("")
                ''
                
            st.caption("Data source: Yahoo Finance")
            ''
            st.subheader("Industry and Sector Comparison", divider = 'gray')
            try:
                col1,col2= st.columns([3,3])
                with col1:  
                    #try:
                    #    st.subheader(f'Valuation for Industry and Sector')
                    #    def highlight_company(s):
                    #        return ['background-color: yellow' if s.name == mb_com_df.columns[1] else '' for _ in s]
                    #    mb_com_styled_df = mb_com_df.style.apply(highlight_company, axis=0)
                    #    st.dataframe(mb_com_styled_df,hide_index=True,use_container_width=True)
                    #except Exception as e:
                    #    st.warning(f"Valuation Comparison: No data available.")
                    #st.caption("Data source: Market Beat")
    
                    
                        vscolors = ['#4FC1E9', '#48CFAD', '#EC87C0', '#FFCE54']
                        try:
                            numeric_df = mb_com_df.copy()
                            def convert_to_billions(value):
                                if 'T' in value:
                                    return float(value.replace('$', '').replace('T', '')) * 1000
                                elif 'B' in value:
                                    return float(value.replace('$', '').replace('B', ''))
                                elif 'M' in value:
                                    return float(value.replace('$', '').replace('M', '')) / 1000
                                return float(value.replace('$', ''))
                            income_data = mb_com_df[mb_com_df['Metric'] == 'Net Income']
                            income_values = [convert_to_billions(str(x)) for x in income_data.iloc[0, 1:]]
                            fig4 = go.Figure()
                            fig4.add_trace(go.Bar(
                                x=mb_com_df.columns[1:],
                                y=income_values,
                                text=[f"${x:.2f}B" for x in income_values],
                                textposition='auto',
                                marker=dict(cornerradius=5),
                                marker_color=vscolors
                            ))
                            fig4.update_layout(
                                title={"text":"Net Income Comparison (in Billions USD)","font": {"size": 22}},
                                xaxis_title=None,
                                yaxis_title='Net Income (Billion $)',
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig4, use_container_width=True)
                        except Exception as e:
                            st.warning("Net Income Comparison: No data available.")
                            
                with col2:            
                        try:
                            ratio_metrics = ['P/E Ratio', 'Price / Sales', 'Price / Cash', 'Price / Book']
                            ratio_data = numeric_df[numeric_df['Metric'].isin(ratio_metrics)]
                            fig2 = go.Figure()
                            for i,column in enumerate(mb_com_df.columns[1:]):
                                fig2.add_trace(go.Bar(
                                    name=column,
                                    x=ratio_metrics,
                                    y=ratio_data[column],
                                    text=ratio_data[column],  
                                    textposition='auto',
                                    textangle=-90,
                                    marker=dict(cornerradius=5),
                                    marker_color=vscolors[i]
                                ))
                            fig2.update_layout(
                                title={"text":"Ratios Comparison","font": {"size": 22}},
                                xaxis_title=None,
                                yaxis_title='Value',
                                barmode='group',
                                height=400,
                                showlegend=True,
                                legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.010)
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        except Exception as e:
                            st.warning("Ratio Comparison: No data available.")
                col3,col4 = st.columns([3,3])
                with col3:
                        try:
                            performance_metrics = ['7 Day Performance', '1 Month Performance', '1 Year Performance']
                            performance_data = numeric_df[numeric_df['Metric'].isin(performance_metrics)]
                            fig3 = go.Figure()
                            for i,column in enumerate(mb_com_df.columns[1:]):
                                fig3.add_trace(go.Bar(
                                    name=column,
                                    x=performance_metrics,
                                    y=performance_data[column],
                                    text=performance_data[column],
                                    textposition='auto',
                                    textangle=-90,
                                    marker=dict(cornerradius=5),
                                    marker_color=vscolors[i]
                                ))
                            fig3.update_layout(
                                title={"text":"Performance Comparison","font": {"size": 22}},
                                xaxis_title=None,
                                yaxis_title='Performance (%)',
                                barmode='group',
                                height=400,
                                showlegend=True,
                                legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.010)
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        except Exception as e:
                            st.warning("Performance Comparison: No data available.")
                with col4:
                        try:
                            def convert_value(x):
                                if not isinstance(x, str):
                                    return x
                                x = x.replace('$', '').replace('%', '').replace(',', '')
                                if 'T' in x:
                                    return float(x.replace('T', '')) * 1000
                                elif 'B' in x:
                                    return float(x.replace('B', ''))
                                elif 'M' in x:
                                    return float(x.replace('M', '')) / 1000
                                return float(x)
                            for col in numeric_df.columns:
                                if col != "Metric":
                                    numeric_df[col] = numeric_df[col].apply(convert_value)
                            dividend_data = numeric_df[numeric_df['Metric'] == 'Dividend Yield']
                            fig1 = go.Figure()
                            fig1.add_trace(go.Bar(
                                x=mb_com_df.columns[1:],
                                y=dividend_data.iloc[0, 1:],
                                text=[f"{x:.2f}%" for x in dividend_data.iloc[0, 1:]],
                                textposition='auto',
                                marker=dict(cornerradius=5),
                                marker_color=vscolors
                            ))
                            fig1.update_layout(
                                title={"text":"Dividend Yield Comparison","font": {"size": 22}},
                                xaxis_title=None,
                                yaxis_title='Dividend Yield (%)',
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        except Exception as e:
                            st.warning("Dividend Yield Comparison: No data available.")
                            
            except Exception as e:
                st.warning(f"Valuation Comparison: No data available.")
            st.caption("Data source: Market Beat")
            ''            
                
            st.subheader("Dividend Comparison", divider = 'gray')
            try:
                col5, col6 = st.columns([3,3])
                with col5:
                    #try:
                    #    st.subheader('Dividends Comparison')
                    #    def highlight_company(s):
                    #        return ['background-color: yellow' if s.name == mb_div_df.columns[1] else '' for _ in s]
                    #    mb_div_styled_df = mb_div_df.style.apply(highlight_company, axis=0)
                    #    st.dataframe(mb_div_styled_df,hide_index=True,use_container_width=True)
                    #except Exception as e:
                    #    st.warning(f"Dividends Comparison: No data available.")
                    #st.caption("Data source: Market Beat")
    
                        try:
                            numeric_df = mb_div_df.copy()
                            for col in numeric_df.columns:
                                if col != "Type":
                                    mask = numeric_df['Type'] != 'Track Record'
                                    numeric_df.loc[mask, col] = numeric_df.loc[mask, col].replace('[\$,\%]', '', regex=True).astype(float)
                            vscolors2 = ['#4FC1E9', '#48CFAD', '#EC87C0', '#FFCE54']
                            annual_data = numeric_df[numeric_df['Type'] == 'Annual Dividend']
                            fig1 = go.Figure()
                            fig1.add_trace(go.Bar(
                                x=mb_div_df.columns[1:],
                                y=annual_data.iloc[0, 1:],
                                text=[f"${x:.2f}" for x in annual_data.iloc[0, 1:]],
                                textposition='auto',
                                marker=dict(cornerradius=5),
                                marker_color=vscolors2[:3]
                            ))
                            fig1.update_layout(
                                title={"text":"Annual Dividend Comparison","font": {"size": 22}},
                                xaxis_title=None,
                                yaxis_title='Annual Dividend ($)',
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        except Exception as e:
                            st.warning("Annual Dividend Comparison: No data available.")
                with col6:
                        try:
                            yield_data = numeric_df[numeric_df['Type'] == 'Dividend Yield']
                            fig2 = go.Figure()
                            fig2.add_trace(go.Bar(
                                x=mb_div_df.columns[1:],
                                y=yield_data.iloc[0, 1:],
                                text=[f"{x:.2f}%" for x in yield_data.iloc[0, 1:]],
                                textposition='auto',
                                marker=dict(cornerradius=5),
                                marker_color=vscolors2[:3]
                            ))
                            fig2.update_layout(
                                title={"text":"Dividend Yield Comparison","font": {"size": 22}},
                                xaxis_title=None,
                                yaxis_title='Dividend Yield (%)',
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        except Exception as e:
                            st.warning("Dividend Yield Comparison: No data available.")
                col7, col8 = st.columns ([3,3])
                with col7:
                        try:
                            growth_data = numeric_df[numeric_df['Type'] == 'Annualized 3-Year Dividend Growth']
                            fig3 = go.Figure()
                            fig3.add_trace(go.Bar(
                                x=mb_div_df.columns[1:],
                                y=growth_data.iloc[0, 1:],
                                text=[f"{x:.2f}%" for x in growth_data.iloc[0, 1:]],
                                textposition='auto',
                                marker=dict(cornerradius=5),
                                marker_color=vscolors2[:3]
                            ))
                            fig3.update_layout(
                                title={"text":"Annualized 3-Year Dividend Growth Comparison","font": {"size": 22}},
                                xaxis_title=None,
                                yaxis_title='Growth Rate (%)',
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        except Exception as e:
                            st.warning("Dividend Growth Comparison: No data available.")
            except Exception as e:
                st.warning(f"Dividends Comparison: No data available.")
            st.caption("Data source: Market Beat")
            ''

            try:
                SPECIAL_TICKERS = {'NVDA', 'ARM', 'NXPI', 'LHX', 'AVGO', 'QCOM', 'TXN', 'INTC', 'STM',
                                    'THO', 'EME', 'KBR', 'ACM', 'PCAR', 'HPQ', 'SAP'
                                    }
                def clean_ticker_name(text):
                    for ticker in SPECIAL_TICKERS:
                        if text.startswith(ticker):
                            remaining = text[len(ticker):].strip()
                            return f"{ticker} {remaining}"
                    match = re.match(r"([A-Z]+)([A-Z][a-z].*)", text)
                    if match:
                        ticker, name = match.groups()
                        return f"{ticker} {name.strip()}"
                    return text
                mb_alt_df[mb_alt_headers[0]] = mb_alt_df[mb_alt_headers[0]].apply(clean_ticker_name)
                def get_star_rating(rating_text):
                    try:
                        rating = round(float(rating_text.split(' ')[0]))
                        return '★' * rating + '☆' * (5 - rating)
                    except ValueError:
                        return rating_text
                mb_alt_df[mb_alt_headers[1]] = mb_alt_df[mb_alt_headers[1]].apply(get_star_rating)
                def add_space_after_dollar(text):
                    string = re.sub(r'(\$\d+\.\d{2})(\d+\.\d+%)', r'\1 \2', text)
                    string = re.sub(r'(\$\d+\.\d{2})([+-])', r'\1 \2', string)
                    return string
                mb_alt_df[mb_alt_headers[2]] = mb_alt_df[mb_alt_headers[2]].apply(add_space_after_dollar)
                mb_alt_df[mb_alt_headers[3]] = mb_alt_df[mb_alt_headers[3]].apply(add_space_after_dollar)
                mb_alt_df = mb_alt_df.iloc[:, :-1]
                st.subheader(f'{name} Competitors List')
                st.dataframe(mb_alt_df,hide_index=True,use_container_width=True)
                st.caption("Data source: Market Beat")
                ''
                compcol3,compcol4 = st.columns([3,1])
                with compcol3:
                    try:
                        ticker_2 = mb_alt_df.iloc[1, 0].split()[0]
                        ticker2 = '' if len(ticker_2) > 4 else ticker_2
                        ticker_3 = mb_alt_df.iloc[2, 0].split()[0]
                        ticker3 = '' if len(ticker_3) > 4 else ticker_3
                        ticker_4 = mb_alt_df.iloc[3, 0].split()[0]
                        ticker4 = '' if len(ticker_4) > 4 else ticker_4
                        scompare_tickers = [upper_ticker for upper_ticker in (upper_ticker, ticker2, ticker3, ticker4) if upper_ticker]
                        if scompare_tickers:
                            send = datetime.datetime.today()
                            sstart = send - relativedelta(years=5)
                            def relativereturn(mb_alt_df):
                                rel = mb_alt_df.pct_change()
                                cumret = (1+rel).cumprod()-1
                                cumret = cumret.fillna(0)
                                return cumret
                            mb_alt_df = relativereturn(yf.download(scompare_tickers, sstart.strftime('%Y-%m-%d'), send.strftime('%Y-%m-%d'))['Close'])
                            mb_alt_df_melted = mb_alt_df.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Relative Return')
                            #unique_years_sorted = df_melted['Date'].dt.year.unique()
                            custom_colors = {
                                                upper_ticker: '#DA4453',  
                                                ticker2: '#4FC1E9',
                                                ticker3: '#A0D468',
                                                ticker4: '#FFCE54'
                            }
                            custom_colors = {k: v for k, v in custom_colors.items() if k in scompare_tickers}
                            def plot_relative_return_comparison(mb_alt_df_melted, custom_colors, upper_ticker):
                                df_plot = mb_alt_df_melted.copy()
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
                                            line=dict(color=custom_colors.get(ticker, '#1f77b4'), shape='spline', smoothing=1.3),
                                            showlegend=True,
                                            hoverinfo="text",
                                            text=[f"{date}: {ret:.2f}%" for date, ret in zip(df_ticker['Date'], df_ticker['Relative Return'])]
                                        )
                                    )
                                fig.update_layout(
                                    title={"text":f'{upper_ticker} - 5 Years Price Performance Comparison With Competitors', "font": {"size": 22}},
                                    title_y=1,  
                                    title_x=0, 
                                    margin=dict(t=30, b=40, l=40, r=30),
                                    xaxis=dict(
                                        title=None,
                                        showticklabels=show_labels,
                                        showgrid=True
                                    ),
                                    yaxis=dict(
                                        title="Cumulative Relative Return",
                                        showgrid=True
                                    ),
                                    legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.010),
                                    height=500,
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            plot_relative_return_comparison(mb_alt_df_melted, custom_colors, upper_ticker)
                    except Exception as e:
                        print(f"Failed to scrape ticker data from table.")
                with compcol4:
                    try:
                        st.subheader("5 Years Performance Summary")
                        last_values = mb_alt_df_melted.groupby('Ticker').last()
                        for ticker in scompare_tickers:
                            if ticker in last_values.index:
                                st.metric(
                                    label=ticker,
                                    value=f"{last_values.loc[ticker, 'Relative Return']:.2f}%"
                                )
                        st.write("")  # Add some spacing
                        best_performer = last_values['Relative Return'].idxmax()
                        worst_performer = last_values['Relative Return'].idxmin()
                        best_return = last_values.loc[best_performer, 'Relative Return']
                        worst_return = last_values.loc[worst_performer, 'Relative Return']
                        
                        summary = f"Among the competitors, {best_performer} showed the strongest performance with {best_return:.2f}% return, while {worst_performer} had the lowest return at {worst_return:.2f}%."
                        st.caption(summary)       
                    except Exception as e:
                        st.write("")
            except Exception as e:
                print(f"Failed to scrape ticker data from table.")
            st.caption("Data source: Yahoo Finance")

#############################################            #############################################
############################################# Statements #############################################
#############################################            #############################################
        with statements_data:
#Income Statement
            st.subheader("Income Statement (P&L)", divider ='gray')
            st.info("Notes: An income statement or profit and loss account shows the company's revenues and expenses during a particular period. It indicates how the revenues (also known as the “top line”) are transformed into the net income or net profit (the result after all revenues and expenses have been accounted for). The purpose of the income statement is to show managers and investors whether the company made money (profit) or lost money (loss) during the period being reported. It provides insight into a company’s operations, efficiency, management, and performance relative to others in the same sector.")
            try:
                income_statement = income_statement_tb  
                quarterly_income_statement = quarterly_income_statement_tb
                ttm = quarterly_income_statement.iloc[:, :4].sum(axis=1)
                income_statement.insert(0, 'TTM', ttm)
                income_statement_flipped = income_statement.iloc[::-1]
                formatted_columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in income_statement_flipped.columns]
                income_statement_flipped.columns = formatted_columns
                st.dataframe(income_statement_flipped,use_container_width=True)
                #chart_setup
                income_items = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EBITDA']
                income_bar_data = income_statement_flipped.loc[income_items].transpose()
                income_bar_data_million = income_bar_data / 1e6
                income_bar_data_million = income_bar_data_million.reset_index().rename(columns={'index': 'Date'})
                income_bar_data_melted = income_bar_data_million.melt('Date', var_name='Key Values', value_name='USD in Million')
                income_bar_data_melted['Key Values'] = pd.Categorical(income_bar_data_melted['Key Values'], categories=income_items, ordered=True)

                colors = {
                    'Total Revenue': '#ED5565',
                    'Gross Profit': '#EC87C0',
                    'Operating Income': '#FFCE54',
                    'Net Income': '#AC92EC',
                    'EBITDA': '#4FC1E9'
                }
                fig = go.Figure()
                for item in income_items:
                    fig.add_trace(
                        go.Bar(
                            x=income_bar_data_melted[income_bar_data_melted['Key Values'] == item]['Date'],
                            y=income_bar_data_melted[income_bar_data_melted['Key Values'] == item]['USD in Million'],
                            name=item,
                            marker_color=colors[item]
                        )
                    )
                fig.update_layout(
                    title={"text":"Income Statement Key Values Chart", "font": {"size": 20}},
                    title_y=1,  
                    title_x=0, 
                    margin=dict(t=30, b=40, l=40, r=30),
                    barmode='group', 
                    xaxis_title=None,
                    yaxis_title='USD in Million',
                    legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.010),
                    height=400
                )
                fig.update_xaxes(
                    type='category',
                    categoryorder='category ascending'
                )       
                st.plotly_chart(fig, use_container_width=True)
                #barChartColumn
                # income_statement.fillna(0, inplace=True)
                # date_columns = sorted([col for col in income_statement_flipped.columns if col not in ['TTM', 'Trend']])
                # ordered_columns = date_columns + ['TTM', 'Trend']
                # income_statement_flipped['Trend'] = income_statement_flipped.apply(lambda row: list(row)[::-1], axis=1)
                # income_statement_flipped = income_statement_flipped[ordered_columns]
                # st.dataframe(
                #     income_statement_flipped,
                #     column_config={
                #         "Trend": st.column_config.BarChartColumn(
                #             label="Trend",
                #             help="Trend of financial values across years",
                #             width="small"
                #         )
                #     }, use_container_width=True
                # )
            except: st.warning("Failed to get Income Statement.")
            st.caption("Data source: Yahoo Finance")
            ''

#Balance Sheet
            st.subheader("Balance Sheet (Financial Position)", divider ='gray')
            st.info("Notes: A balance sheet is a financial statement that reports a company's assets, liabilities, and shareholder equity. It provides a snapshot of a company's finances (what it owns and owes) as of the date of publication. The balance sheet adheres to an equation that equates assets with the sum of liabilities and shareholder equity.")
            try:
                balance_sheet = balance_sheet_tb
                quarterly_balance_sheet = quarterly_balance_sheet_tb
                ttm = quarterly_balance_sheet.iloc[:, :4].sum(axis=1)
                balance_sheet.insert(0, 'TTM', ttm)
                balance_sheet_flipped = balance_sheet.iloc[::-1]
                formatted_columns2 = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in balance_sheet_flipped.columns]
                balance_sheet_flipped.columns = formatted_columns2
                st.dataframe(balance_sheet_flipped,use_container_width=True)
                #chart_setup
                balance_items = ['Cash And Cash Equivalents','Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']
                balance_bar_data = balance_sheet_flipped.loc[balance_items].transpose()
                balance_bar_data_million = balance_bar_data / 1e6
                balance_bar_data_million = balance_bar_data_million.reset_index().rename(columns={'index': 'Date'})
                balance_bar_data_melted = balance_bar_data_million.melt('Date', var_name='Key Values', value_name='USD in Million')
                balance_bar_data_melted['Key Values'] = pd.Categorical(balance_bar_data_melted['Key Values'], categories=balance_items, ordered=True)
                colors = {
                    'Cash And Cash Equivalents': '#FFCE54',
                    'Total Assets': '#5F9BEB',
                    'Total Liabilities Net Minority Interest': '#FB6E51',
                    'Stockholders Equity': '#48CFAD',
                }
                fig = go.Figure()
                for item in balance_items:
                    fig.add_trace(
                        go.Bar(
                            x=balance_bar_data_melted[balance_bar_data_melted['Key Values'] == item]['Date'],
                            y=balance_bar_data_melted[balance_bar_data_melted['Key Values'] == item]['USD in Million'],
                            name=item,
                            marker_color=colors[item]
                        )
                    )
                fig.update_layout(
                    title={"text":"Balance Sheet Key Values Chart", "font": {"size": 20}},
                    title_y=1,  
                    title_x=0, 
                    margin=dict(t=30, b=40, l=40, r=30),
                    barmode='group', 
                    xaxis_title=None,
                    yaxis_title='USD in Million',
                    legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.010),
                    height=400
                )
                fig.update_xaxes(
                    type='category',
                    categoryorder='category ascending'
                )       
                st.plotly_chart(fig, use_container_width=True)
                #
                # balance_sheet.fillna(0, inplace=True)
                # date_columns = sorted([col for col in balance_sheet_flipped.columns if col not in ['TTM', 'Trend']])
                # ordered_columns = date_columns + ['TTM', 'Trend']
                # balance_sheet_flipped['Trend'] = balance_sheet_flipped.apply(lambda row: list(row)[::-1], axis=1)
                # balance_sheet_flipped = balance_sheet_flipped[ordered_columns]
                # st.dataframe(
                #     balance_sheet_flipped,
                #     column_config={
                #         "Trend": st.column_config.BarChartColumn(
                #             label="Trend",
                #             help="Trend of financial values across years",
                #             width="small"
                #         )
                #     }, use_container_width=True
                # )
            except: st.warning("Failed to get Balance Sheet.")
            st.caption("Data source: Yahoo Finance")
            ''
        
#Cashflow Statement
            st.subheader("Cashflow Statement (CFS)", divider ='gray')
            st.info("Notes: A cash flow statement summarizes the amount of cash and cash equivalents entering and leaving a company. The CFS highlights a company's cash management, including how well it generates cash. This financial statement complements the balance sheet and the income statement. The main components of the CFS are cash from three areas: Operating activities, investing activities, and financing activities.")
            cashflow_statement = cashflow_statement_tb
            quarterly_cashflow_statement = quarterly_cashflow_statement_tb
            try:
                ttm = quarterly_cashflow_statement.iloc[:, :4].sum(axis=1)
                cashflow_statement.insert(0, 'TTM', ttm)
                cashflow_statement_flipped = cashflow_statement.iloc[::-1]
                formatted_columns3 = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in cashflow_statement_flipped.columns]
                cashflow_statement_flipped.columns = formatted_columns3
                st.dataframe(cashflow_statement_flipped,use_container_width=True)
                #chart_setup
                cashflow_items = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow']
                cashflow_bar_data = cashflow_statement_flipped.loc[cashflow_items].transpose()
                cashflow_bar_data_million = cashflow_bar_data / 1e6
                cashflow_bar_data_million = cashflow_bar_data_million.reset_index().rename(columns={'index': 'Date'})
                cashflow_bar_data_melted = cashflow_bar_data_million.melt('Date', var_name='Key Values', value_name='USD in Million')
                cashflow_bar_data_melted['Key Values'] = pd.Categorical(cashflow_bar_data_melted['Key Values'], categories=cashflow_items, ordered=True)
                colors = {
                    'Operating Cash Flow': '#8CC152',
                    'Investing Cash Flow': '#ED5565',
                    'Financing Cash Flow': '#FB6E51',
                    'Free Cash Flow': '#48CFAD',
                }
                fig = go.Figure()
                for item in cashflow_items:
                    fig.add_trace(
                        go.Bar(
                            x=cashflow_bar_data_melted[cashflow_bar_data_melted['Key Values'] == item]['Date'],
                            y=cashflow_bar_data_melted[cashflow_bar_data_melted['Key Values'] == item]['USD in Million'],
                            name=item,
                            marker_color=colors[item]
                        )
                    )
                fig.update_layout(
                    title={"text":"Cashflow Statement Key Values Chart", "font": {"size": 20}},
                    title_y=1,  
                    title_x=0, 
                    margin=dict(t=30, b=40, l=40, r=30),
                    barmode='group', 
                    xaxis_title=None,
                    yaxis_title='USD in Million',
                    legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.010),
                    height=400
                )
                fig.update_xaxes(
                    type='category',
                    categoryorder='category ascending'
                )
                st.info("Operating activities should be positive (+ve). Investing activities should be negative (-ve). Financing activities should be negative (-ve).")       
                st.plotly_chart(fig, use_container_width=True)
                #
                # cashflow_statement.fillna(0, inplace=True)
                # date_columns = sorted([col for col in cashflow_statement_flipped.columns if col not in ['TTM', 'Trend']])
                # ordered_columns = date_columns + ['TTM', 'Trend']
                # cashflow_statement_flipped['Trend'] = cashflow_statement_flipped.apply(lambda row: list(row)[::-1], axis=1)
                # cashflow_statement_flipped = cashflow_statement_flipped[ordered_columns]
                # st.dataframe(
                #     cashflow_statement_flipped,
                #     column_config={
                #         "Trend": st.column_config.BarChartColumn(
                #             label="Trend",
                #             help="Trend of financial values across years",
                #             width="small",
                #         )
                #     }, use_container_width=True
                # )
            except Exception as e: st.warning(f'Failed to get Cash Flow Statement.')
            st.caption("Data source: Yahoo Finance")
            ''

#Financial Ratios
            st.subheader("Statistical Data Visualization", divider ='gray')
            rcol1, rcol2 =st.columns([3,3])
            with rcol1:
                try:
                    hmetrics = ['Debt / Equity Ratio', 'Debt / EBITDA Ratio', 'Debt / FCF Ratio', 'Current Ratio', 'Quick Ratio']
                    sa_metrics_df_filtered = sa_metrics_df[sa_metrics_df['Fiscal Year'].isin(hmetrics)]
                    sa_metrics_df_melted = sa_metrics_df_filtered.melt(id_vars=['Fiscal Year'], 
                                                var_name='Year', 
                                                value_name='Value')
                    unique_years = sa_metrics_df_melted['Year'].unique()
                    unique_years_sorted = sorted([year for year in unique_years if year != 'Current'])
                    if 'Current' in unique_years:
                        unique_years_sorted.append('Current')
                    figf = go.Figure()
                    for fiscal_year in sa_metrics_df_melted['Fiscal Year'].unique():
                        filtered_data = sa_metrics_df_melted[sa_metrics_df_melted['Fiscal Year'] == fiscal_year]
                        figf.add_trace(go.Scatter(
                            x=filtered_data['Year'],
                            y=filtered_data['Value'],
                            mode='lines+markers',
                            name=str(fiscal_year)
                        ))
                    figf.update_layout(
                        title={"text":"Financial Health Data", "font": {"size": 20}},
                        title_y=1,  
                        title_x=0, 
                        margin=dict(t=30, b=30, l=40, r=30),
                        xaxis_title=None,
                        yaxis_title='Value',
                        xaxis=dict(tickmode='array', tickvals=unique_years_sorted, autorange='reversed',showgrid=True),
                        yaxis=dict(showgrid=True),
                        height=400
                    )
                    st.plotly_chart(figf, use_container_width=True)
                except Exception as e:
                    st.warning(f"Financial Health: No data available.")
            with rcol2:
                try:
                    vmetrics = ['PE Ratio', 'PS Ratio', 'PB Ratio', 'P/FCF Ratio']
                    sa_metrics_df_filtered = sa_metrics_df[sa_metrics_df['Fiscal Year'].isin(vmetrics)]
                    sa_metrics_df_melted = sa_metrics_df_filtered.melt(id_vars=['Fiscal Year'], 
                                                var_name='Year', 
                                                value_name='Value')
                    unique_years = sa_metrics_df_melted['Year'].unique()
                    unique_years_sorted = sorted([year for year in unique_years if year != 'Current'])
                    if 'Current' in unique_years:
                        unique_years_sorted.append('Current')
                    figv = go.Figure()
                    for fiscal_year in sa_metrics_df_melted['Fiscal Year'].unique():
                        filtered_data = sa_metrics_df_melted[sa_metrics_df_melted['Fiscal Year'] == fiscal_year]
                        figv.add_trace(go.Scatter(
                            x=filtered_data['Year'],
                            y=filtered_data['Value'],
                            mode='lines+markers',
                            name=str(fiscal_year)
                        ))
                    figv.update_layout(
                        title={"text":"Valuation Data", "font": {"size": 20}},
                        title_y=1,  
                        title_x=0, 
                        margin=dict(t=30, b=30, l=40, r=30),
                        xaxis_title=None,
                        yaxis_title='Value',
                        xaxis=dict(tickmode='array', tickvals=unique_years_sorted, autorange='reversed',showgrid=True),
                        yaxis=dict(showgrid=True),
                        height=400
                    )
                    st.plotly_chart(figv, use_container_width=True)
                except Exception as e:
                    st.warning(f"Valuation: No data available.")
            pcol1, pcol2 = st.columns([3,3])
            with pcol1:
                try:
                    pmetrics = ['Return on Equity (ROE)', 'Return on Assets (ROA)', 'Return on Capital (ROIC)']
                    sa_metrics_df_filtered = sa_metrics_df[sa_metrics_df['Fiscal Year'].isin(pmetrics)]
                    sa_metrics_df_melted = sa_metrics_df_filtered.melt(id_vars=['Fiscal Year'], 
                                                var_name='Year', 
                                                value_name='Value (%)')
                    sa_metrics_df_melted['Value (%)'] = sa_metrics_df_melted['Value (%)'].replace({'%': ''}, regex=True).astype(float)
                    unique_years = sa_metrics_df_melted['Year'].unique()
                    unique_years_sorted = sorted([year for year in unique_years if year != 'Current'])
                    if 'Current' in unique_years:
                        unique_years_sorted.append('Current')
                    figp = go.Figure()
                    for fiscal_year in sa_metrics_df_melted['Fiscal Year'].unique():
                        filtered_data = sa_metrics_df_melted[sa_metrics_df_melted['Fiscal Year'] == fiscal_year]
                        figp.add_trace(go.Scatter(
                            x=filtered_data['Year'],
                            y=filtered_data['Value (%)'],
                            mode='lines+markers',
                            name=str(fiscal_year)
                        ))
                    figp.update_layout(
                        title={"text":"Profitability Data", "font": {"size": 20}},
                        title_y=1,  
                        title_x=0, 
                        margin=dict(t=30, b=30, l=40, r=30),
                        xaxis_title=None,
                        yaxis_title='Value (%)',
                        xaxis=dict(tickmode='array', tickvals=unique_years_sorted, autorange='reversed',showgrid=True),
                        yaxis=dict(showgrid=True),
                        height=400
                    )
                    st.plotly_chart(figp, use_container_width=True)
                except Exception as e:
                    st.warning(f"Profitability: No data available.")
            with pcol2:
                try:
                    pmetrics = ['Earnings Yield', 'FCF Yield', 'Dividend Yield']
                    sa_metrics_df_filtered = sa_metrics_df[sa_metrics_df['Fiscal Year'].isin(pmetrics)]
                    sa_metrics_df_melted = sa_metrics_df_filtered.melt(id_vars=['Fiscal Year'], var_name='Year', value_name='Value (%)')
                    sa_metrics_df_melted['Value (%)'] = sa_metrics_df_melted['Value (%)'].replace({'%': ''}, regex=True).astype(float)
                    unique_years = sa_metrics_df_melted['Year'].unique()
                    unique_years_sorted = sorted([year for year in unique_years if year != 'Current'])
                    if 'Current' in unique_years:
                        unique_years_sorted.append('Current')
                    figy = go.Figure()
                    for fiscal_year in sa_metrics_df_melted['Fiscal Year'].unique():
                        filtered_data = sa_metrics_df_melted[sa_metrics_df_melted['Fiscal Year'] == fiscal_year]
                        figy.add_trace(go.Scatter(
                            x=filtered_data['Year'],
                            y=filtered_data['Value (%)'],
                            mode='lines+markers',
                            name=str(fiscal_year)
                        ))
                    figy.update_layout(
                        title={"text":"Yield Data", "font": {"size": 20}},
                        title_y=1,  
                        title_x=0, 
                        margin=dict(t=30, b=30, l=40, r=30),
                        xaxis_title=None,
                        yaxis_title='Value (%)',
                        xaxis=dict(tickmode='array', tickvals=unique_years_sorted, autorange='reversed',showgrid=True),
                        yaxis=dict(showgrid=True),
                        height=400
                    )
                    st.plotly_chart(figy, use_container_width=True)
                except Exception as e:
                    st.warning(f"Yield: No data available.")

            mcol1, mcol2 = st.columns([3,3])
            with mcol1:
                try:
                    mmetrics = ['Gross Margin', 'Operating Margin', 'Profit Margin', 'EBITDA Margin']
                    sa_metrics_df_filtered = sa_metrics_df2[sa_metrics_df2['Fiscal Year'].isin(mmetrics)]
                    sa_metrics_df_melted = sa_metrics_df_filtered.melt(id_vars=['Fiscal Year'], 
                                                var_name='Year', 
                                                value_name='Value (%)')
                    sa_metrics_df_melted['Value (%)'] = sa_metrics_df_melted['Value (%)'].replace({'%': ''}, regex=True).astype(float)
                    unique_years = sa_metrics_df_melted['Year'].unique()
                    unique_years_sorted = sorted([year for year in unique_years if year != 'TTM'])
                    if 'TTM' in unique_years:
                        unique_years_sorted.append('TTM')
                    fiscal_years = sa_metrics_df_melted['Fiscal Year'].unique()
                    figm = go.Figure()
                    for fiscal_year in fiscal_years:
                        filtered_data = sa_metrics_df_melted[sa_metrics_df_melted['Fiscal Year'] == fiscal_year]
                        figm.add_trace(go.Bar(
                            x=filtered_data['Year'],
                            y=filtered_data['Value (%)'],
                            name=fiscal_year,
                            hoverinfo='y+name',
                            #marker=dict(line=dict(width=1))
                        ))
                    figm.update_layout(
                        title={"text":"Margin Data", "font": {"size": 20}},
                        title_y=1,  
                        title_x=0, 
                        margin=dict(t=30, b=30, l=40, r=30),
                        xaxis_title=None,
                        yaxis_title='Value (%)',
                        barmode='group',  
                        xaxis=dict(autorange='reversed'),
                        height=400  
                    )
                    st.plotly_chart(figm, use_container_width=True)
                except Exception as e:
                    st.warning(f"Margins: No data available.")
            with mcol2:
                try:
                    mmetrics = ['Net Income Growth', 'EPS Growth', 'Dividend Growth']
                    sa_metrics_df_filtered = sa_metrics_df2[sa_metrics_df2['Fiscal Year'].isin(mmetrics)]
                    sa_metrics_df_melted = sa_metrics_df_filtered.melt(id_vars=['Fiscal Year'], 
                                                var_name='Year', 
                                                value_name='Value (%)')
                    sa_metrics_df_melted['Value (%)'] = sa_metrics_df_melted['Value (%)'].replace({'%': ''}, regex=True).astype(float)
                    unique_years = sa_metrics_df_melted['Year'].unique()
                    unique_years_sorted = sorted([year for year in unique_years if year != 'TTM'])
                    if 'TTM' in unique_years:
                        unique_years_sorted.append('TTM')
                    figg = go.Figure()
                    for fiscal_year in sa_metrics_df_melted['Fiscal Year'].unique():
                        filtered_data = sa_metrics_df_melted[sa_metrics_df_melted['Fiscal Year'] == fiscal_year]
                        figg.add_trace(go.Scatter(
                            x=filtered_data['Year'],
                            y=filtered_data['Value (%)'],
                            mode='lines+markers',
                            name=str(fiscal_year)
                        ))
                    figg.update_layout(
                        title={"text":"Growth Data", "font": {"size": 20}},
                        title_y=1,  
                        title_x=0, 
                        margin=dict(t=30, b=30, l=40, r=30),
                        xaxis_title=None,
                        yaxis_title='Value (%)',
                        xaxis=dict(tickmode='array', tickvals=unique_years_sorted, autorange='reversed',showgrid=True),
                        yaxis=dict(showgrid=True),
                        height=400
                    )
                    st.plotly_chart(figg, use_container_width=True)
                except Exception as e:
                    st.warning(f"Growth: No data available.")
            st.caption("Data source: Stockanalysis.com")
            ''

            def highlight_result(val):
                if val == 'GOOD':
                    color = 'green'
                elif val == 'BAD':
                    color = 'red'
                else:
                    color ='#AAB2BD'
                return f'background-color: {color}; color: white'

            st.subheader("Financial Statements Checklist", divider ='gray')
            st.caption("These checklists are derived from the concepts outlined by Brian Feroldi for analyzing financial statements.")
            st.info("Many companies may have valid reasons to violate these rules. Every negative result should be re-examined to understand the underlying reasons for its occurrence.")
            fscol1, fscol2, fscol3 =st.columns([3,3,3])
            with fscol1:
                # growth rate drop check
                try:
                    eps_select = ['Diluted EPS']
                    eps_values = income_statement.loc[eps_select].fillna(0).values.flatten()
                    eps_values = [float(eps_value) for eps_value in eps_values]
                    has_negative_value = any(value < 0 for value in eps_values)
                    if has_negative_value:
                        drop_result = "BAD"
                    else:
                        try:
                            ebitda_values = sa_metrics_df2[sa_metrics_df2['Fiscal Year'].isin(['EBITDA'])].iloc[:, 2:].fillna(0).values.flatten()
                            ebitda_values = pd.Series(ebitda_values).fillna(0)
                            ebitda_values = [int(value.replace(",", "")) for value in ebitda_values]
                            ebitda_growth_rates = [((ebitda_values[i] - ebitda_values[i + 1]) / ebitda_values[i + 1]) * 100 for i in range(len(ebitda_values) - 1)]
                            drop_threshold = -10
                            is_drop = ebitda_growth_rates[0] < drop_threshold
                            if is_drop: drop_result = "BAD"
                            else: drop_result = "GOOD"
                        except: drop_result = 'N/A'
                except: drop_result = 'N/A'   
                # revenue check
                try:
                    revenue_select = ['Total Revenue']
                    revenue_values = income_statement.loc[revenue_select].fillna(0).values.flatten()
                    revenue_values = [float(revenue_value) for revenue_value in revenue_values]
                    is_revenue_growing = all(revenue_values[i] >= revenue_values[i + 1] for i in range(len(revenue_values) - 1))
                    if is_revenue_growing: revenue_result = "GOOD"
                    else: revenue_result = "BAD"
                except: revenue_result = 'N/A'
                # gross margin check
                try: 
                    gross_margin_values = sa_metrics_df2[sa_metrics_df2['Fiscal Year'].isin(['Gross Margin'])].iloc[:, 2:].fillna(0).values.flatten()
                    gross_margin_values = [value.rstrip('%') if isinstance(value, str) else value for value in gross_margin_values]
                    gross_margin_values = pd.to_numeric(gross_margin_values, errors='coerce')
                    gross_margin_values = pd.Series(gross_margin_values).fillna(0)
                    gross_margin_values = [float(value) for value in gross_margin_values]
                    is_gross_growing = all(gross_margin_values[i] >= gross_margin_values[i + 1] for i in range(len(gross_margin_values) - 1))
                    if is_gross_growing: gross_growing_result = "GOOD"
                    else: gross_growing_result = "BAD"
                except: gross_growing_result = 'N/A'
                # operating expense vs revenue
                try:
                    op_expense_select = ['Operating Expense']
                    op_expense_values = income_statement.loc[op_expense_select].fillna(0).values.flatten()
                    op_expense_values = [float(op_expense_value) for op_expense_value in op_expense_values]
                    def calculate_growth_rates(values):
                        growth_rates = []
                        for i in range(len(values) - 1):
                            if values[i + 1] != 0:  
                                growth_rate = (values[i] - values[i + 1]) / values[i + 1] * 100
                                growth_rates.append(growth_rate)
                            else:
                                growth_rates.append(None)  
                        return growth_rates
                    revenue_growth_rates = calculate_growth_rates(revenue_values)
                    op_expense_growth_rates = calculate_growth_rates(op_expense_values)
                    revenue_growth_avg_rate = pd.Series(revenue_growth_rates).replace(0, np.nan).mean()
                    op_expense_avg_rate = pd.Series(op_expense_growth_rates).replace(0, np.nan).mean()
                    op_expenses_grow_faster = op_expense_avg_rate > revenue_growth_avg_rate
                    if op_expenses_grow_faster: op_expense_result = "BAD"
                    else: op_expense_result = "GOOD"
                except: op_expense_result = 'N/A'
                # operating expenses check
                try:
                    op_expense_values_filtered = [value for value in op_expense_values if value > 0]
                    percentile_90 = np.percentile(op_expense_values_filtered, 90)
                    large_charges = [value for value in op_expense_values if value >= percentile_90]
                    large_charges_frequency = len(large_charges)
                    frequent_occurrence_threshold = 2
                    extraordinary_charges = large_charges_frequency >= frequent_occurrence_threshold
                    if extraordinary_charges: hop_exp_result= "BAD"
                    else: hop_exp_result = "GOOD"
                except: hop_exp_result = 'N/A'
                # dilution check
                try:
                    share_change_values = sa_metrics_df2[sa_metrics_df2['Fiscal Year'].isin(['Shares Change (YoY)'])].iloc[:, 2:].fillna(0).values.flatten()
                    share_change_values = [value.rstrip('%') if isinstance(value, str) else value for value in share_change_values]
                    share_change_values = pd.to_numeric(share_change_values, errors='coerce')
                    share_change_values = pd.Series(share_change_values).fillna(0)
                    share_change_values = [float(value) for value in share_change_values]
                    if share_change_values[0] > 0: share_dilution_result = "BAD"
                    else: share_dilution_result = "GOOD"
                except: share_dilution_result = 'N/A'
                # tax rate check
                try: 
                    tax_select = ['Tax Rate For Calcs']
                    tax_rate_values = income_statement.loc[tax_select].fillna(0).values.flatten()
                    tax_rate_values = [float(tax_rate_value) for tax_rate_value in tax_rate_values]
                    decline_value = tax_rate_values[0]-tax_rate_values[1]
                    if decline_value<0: tax_rate_result= "BAD"
                    else: tax_rate_result = "GOOD"
                except: tax_rate_result = 'N/A'
                income_checklist_data = [
                {"Income Statment Checklist": "Growth rate should be consistently increasing", "Result": drop_result},
                {"Income Statment Checklist": "Revenue should be consistently growing", "Result": revenue_result},
                {"Income Statment Checklist": "Gross margin should be consistently growing", "Result": gross_growing_result},
                {"Income Statment Checklist": "Opex growth rate < Revenue growth rate", "Result": op_expense_result},
                {"Income Statment Checklist": "Large operating expenses should not exist", "Result": hop_exp_result},
                {"Income Statment Checklist": "High share dilution should not exist", "Result": share_dilution_result},
                {"Income Statment Checklist": "Sudden decline of tax rate should not exist", "Result": tax_rate_result}
                ]
                df_income = pd.DataFrame(income_checklist_data)
                st.dataframe(df_income.style.applymap(highlight_result, subset=['Result']),use_container_width=True, hide_index=True)
            
            with fscol2:
                #cash and debt ####### #Total Debt and Current Assets
                try:
                    totaldebt_value = balance_sheet.loc['Total Debt'].iloc[0]
                    if pd.isna(totaldebt_value):
                        totaldebt_value = 0
                except:
                    totaldebt_value = 0
                try:
                    currentasset_value = balance_sheet.loc['Current Assets'].iloc[0]
                    if pd.isna(currentasset_value):
                        currentasset_value = 0
                except:
                    currentasset_value = 0
                try:
                    if totaldebt_value / currentasset_value < 1.1:
                        debt_to_asset_result = 'GOOD'
                    else: 
                        debt_to_asset_result = 'BAD'
                except:
                    debt_to_asset_result = 'N/A'

                try:
                    cash_value = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
                    if pd.isna(cash_value):
                        cash_value = 0
                except:
                    cash_value = 0
                try:
                    if cash_value > totaldebt_value:
                        cash_and_debt_result = 'GOOD'
                    else: 
                        cash_and_debt_result = 'BAD'
                except:
                    cash_and_debt_result = 'N/A'
                # Accounts receivable
                try:
                    accountReceivable_value = balance_sheet.loc['Accounts Receivable'].iloc[0]
                    if pd.isna(accountReceivable_value):
                        accountReceivable_value = 0
                except:
                    accountReceivable_value = 0
                try:
                    if accountReceivable_value > 0:
                        accountReceivable_result = "BAD"
                    else: accountReceivable_result = "GOOD"
                except:
                    accountReceivable_result = "N/A"
                # Inventory
                try:
                    inventory_value = balance_sheet.loc['Inventory'].iloc[0]
                    if pd.isna(inventory_value):
                        inventory_value = 0
                except:
                    inventory_value = 0
                try:
                    if inventory_value > 0:
                        inventory_result = "BAD"
                    else: inventory_result = "GOOD"
                except:
                    inventory_result = "N/A"
                # Current liabilities and cash
                try:
                    currentLiabilities_value = balance_sheet.loc['Current Liabilities'].iloc[0]
                    if pd.isna(currentLiabilities_value):
                        currentLiabilities_value = 0
                except:
                    currentLiabilities_value = 0
                try:
                    if currentLiabilities_value > cash_value:
                        currentLiabilities_result = "BAD"
                    else: currentLiabilities_result = "GOOD"
                except:
                    currentLiabilities_result = "N/A"
                # short term & long term debt
                try:
                    sdebt_value = balance_sheet.loc['Short Term Debt'].iloc[0]
                    if pd.isna(sdebt_value):
                        sdebt_value = 0
                except:
                    sdebt_value = 0
                try:
                    ldebt_value = balance_sheet.loc['Long Term Debt'].iloc[0]
                    if pd.isna(ldebt_value):
                        ldebt_value = 0
                except:
                    ldebt_value = 0
                try:
                    if sdebt_value > 0 or ldebt_value > 0:
                        snldebt_result = "BAD"
                    else: snldebt_result = "GOOD"
                except:
                    snldebt_result = "N/A"
                # goodwill and total asset
                try:
                    goodwill_value = balance_sheet.loc['Goodwill'].iloc[0]
                    if pd.isna(goodwill_value):
                        goodwill_value = 0
                except:
                    goodwill_value = 0
                try:
                    totalassets_value = balance_sheet.loc['Total Assets'].iloc[0]
                    if pd.isna(totalassets_value):
                        totalassets_value = 0
                except:
                    totalassets_value = 0
                try:
                    if goodwill_value > 0.1 * totalassets_value:
                        goodwillandTotalassets_result = "BAD"
                    else: goodwillandTotalassets_result = "GOOD"
                except:
                    goodwillandTotalassets_result = "N/A"
                # preferred stocks
                try:
                    preferred_value = balance_sheet.loc['Preferred Stock'].iloc[0]
                    if pd.isna(preferred_value):
                        preferred_value = 0
                except:
                    preferred_value = 0
                try:
                    if preferred_value > 0: 
                        preferred_result = 'BAD'
                    else:
                        preferred_result = 'GOOD'
                except:
                    preferred_result = 'N/A'
                # retained earnings 
                try:
                    retained_earnings_select = ['Retained Earnings']
                    retained_earnings_values = balance_sheet.loc[retained_earnings_select].fillna(0).values.flatten()
                    retained_earnings_values = [float(retained_earnings_values) for retained_earnings_values in retained_earnings_values]
                    no_negative_retained_earnings = all(retained_earnings >= 0 for retained_earnings in retained_earnings_values)
                    retained_earnings_result = all(retained_earnings_values[i] >= retained_earnings_values[i+1] for i in range(len(retained_earnings_values) - 1))
                except:
                    no_negative_retained_earnings = retained_earnings_result = "N/A"
                # treasury stock
                try:
                    treasury_value = balance_sheet.loc['Treasury Shares Number'].iloc[0]
                    if pd.isna(treasury_value):
                        treasury_value = 0
                except:
                    treasury_value = 0
                try:
                    if treasury_value > 0:
                        treasury_result = 'GOOD'
                    else: 
                        treasury_result = 'BAD'
                except:
                    treasury_result = 'N/A'
                # deferred revenue
                try:
                    deferredRevenue_value = balance_sheet.loc['Non Current Deferred Revenue'].iloc[0]
                    if pd.isna(deferredRevenue_value):
                        deferredRevenue_value = 0
                except:
                    deferredRevenue_value = 0
                try:
                    if deferredRevenue_value > 0:
                        deferredRevenue_result = 'GOOD'
                    else: 
                        deferredRevenue_result = 'BAD'
                except:
                    deferredRevenue_result = 'N/A'
                  
                balance_checklist_data = [
                {"Balance Sheet Checklist": "Cash should be more than debt", "Result": debt_to_asset_result},
                {"Balance Sheet Checklist": "Accounts receivables should not exist", "Result": accountReceivable_result},
                {"Balance Sheet Checklist": "Inventory should not exist", "Result": inventory_result},
                {"Balance Sheet Checklist": "Current liabilities < Cash", "Result": currentLiabilities_result},
                {"Balance Sheet Checklist": "No short-term and long-term debts", "Result": snldebt_result},
                {"Balance Sheet Checklist": "Goodwill should be less than 10% of total assets", "Result": goodwillandTotalassets_result},
                {"Balance Sheet Checklist": "Preferred stocks should not exist", "Result": preferred_result},
                {"Balance Sheet Checklist": "Retained earnings should be positive", "Result": 'GOOD' if no_negative_retained_earnings else 'BAD'},
                {"Balance Sheet Checklist": "Retained earnings should be growing", "Result": 'GOOD' if retained_earnings_result else 'BAD'},
                {"Balance Sheet Checklist": "Treasury stocks should exist", "Result": treasury_result},
                {"Balance Sheet Checklist": "Deferred revenue should exist", "Result": deferredRevenue_result}
                ]
                df_balance = pd.DataFrame(balance_checklist_data)
                st.dataframe(df_balance.style.applymap(highlight_result, subset=['Result']),use_container_width=True, hide_index=True)

            with fscol3:
                # net income
                try:
                    netIncome_select = ['Net Income From Continuing Operations']
                    netIncome_values = cashflow_statement.loc[netIncome_select].fillna(0).values.flatten()
                    netIncome_values = [float(netIncome_values) for netIncome_values in netIncome_values]
                    no_negative_netIncome = all(netIncome >= 0 for netIncome in netIncome_values)
                    netIncome_result = all(netIncome_values[i] >= netIncome_values[i+1] for i in range(len(netIncome_values) - 1))
                except: no_negative_netIncome = netIncome_result = "N/A"
                # share based compensation and net income
                try:
                    shareBasedCompensation_value = cashflow_statement.loc['Stock Based Compensation'].iloc[0]
                    if pd.isna(shareBasedCompensation_value):
                        shareBasedCompensation_value = 0
                except:
                    shareBasedCompensation_value = 0
                try:
                    netIncome_value = cashflow_statement.loc['Net Income From Continuing Operations'].iloc[0]
                    if pd.isna(netIncome_value):
                        netIncome_value = 0
                except:
                    netIncome_value = 0
                try:
                    if shareBasedCompensation_value < 0.1 * netIncome_value:
                        shareBased_and_netIncome_result = 'GOOD'
                    else: 
                        shareBased_and_netIncome_result = 'BAD'
                except:
                    shareBased_and_netIncome_result = 'N/A'
                # operating cash flow
                try:
                    operatingCash_value = cashflow_statement.loc['Operating Cash Flow'].iloc[0]
                    if pd.isna(operatingCash_value):
                        operatingCash_value = 0
                except:
                    operatingCash_value = 0
                try:
                    if operatingCash_value > netIncome_value:
                        operating_result = 'GOOD'
                    else: 
                        operating_result = 'BAD'
                except: 
                    operating_result = 'N/A'
                # free cash flow
                try:
                    fcf_value = cashflow_statement.loc['Free Cash Flow'].iloc[0]
                    if pd.isna(fcf_value):
                        fcf_value = 0
                except:
                    fcf_value = 0
                try:
                    if fcf_value > netIncome_value:
                        fcf_result = 'GOOD'
                    else: 
                        fcf_result = 'BAD'
                except: 
                    fcf_result = 'N/A'
                # CAPEX
                try:
                    capex_value = cashflow_statement.loc['Capital Expenditure'].iloc[0]
                    netincome_number = income_statement.loc['Net Income'].iloc[0]
                    if pd.isna(capex_value):
                        capex_value = 0
                except:
                    capex_value = 0
                try:
                    if capex_value / netincome_number < 0.25:
                        capex_result = 'GOOD'
                    else:
                        capex_result = 'BAD'
                except:
                    capex_result = 'N/A'
                # debt reduction
                try:
                    debtIssuance_select = ['Issuance Of Debt']
                    debtIssuance_values = cashflow_statement.loc[debtIssuance_select].fillna(0).values.flatten()
                    debtIssuance_values = [float(debtIssuance_values) for debtIssuance_values in debtIssuance_values]
                    negative_debtIssuance = all(debtIssuance <= 0 for debtIssuance in debtIssuance_values)
                except: negative_debtIssuance = "N/A"
                # stock repurchase
                try:
                    stock_repurchase_value = cashflow_statement.loc['Repurchase Of Capital Stock'].iloc[0]
                    if pd.isna(stock_repurchase_value):
                        stock_repurchase_value = 0
                except:
                    stock_repurchase_value = 0
                try:
                    if stock_repurchase_value <= 0:
                        stock_repurchase_result = 'GOOD'
                    else:
                        stock_repurchase_result = 'BAD'
                except:
                    stock_repurchase_result = 'N/A'
                # dividend paid
                try:
                    dividendPaid_select = ['Cash Dividends Paid']
                    dividendPaid_values = cashflow_statement.loc[dividendPaid_select].fillna(0).values.flatten()
                    dividendPaid_values = [float(dividendPaid_values) for dividendPaid_values in dividendPaid_values]
                    negative_dividendPaid = all(dividendPaid <= 0 for dividendPaid in dividendPaid_values)
                except: negative_dividendPaid = "N/A"
                # changes in cash balance
                try:
                    cic_select = ['Changes In Cash']
                    cic_values = cashflow_statement.loc[cic_select].fillna(0).values.flatten()
                    cic_values = [float(cid_values) for cid_values in cid_values]
                    cic_result = all(cic_values[i] >= cic_values[i+1] for i in range(len(cic_values) - 1))
                except: cic_result = "N/A"
                cash_flow_checklist_data = [
                {"Cash Flow Statement Checklist": "Net incomes should be positive", "Result": 'GOOD' if no_negative_netIncome else 'BAD'},
                {"Cash Flow Statement Checklist": "Net incomes should be growing", "Result": 'GOOD' if netIncome_result else 'BAD'},
                {"Cash Flow Statement Checklist": "Stock based compensation < 10% of net income", "Result": shareBased_and_netIncome_result},
                {"Cash Flow Statement Checklist": "OCF should be higher than net income", "Result": operating_result},
                {"Cash Flow Statement Checklist": "FCF should be higher than net income", "Result": fcf_result},
                {"Cash Flow Statement Checklist": "CAPEX margin should be less than 25%", "Result": capex_result},
                {"Cash Flow Statement Checklist": "Debt issuance should be negative", "Result": 'GOOD' if negative_debtIssuance else 'BAD'},
                {"Cash Flow Statement Checklist": "Stock repurchase should be negative", "Result": stock_repurchase_result},
                {"Cash Flow Statement Checklist": "Dividend paid should be negative", "Result": 'GOOD' if negative_dividendPaid else 'BAD'},
                {"Cash Flow Statement Checklist": "Change in cash should be increasing", "Result": 'GOOD' if cic_result else 'BAD'}
                ]
                df_cash_flow = pd.DataFrame(cash_flow_checklist_data)
                st.dataframe(df_cash_flow.style.applymap(highlight_result, subset=['Result']),use_container_width=True, hide_index=True)


#############################################                    #############################################
############################################# Guru Analysis Data #############################################
#############################################                    ############################################# 

        with guru_checklist:
            try:
                st.info("Please be advised that no company is flawless, and it is unlikely for any company to achieve good results in all aspects outlined in the following checklists. Every negative result should be re-examined to understand the underlying reasons for its occurrence. These checklists are provided solely for reference purposes and should not be considered as financial advice.")
                st.caption("This page is derived from the financial statements data provided by Yahoo Finance.")
                # Long Term Debt and Net Income
                try:
                    longterm_debt_value = balance_sheet.loc['Long Term Debt'].iloc[0]
                    if pd.isna(longterm_debt_value):
                        longterm_debt_value = 0
                except:
                    longterm_debt_value = 0
                try:
                    netincome_value = income_statement.loc['Net Income'].iloc[0]
                    if pd.isna(netincome_value):
                        netincome_value = 0
                except:
                    netincome_value = 0
                if longterm_debt_value <= 5 * netincome_value:
                    lt_debt_result = 'GOOD'
                else:
                    lt_debt_result = 'BAD'
                #st.write(longterm_debt_value)
                #st.write(netincome_value)
                #st.write(f'Longterm Debt: {lt_debt_result}')
                # EPS
                eps_current = float(eps_values[0])
                no_negative_earnings = all(eps >= 0 for eps in eps_values)
                generally_increasing = all(eps_values[i] >= eps_values[i+1] for i in range(len(eps_values) - 1))
                eps_5y = next((val for val in reversed(eps_values) if val != 0), None)
                if eps_current > eps_5y:
                    eps_result = 'GOOD'
                else:
                    eps_result = 'BAD'
                #st.write(eps_current)
                #st.write(eps_5y)
                #st.write(f'EPS: {eps_result}')
                # Earning Yield
                earning_yield_value = eps/price
                if earning_yield_value > 0.03:
                    earningyield_result = 'GOOD'
                else:
                    earningyield_result = 'BAD'
                #st.write(earning_yield_value)
                #st.write(f'Earning yield: {earningyield_result}')
                # ROE
                roe_values = sa_metrics_df[sa_metrics_df['Fiscal Year'].isin(['Return on Equity (ROE)'])].iloc[:, 2:].fillna(0).values.flatten()
                roe_values = [value.rstrip('%') if isinstance(value, str) else value for value in roe_values]
                roe_values = pd.to_numeric(roe_values, errors='coerce')
                roe_values = pd.Series(roe_values).fillna(0)
                roe_avg_value = roe_values[roe_values != 0].mean()
                if roe_avg_value >= 15:
                    roe_avg_result = 'GOOD'
                else:
                    roe_avg_result = 'BAD'
                #st.write(f'ROE: {roe_avg_result}')
                # ROIC
                roic_values = sa_metrics_df[sa_metrics_df['Fiscal Year'].isin(['Return on Capital (ROIC)'])].iloc[:, 2:].fillna(0).values.flatten()
                roic_values = [value.rstrip('%') if isinstance(value, str) else value for value in roic_values]
                roic_values = pd.to_numeric(roic_values, errors='coerce')
                roic_values = pd.Series(roic_values).fillna(0)
                roic_avg_value = roic_values[roic_values != 0].mean()
                if roic_avg_value >= 12:
                    roic_avg_result = 'GOOD'
                else:
                    roic_avg_result = 'BAD'
                #st.write(f'ROIC: {roic_avg_result}')
                # Dividend yield
                divyield = 'N/A' if dividendYield == 'N/A' else dividendYield*100
                if divyield != 'N/A':
                    if divyield > 0:
                        divyield_result = 'GOOD'
                    else: 
                        divyield_result = 'BAD'
                else:
                    divyield_result = 'No Dividend'
                # Current Ratio
                current_ratio_value = 0 if current_ratio =='N/A' else current_ratio
                if current_ratio_value >= 1.5:
                    current_ratio_result = 'GOOD'
                else:
                    current_ratio_result = 'BAD'
                # Total Debt and Current Assets
                
                # PE Ratio
                pe = 'N/A' if peRatio == 'N/A' else peRatio
                if pe != 'N/A':
                    if pe <= 15:
                        pe_result = 'GOOD'
                    else: 
                        pe_result = 'BAD'
                else:
                    pe_result = "No data for PE Ratio."
                # PB Ratio
                pb = 'N/A' if pbRatio == 'N/A' else pbRatio
                if pb != 'N/A':
                    if pb <= 1.5:
                        pb_result = 'GOOD'
                    else: 
                        pb_result = 'BAD'
                else:
                    pb_result = "No data for PB Ratio."
                # Gross Margin
                grossm_value = 'N/A' if grossmargin == 'N/A' else float(grossmargin)*100
                if grossm_value != 'N/A':
                    if grossm_value > 40:
                        grossm_result = 'GOOD'
                    else: 
                        grossm_result = 'BAD'
                else:
                    grossm_result = 'N/A'
                # SG&A Margin
                try:
                    sgna_value = income_statement.loc['Selling General And Administration'].iloc[0]
                    if pd.isna(sgna_value):
                        sgna_value = 0
                except:
                    sgna_value = 0
                try:
                    gross_value = income_statement.loc['Gross Profit'].iloc[0]
                    if pd.isna(gross_value):
                        gross_value = 0
                except:
                    gross_value = 0
                try:
                    if sgna_value / gross_value < 0.3:
                        sgna_result = 'GOOD'
                    else:
                        sgna_result = 'BAD'
                except:
                    sgna_result = 'N/A'
                # R&D Margin
                try:
                    rnd_value = income_statement.loc['Research And Development'].iloc[0]
                    if pd.isna(rnd_value):
                        rnd_value = 0
                except:
                    rnd_value = 0
                try:
                    if rnd_value / gross_value < 0.3:
                        rnd_result = 'GOOD'
                    else: 
                        rnd_result = 'BAD'
                except:
                    rnd_result = 'N/A'
                # Depreciation Margin
                try:
                    depreciation_value = income_statement.loc['Depreciation And Amortization In Income Statement'].iloc[0]
                    if pd.isna(depreciation_value):
                        depreciation_value = 0
                except:
                    depreciation_value = 0
                try:
                    if depreciation_value / gross_value < 0.1:
                        depreciation_result = 'GOOD'
                    else: 
                        depreciation_result = 'BAD'
                except:
                    depreciation_result = 'N/A'
                # Interest Margin
                try:
                    operate_value = income_statement.loc['Operating Income'].iloc[0]
                    if pd.isna(operate_value):
                        operate_value = 0
                except:
                    operate_value = 0
                try:
                    interest_value = income_statement.loc['Interest Expense'].iloc[0]
                    if pd.isna(interest_value):
                        interest_value = 0
                except:
                    interest_value = 0
                try:
                    if interest_value / operate_value < 0.15:
                        interest_result = 'GOOD'
                    else: 
                        interest_result = 'BAD'
                except:
                    interest_result = 'N/A'
                # Income tax expense
                try:
                    taxrate_value = income_statement.loc['Tax Rate For Calcs'].iloc[0]
                    if pd.isna(taxrate_value):
                        taxrate_value = 0
                except:
                    taxrate_value = 0
                try:
                    if taxrate_value >= 0.21:
                        taxrate_result = 'GOOD'
                    else: 
                        taxrate_result = 'BAD'
                except:
                    taxrate_result = 'N/A'
                # Profit Margin
                profitm_value = 'N/A' if profitmargin == 'N/A' else float(profitmargin)*100
                if profitm_value != 'N/A':
                    if profitm_value > 20:
                        profitm_result = 'GOOD'
                    else: 
                        profitm_result = 'BAD'
                else:
                    profitm_result = 'N/A'
                # Cash and Debt
                
                #Adjust Debt to Equity
                try:
                    liability_value = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
                    if pd.isna(liability_value):
                        liability_value = 0
                except:
                    liability_value = 0
                try:
                    shareholder_value = balance_sheet.loc['Stockholders Equity'].iloc[0]
                    if pd.isna(shareholder_value):
                        shareholder_value = 0
                except:
                    shareholder_value = 0
                try:
                    treasury_share_value = balance_sheet.loc['Treasury Shares Number'].iloc[0]
                    if pd.isna(treasury_share_value):
                        treasury_share_value = 0
                except:
                    shareholder_value = 0
                try:
                    if liability_value / shareholder_value < 1:
                        debt_to_equity_result = 'GOOD'
                    else:
                        debt_to_equity_result = 'BAD'
                except:
                    debt_to_equity_result = 'N/A'
                # Preferred stock
                
                # Retained Earnings
                
                # Treasury Stock
                try:
                    if treasury_share_value > 0:
                        treasury_share_result = 'GOOD'
                    else: 
                        treasury_share_result = 'BAD'
                except:
                    treasury_share_result = 'N/A'
                # Capex Margin
                
                # PE and forward PE
                try:
                    if float(pe_value) < 25:
                        peterlynch_pe_result = 'GOOD'
                    else:
                        peterlynch_pe_result = 'BAD'
                except:
                    peterlynch_pe_result = 'N/A'
                try:
                    if float(forwardPe_value) < 15:
                        peterlynch_forwardpe_result = 'GOOD'
                    else:
                        peterlynch_forwardpe_result = 'BAD'
                except:
                    peterlynch_forwardpe_result = 'N/A'
                # Institutional ownership
                try: 
                    if float(institutionsPct) < 0.1:
                        peterlynch_instututional_result = 'GOOD'
                    else: 
                        peterlynch_instututional_result = 'BAD'
                except:
                    peterlynch_instututional_result = 'N/A'
                # Insider ownership
                try:
                    if float(insiderPct) > 0.2:
                        peterlynch_insider_result = 'GOOD'
                    else:
                        peterlynch_insider_result = 'BAD'
                except:
                    peterlynch_insider_result = 'N/A'
                # EPS growth
                eps_growth_values = sa_metrics_df2.loc[sa_metrics_df2['Fiscal Year'] == 'EPS Growth'].iloc[:, 2:].fillna(0).values.flatten()
                eps_growth_values = [value.rstrip('%') if isinstance(value, str) else value for value in eps_growth_values]
                eps_growth_values = pd.to_numeric(eps_growth_values, errors='coerce')
                eps_growth_values = pd.Series(eps_growth_values).fillna(0)
                try:
                    if all(eps_growth_values > 15):
                        peterlynch_epsgrowth_result = 'GOOD'
                    else:
                        peterlynch_epsgrowth_result = 'BAD'
                except: 
                    peterlynch_epsgrowth_result = 'N/A'
                # D/E ratio 
                try:
                    if float(deRatio_value) < 0.35:
                        peterlynch_deratio_result = 'GOOD'
                    else:
                        peterlynch_deratio_result = 'BAD'
                except:
                    peterlynch_deratio_result = 'N/A'

                buffettology_data = [
                    {"Buffettology Checklist": "Track record of no negative earnings", "Result": 'GOOD' if no_negative_earnings else 'BAD'},
                    {"Buffettology Checklist": "Generally increasing EPS", "Result": 'GOOD' if generally_increasing else 'BAD'},
                    {"Buffettology Checklist": "Long-term debt should not be more than 5 times annual earnings", "Result": lt_debt_result},
                    {"Buffettology Checklist": "Earnings yield should be higher than the long-term Treasury yield", "Result": earningyield_result},
                    {"Buffettology Checklist": "5 years average ROE should be at least 15%", "Result": roe_avg_result},
                    {"Buffettology Checklist": "5 years average ROIC should be at least 12%", "Result": roic_avg_result},
                    {"Buffettology Checklist": "EPS from current year should be higher than EPS from last 5 years", "Result": eps_result}
                ]
                df_buffettology = pd.DataFrame(buffettology_data)

                grahamprinciple_data = [
                    {"Graham's Principles Checklist": "Dividend yield should be greater than 0", "Result": divyield_result},
                    {"Graham's Principles Checklist": "Current ratio should be greater than or equal 1.5", "Result": current_ratio_result},
                    {"Graham's Principles Checklist": "Total debt to current asset should be less than 1.1", "Result": debt_to_asset_result},
                    {"Graham's Principles Checklist": "Positive earnings growth for last 5 years", "Result": 'GOOD' if generally_increasing else 'BAD'},
                    {"Graham's Principles Checklist": "PE ratio should be less than or equal 15", "Result": pe_result},
                    {"Graham's Principles Checklist": "PB ratio should be less than or equal 1.5", "Result": pb_result},
                    {"Graham's Principles Checklist": "EPS from current year should be higher than EPS from last 5 years", "Result": eps_result}
                ]
                df_grahamprinciple = pd.DataFrame(grahamprinciple_data)

                warren_incomestatement_data = [
                    {"Income Statement Checklist": "Gross Margin should be greater than 40%", "Result": grossm_result},
                    {"Income Statement Checklist": "SG&A Margin should be less than 30%", "Result": sgna_result},
                    {"Income Statement Checklist": "R&D Margin should be less than 30%", "Result": rnd_result},
                    {"Income Statement Checklist": "Depreciation Margin should be less than 10%", "Result": depreciation_result},
                    {"Income Statement Checklist": "Interest Expense Margin should be less than 15%", "Result": interest_result},
                    {"Income Statement Checklist": "Tax Margin should be at corporate tax rate", "Result": taxrate_result},
                    {"Income Statement Checklist": "Net Margin should be greater than 20%", "Result": profitm_result},
                    {"Income Statement Checklist": "Should have track record of no negative earnings", "Result": 'GOOD' if no_negative_earnings else 'BAD'},
                    {"Income Statement Checklist": "EPS should be increasing", "Result": 'GOOD' if generally_increasing else 'BAD'}
                ]
                df_warren_incomestatement = pd.DataFrame(warren_incomestatement_data)

                warren_balancesheet_data = [
                    {"Balance Sheet Checklist": "Cash should be more than debt", "Result": cash_and_debt_result},
                    {"Balance Sheet Checklist": "Adjusted Debt to Equity should be less than 0.8", "Result": debt_to_equity_result},
                    {"Balance Sheet Checklist": "Preferred stocks should not exist", "Result": preferred_result},
                    {"Balance Sheet Checklist": "Retained Earnings should be consistently growing", "Result": 'GOOD' if retained_earnings_result else 'BAD'},
                    {"Balance Sheet Checklist": "Treasury stocks should exist", "Result": treasury_share_result}
                ]
                df_warren_balancesheet = pd.DataFrame(warren_balancesheet_data)

                warren_cashflowstatement_data = [
                    {"Cash Flow Statement Checklist": "CAPEX margin should be less than 25%", "Result": capex_result}
                ]
                df_warren_cashflowstatement = pd.DataFrame(warren_cashflowstatement_data)

                peterlynch_data = [
                    {"Peter Lynch Investing Checklist": "Trailing PE should be less than 25", "Result": peterlynch_pe_result},
                    {"Peter Lynch Investing Checklist": "Forward PE should be less than 15", "Result": peterlynch_forwardpe_result},
                    {"Peter Lynch Investing Checklist": "Institutional ownership should be less than 10%", "Result": peterlynch_instututional_result},
                    {"Peter Lynch Investing Checklist": "Insider ownership should be greater than 20%", "Result": peterlynch_insider_result},
                    {"Peter Lynch Investing Checklist": "EPS growth for last 5 years should be greater than 15%", "Result": peterlynch_epsgrowth_result},
                    {"Peter Lynch Investing Checklist": "Debt/Equity should be less than 35%", "Result": peterlynch_deratio_result}
                ]
                df_peterlynch = pd.DataFrame(peterlynch_data)

                st.subheader("Warren Buffett", divider ='gray')
                guru_col1, guru_col2 = st.columns([2,3])
                with guru_col1:
                    guru_logo_url1='./Image/warren-buffett.png'
                    st.image(guru_logo_url1,width=300)
                with guru_col2:
                    st.dataframe(df_buffettology.style.applymap(highlight_result, subset=['Result']),use_container_width=True, hide_index=True)
                guru_col3, guru_col4, guru_col5 = st.columns([2,2,2])
                with guru_col3:
                    st.dataframe(df_warren_incomestatement.style.applymap(highlight_result, subset=['Result']),use_container_width=True, hide_index=True)
                with guru_col4:
                    st.dataframe(df_warren_balancesheet.style.applymap(highlight_result, subset=['Result']),use_container_width=True, hide_index=True)
                with guru_col5:
                    st.dataframe(df_warren_cashflowstatement.style.applymap(highlight_result, subset=['Result']),use_container_width=True, hide_index=True)
                ''
                st.subheader("Benjamin Graham", divider ='gray')
                guru_col6, guru_col7 = st.columns([2,3])
                with guru_col6:
                    guru_logo_url2='./Image/benjamin-graham.png'
                    st.image(guru_logo_url2,width=300)
                with guru_col7:
                    st.dataframe(df_grahamprinciple.style.applymap(highlight_result, subset=['Result']),use_container_width=True, hide_index=True)
                ''
                st.subheader("Peter Lynch", divider ='gray')
                guru_col8, guru_col9 = st.columns([2,3])
                with guru_col8:
                    guru_logo_url2='./Image/peter-lynch.png'
                    st.image(guru_logo_url2,width=300)
                with guru_col9:
                    st.dataframe(df_peterlynch.style.applymap(highlight_result, subset=['Result']),use_container_width=True, hide_index=True)
            except:
                st.warning("Guru checklist is currently unavailable.")

#############################################                #############################################
############################################# Insider Trades #############################################
#############################################                ############################################# 
        with insider_trades:
            def highlight_insider_trades(val):
                if val == 'Buy':
                    bscolor = 'green'
                elif val == 'Sell':
                    bscolor = 'red'
                else:
                    bscolor ='#AAB2BD'
                return f'background-color: {bscolor}; color: white'
            try:
                insider_mb = pd.DataFrame(insider_mb).iloc[:, :-2]
                def is_valid_date(value):
                    try:
                        pd.to_datetime(value)
                        return True
                    except ValueError:
                        return False
                unwanted_string = "Get Insider Trades Delivered To Your InboxEnter your email address below to receive a concise daily summary of insider buying activity, insider selling activity and changes in hedge fund holdings."
                filtered_insider_mb = insider_mb[
                    insider_mb["Transaction Date"].apply(lambda x: is_valid_date(x) and x != unwanted_string)
                ]
                st.dataframe(filtered_insider_mb.style.applymap(highlight_insider_trades, subset=['Buy/Sell']), use_container_width=True, hide_index=True, height = 600)
                st.caption("Data source: Market Beat")
            except: st.warning("Insider information is not available.")

#############################################                         #############################################
############################################# Technical Analysis Data #############################################
#############################################                         #############################################
        with technicalAnalysis_data:
            st.info("It is important to note that investment decisions should not be based solely on technical analysis. Technical analysis primarily relies on historical price movements and cannot predict future outcomes with certainty.")
            st.caption("This page is derived from the historical price data provided by Yahoo Finance.")
            try:
                            extended_data = extended_data_r 
                            macd_data = macd_data_r 
                            rsi_data = rsi_data_r 
                            ta_data = ta_data_r
                            ta_data = ta_data[['High', 'Low', 'Close']].copy()
                            if extended_data.empty:
                                st.error("No data available for the specified ticker. Please check the ticker symbol and try again.")
                            else:
                                #SMA
                                extended_data['SMA20'] = extended_data['Close'].rolling(window=20).mean()
                                extended_data['SMA50'] = extended_data['Close'].rolling(window=50).mean()
                                extended_data['SMA200'] = extended_data['Close'].rolling(window=200).mean()
                                last_year_start = (end_date - datetime.timedelta(days=int(1 * 365)))
                                data = extended_data.loc[extended_data.index >= last_year_start]
                                data.columns = data.columns.map('_'.join)
                                data.columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA20', 'SMA50', 'SMA200']
                                volume_colors = ['green' if data['Close'][i] >= data['Open'][i] else 'red' for i in range(len(data))]
                                max_volume = data['Volume'].max()
                                #MACD
                                macd_data['EMA12'] = macd_data['Close'].ewm(span=12, adjust=False).mean()
                                macd_data['EMA26'] = macd_data['Close'].ewm(span=26, adjust=False).mean()
                                macd_data['MACD'] = macd_data['EMA12'] - macd_data['EMA26']
                                macd_data['Signal'] = macd_data['MACD'].ewm(span=9, adjust=False).mean()
                                macd_data['MACD_Hist'] = macd_data['MACD'] - macd_data['Signal']
                                macd_data['Crossover'] = macd_data['MACD'] > macd_data['Signal']
                                macd_data['Bullish_Crossover'] = (macd_data['Crossover'] != macd_data['Crossover'].shift(1)) & (macd_data['Crossover'] == True)
                                macd_data['Bearish_Crossover'] = (macd_data['Crossover'] != macd_data['Crossover'].shift(1)) & (macd_data['Crossover'] == False)
                                macd_latest_bullish = macd_data['Bullish_Crossover'].iloc[-1]
                                macd_latest_bearish = macd_data['Bearish_Crossover'].iloc[-1]
                                #macd_data = macd_data[macd_data.index.dayofweek < 5]
                                #RSI
                                change = rsi_data["Close"].diff()
                                change.dropna(inplace=True)
                                up = change.apply(lambda x: max(x, 0))
                                down = change.apply(lambda x: -min(x, 0))
                                rsi_length = 14
                                avg_up = up.ewm(alpha=1/rsi_length, min_periods=rsi_length).mean()
                                avg_down = down.ewm(alpha=1/rsi_length, min_periods=rsi_length).mean()
                                rsi_data['RSI'] = 100 - (100 / (1 + avg_up / avg_down))
                                rsi_data['RSI'] = rsi_data['RSI'].apply(lambda x: 100 if avg_down.iloc[0] == 0 else (0 if avg_up.iloc[0] == 0 else x))
                                latest_rsi = rsi_data['RSI'].iloc[-1]
                                prev_rsi = rsi_data['RSI'].iloc[-2]
                                # Stochastic Oscillator (%K and %D)
                                ta_data['Low14'] = ta_data['Low'].rolling(window=14).min()
                                ta_data['High14'] = ta_data['High'].rolling(window=14).max()
                                ta_data['%K'] = 100 * ((ta_data['Close'] - ta_data['Low14']) / (ta_data['High14'] - ta_data['Low14']))
                                ta_data['%D'] = ta_data['%K'].rolling(window=3).mean()
                                ta_data['STOCH'] = ta_data['%D']
                                # Average Directional Index (ADX)
                                ta_data['+DM'] = np.where((ta_data['High'] - ta_data['High'].shift(1)) > (ta_data['Low'].shift(1) - ta_data['Low']), 
                                                    ta_data['High'] - ta_data['High'].shift(1), 0)
                                ta_data['-DM'] = np.where((ta_data['Low'].shift(1) - ta_data['Low']) > (ta_data['High'] - ta_data['High'].shift(1)), 
                                                    ta_data['Low'].shift(1) - ta_data['Low'], 0)
                                ta_data['TR'] = np.maximum(ta_data['High'] - ta_data['Low'], 
                                                        np.maximum(abs(ta_data['High'] - ta_data['Close'].shift(1)), 
                                                                abs(ta_data['Low'] - ta_data['Close'].shift(1))))
                                ta_data['ATR'] = ta_data['TR'].rolling(window=14).mean()
                                ta_data['+DI'] = 100 * (ta_data['+DM'] / ta_data['ATR']).rolling(window=14).mean()
                                ta_data['-DI'] = 100 * (ta_data['-DM'] / ta_data['ATR']).rolling(window=14).mean()
                                ta_data['DX'] = 100 * abs((ta_data['+DI'] - ta_data['-DI']) / (ta_data['+DI'] + ta_data['-DI']))
                                ta_data['ADX'] = ta_data['DX'].rolling(window=14).mean()
                                # Williams %R
                                ta_data['Williams %R'] = ((ta_data['High14'] - ta_data['Close']) / (ta_data['High14'] - ta_data['Low14'])) * -100
                                # Commodity Channel Index (CCI)
                                ta_data['Mean Price'] = (ta_data['High'] + ta_data['Low'] + ta_data['Close']) / 3
                                ta_data['CCI'] = (ta_data['Mean Price'] - ta_data['Mean Price'].rolling(window=20).mean()) / (0.015 * ta_data['Mean Price'].rolling(window=20).std())
                                # Rate of Change (ROC)
                                ta_data['ROC'] = ((ta_data['Close'] - ta_data['Close'].shift(12)) / ta_data['Close'].shift(12)) * 100
                                # Ultimate Oscillator (UO)
                                ta_data['BP'] = ta_data['Close'] - np.minimum(ta_data['Low'], ta_data['Close'].shift(1))
                                ta_data['TR_UO'] = np.maximum(ta_data['High'] - ta_data['Low'], 
                                                        np.maximum(abs(ta_data['High'] - ta_data['Close'].shift(1)), 
                                                                    abs(ta_data['Low'] - ta_data['Close'].shift(1))))
                                ta_data['Avg7'] = ta_data['BP'].rolling(window=7).sum() / ta_data['TR_UO'].rolling(window=7).sum()
                                ta_data['Avg14'] = ta_data['BP'].rolling(window=14).sum() / ta_data['TR_UO'].rolling(window=14).sum()
                                ta_data['Avg28'] = ta_data['BP'].rolling(window=28).sum() / ta_data['TR_UO'].rolling(window=28).sum()
                                ta_data['UO'] = 100 * (4 * ta_data['Avg7'] + 2 * ta_data['Avg14'] + ta_data['Avg28']) / 7
                                # Bollinger Bands
                                bb_period = 20
                                ta_data['BB_middle'] = ta_data['Close'].rolling(window=bb_period).mean()
                                bb_std = ta_data['Close'].rolling(window=bb_period).std()
                                ta_data['BB_upper'] = ta_data['BB_middle'] + (bb_std * 2)
                                ta_data['BB_lower'] = ta_data['BB_middle'] - (bb_std * 2)
                                # S&R levels
                                def find_support_resistance(data, window=20, threshold=0.05):
                                    highs = []
                                    lows = []
                                    for i in range(window, len(data)-window):
                                        if all(data['High'].iloc[i] >= data['High'].iloc[i-window:i]) and \
                                           all(data['High'].iloc[i] >= data['High'].iloc[i+1:i+window]):
                                            highs.append((data.index[i], data['High'].iloc[i]))
                                        if all(data['Low'].iloc[i] <= data['Low'].iloc[i-window:i]) and \
                                           all(data['Low'].iloc[i] <= data['Low'].iloc[i+1:i+window]):
                                            lows.append((data.index[i], data['Low'].iloc[i]))
                                    def group_levels(levels, threshold):
                                        levels = sorted(levels, key=lambda x: x[1])
                                        grouped = []
                                        current_group = [levels[0]]
            
                                        for level in levels[1:]:
                                            if abs(level[1] - current_group[0][1])/current_group[0][1] < threshold:
                                                current_group.append(level)
                                            else:
                                                avg_price = sum(x[1] for x in current_group)/len(current_group)
                                                earliest_date = min(x[0] for x in current_group)
                                                grouped.append((earliest_date, avg_price))
                                                current_group = [level]
                                        avg_price = sum(x[1] for x in current_group)/len(current_group)
                                        earliest_date = min(x[0] for x in current_group)
                                        grouped.append((earliest_date, avg_price))
                                        return grouped
                                    resistance_levels = group_levels(highs, threshold)
                                    support_levels = group_levels(lows, threshold)
                                    return support_levels, resistance_levels
                                #
                                fig = go.Figure()
                                fig_macd = go.Figure()
                                fig_rsi = go.Figure()
                                fig_bb = go.Figure()
                                fig_sr = go.Figure()
                                #
                                rsi_latest = rsi_data['RSI'].iloc[-1]
                                rsi_score = 80 if rsi_latest < 30 else 20 if rsi_latest > 70 else 50
                                macd_latest = macd_data['MACD'].iloc[-1]
                                signal_latest = macd_data['Signal'].iloc[-1]
                                macd_score = 80 if macd_latest > signal_latest else 20
                                ma_scores = []
                                if extended_data['Close'].iloc[-1] > extended_data['SMA20'].iloc[-1]: ma_scores.append(80)
                                else: ma_scores.append(20) 
                                if extended_data['Close'].iloc[-1] > extended_data['SMA50'].iloc[-1]: ma_scores.append(80)
                                else: ma_scores.append(20)
                                if extended_data['Close'].iloc[-1] > extended_data['SMA200'].iloc[-1]: ma_scores.append(80)
                                else: ma_scores.append(20)
                                ma_score = np.mean(ma_scores)
                                stoch_latest =  ta_data['STOCH'].iloc[-1]
                                stoch_score = 80 if stoch_latest > 80 else 20 if stoch_latest < 20 else 50
                                adx_latest = ta_data['ADX'].iloc[-1]
                                adx_score = 80 if adx_latest > 25 else 50
                                williamsr_latest = ta_data['Williams %R'].iloc[-1]
                                williamsr_score = 80 if williamsr_latest < -80 else 20 if williamsr_latest > -20 else 50
                                cci_latest = ta_data['CCI'].iloc[-1]
                                cci_score = 80 if cci_latest > 100 else 20 if cci_latest < -100 else 50
                                roc_latest = ta_data['ROC'].iloc[-1]
                                roc_score = 80 if roc_latest > 0 else 20
                                uo_latest = ta_data['UO'].iloc[-1]
                                uo_score = 80 if uo_latest > 70 else 20 if uo_latest < 30 else 50
                                overall_score = np.mean([rsi_score, macd_score, ma_score,stoch_score, adx_score, williamsr_score, cci_score, roc_score, uo_score])
                                #
                                def get_signal(price, sma, period):
                                    if price > sma:
                                        return f"🟢  {ticker}'s share price is ${price:.2f} and {period}SMA is {sma:.2f}, suggesting a BUY signal."
                                    else:
                                        return f"🔴  {ticker}'s share price is ${price:.2f} and {period}SMA is {sma:.2f}, suggesting a SELL signal."
                                def get_shortsignal(price, sma, period):
                                    if price > sma:
                                        return "Buy"
                                    else:
                                        return "Sell"
                                def detect_cross(data, short_sma, long_sma, short_period, long_period):
                                    last_cross = None
                                    if extended_data[short_sma].iloc[-2] < extended_data[long_sma].iloc[-2] and extended_data[short_sma].iloc[-1] > extended_data[long_sma].iloc[-1]:
                                        last_cross = f"🟢  Golden Cross: {short_period}SMA crossed above the {long_period}SMA."
                                    elif extended_data[short_sma].iloc[-2] > extended_data[long_sma].iloc[-2] and extended_data[short_sma].iloc[-1] < extended_data[long_sma].iloc[-1]:
                                        last_cross = f"🔴  Death Cross: {short_period}SMA crossed below the {long_period}SMA."
                                    return last_cross
                                cross_20_50 = detect_cross(data, 'SMA20', 'SMA50', 20, 50)
                                cross_50_200 = detect_cross(data, 'SMA50', 'SMA200', 50, 200)
                                def get_sentiment_label(score):
                                    if score <= 20:
                                        return "Strong Negative Bias"
                                    elif score <= 40:
                                        return "Negative Bias"
                                    elif score <= 60:
                                        return "Neutral"
                                    elif score <= 80:
                                        return "Positive Bias"
                                    else:
                                        return "Strong Positive Bias"
                                def consensus(value, thresholds):
                                    if value < thresholds[0]:
                                        return "Strong Sell"
                                    elif value < thresholds[1]:
                                        return "Sell"
                                    elif value < thresholds[2]:
                                        return "Neutral"
                                    elif value < thresholds[3]:
                                        return "Buy"
                                    else:
                                        return "Strong Buy"
                                def create_gauge(title, score):
                                    label = get_sentiment_label(score)
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge",
                                        value=score,  
                                        number={'font': {'size': 24}},  
                                        title={'text': title, 'font': {'size': 20}},
                                        gauge={'axis': {'range': [0, 100]},
                                            'bar': {'color': "#5F9BEB"},
                                            'steps': [
                                                {'range': [0, 15], 'color': "#da4453", 'name': 'Strong Neg'},
                                                {'range': [15, 45], 'color': "#e9573f", 'name': 'Neg'},
                                                {'range': [45, 55], 'color': "#f6bb42", 'name': 'Neutral'},
                                                {'range': [55, 85], 'color': "#a0d468", 'name': 'Pos'},
                                                {'range': [85, 100], 'color': "#37bc9b", 'name': 'Strong Pos'}]}))
                                    fig.add_annotation(x=0.5, y=0.25, text=label, showarrow=False, font=dict(size=20))
                                    fig.update_layout(
                                        font=dict(size=14),
                                        margin=dict(t=10, b=0, l=50, r=50),
                                        height=350
                                        )
                                    return fig
                                #thresholds for table
                                ta_data['STOCH Consensus'] = ta_data['%K'].astype(float).apply(lambda x: consensus(x, [20, 40, 60, 80]))
                                ta_data['ADX Consensus'] = ta_data['ADX'].astype(float).apply(lambda x: "Strong Trend" if x > 25 else "Weak Trend")
                                ta_data['Williams %R Consensus'] = ta_data['Williams %R'].astype(float).apply(lambda x: consensus(x, [-80, -50, -20, 0]))
                                ta_data['CCI Consensus'] = ta_data['CCI'].astype(float).apply(lambda x: consensus(x, [-100, -50, 50, 100]))
                                ta_data['ROC Consensus'] = ta_data['ROC'].astype(float).apply(lambda x: consensus(x, [-5, 0, 5, 10]))
                                ta_data['UO Consensus'] = ta_data['UO'].astype(float).apply(lambda x: consensus(x, [30, 50, 70, 80]))
                                #
                                fig.add_trace(go.Candlestick(
                                    x=data.index,
                                    open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                                    name="Candlestick",
                                    showlegend=False,
                                    increasing_line_width=0.5, decreasing_line_width=0.5,
                                    increasing_line_color='rgba(0,150,0,1)',
                                    decreasing_line_color='rgba(150,0,0,1)',
                                    opacity=1
                                ))
                                fig.add_trace(go.Scatter(
                                    x=data.index, y=data['SMA20'],
                                    line=dict(color='#3BAFDA', width=1),
                                    name="SMA 20",
                                    opacity=0.5
                                ))
                                fig.add_trace(go.Scatter(
                                    x=data.index, y=data['SMA50'],
                                    line=dict(color='#F6BB42', width=1),
                                    name="SMA 50",
                                    opacity=0.5
                                ))
                                fig.add_trace(go.Scatter(
                                    x=data.index, y=data['SMA200'],
                                    line=dict(color='#D772AD', width=1.5),
                                    name="SMA 200",
                                    opacity=0.5
                                ))
                                fig.add_trace(go.Bar(
                                    x=data.index, y=data['Volume'],
                                    marker=dict(color=volume_colors),
                                    name="Volume",
                                    yaxis="y2",
                                    showlegend=False,  
                                    opacity=0.3  
                                ))
                                tick_vals = data.index[::30]
                                tick_text = [date.strftime("%b %Y") for date in tick_vals]
                                fig.update_layout(
                                    title={"text":f"Price Data with Moving Average & RSI", "font": {"size": 30}},
                                    xaxis_rangeslider_visible=False,
                                    xaxis=dict(
                                        type="category",
                                        showgrid=True,
                                        ticktext=tick_text,
                                        tickvals=tick_vals,
                                        showticklabels=False 
                                    ),
                                    yaxis=dict(
                                        title="Price (USD)",
                                        side="left",
                                        showgrid=True
                                    ),
                                    yaxis2=dict(
                                        side="right",
                                        overlaying="y",
                                        showgrid=False,
                                        range=[0, max_volume * 3],
                                        showticklabels=False
                                    ),
                                    margin=dict(l=0, r=0, t=None, b=0),
                                    height=None,
                                    legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.010)
                                )
                                fig_macd.add_trace(go.Scatter(
                                    x=macd_data.index, y=macd_data['MACD'],
                                    line=dict(color='#3BAFDA', width=1.5),
                                    opacity=0.5,
                                    name="MACD"
                                ))
                                fig_macd.add_trace(go.Scatter(
                                    x=macd_data.index, y=macd_data['Signal'],
                                    line=dict(color='#F6BB42', width=1),
                                    opacity=1,
                                    name="Signal"
                                ))
                                fig_macd.add_trace(go.Bar(
                                    x=macd_data.index, y=macd_data['MACD_Hist'].where(macd_data['MACD_Hist'] >= 0, 0),  
                                    marker=dict(color='green'),
                                    showlegend=False,
                                    opacity=0.5,
                                    name="MACD Histogram (Above Zero)"
                                ))
                                fig_macd.add_trace(go.Bar(
                                    x=macd_data.index, y=macd_data['MACD_Hist'].where(macd_data['MACD_Hist'] < 0, 0),  
                                    marker=dict(color='red'),
                                    showlegend=False,
                                    opacity=0.5,
                                    name="MACD Histogram (Below Zero)"
                                ))
                                tick_vals = macd_data.index[::30]
                                tick_text = [date.strftime("%b %Y") for date in tick_vals]
                                fig_macd.update_layout(
                                    title={"text":f"MACD Chart", "font": {"size": 30}}, xaxis_title=None, yaxis_title="MACD Value",
                                    xaxis_rangeslider_visible=False,
                                    xaxis=dict(
                                        type="category",
                                        ticktext=tick_text,
                                        tickvals=tick_vals,
                                        showgrid=True
                                    ),
                                    legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.010)
                                )
                                fig_rsi.add_trace(go.Scatter(
                                    x=rsi_data.index, y=rsi_data['RSI'],
                                    line=dict(color='#D772AD', width=1),
                                    showlegend=False,
                                    name="RSI"
                                ))
                                fig_rsi.add_hline(y=70, line=dict(color='red', width=1, dash='dash'),annotation_text="70", annotation_position="top left",showlegend=False, name="Overbought")
                                fig_rsi.add_hline(y=30, line=dict(color='green', width=1, dash='dash'),annotation_text="30", annotation_position="bottom left",showlegend=False, name="Oversold")
                                tick_vals_rsi = rsi_data.index[::30]
                                tick_text_rsi = [date.strftime("%b %Y") for date in tick_vals_rsi]
                                fig_rsi.update_layout(
                                    xaxis_title=None,
                                    yaxis_title="RSI",
                                    xaxis=dict(
                                        type="category",
                                        ticktext=tick_text_rsi,
                                        tickvals=tick_vals_rsi,
                                        showgrid=True
                                    ),
                                    yaxis=dict(range=[0, 100], showgrid=False, showticklabels=True),
                                    height=200,
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.010)
                                )
                                #
                                fig_bb.add_trace(go.Candlestick(
                                    x=data.index,
                                    open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                                    name="Price",
                                    showlegend=False,
                                    increasing_line_width=0.5, decreasing_line_width=0.5,
                                    increasing_line_color='rgba(0,150,0,1)',
                                    decreasing_line_color='rgba(150,0,0,1)',
                                    opacity=1
                                ))
                                fig_bb.add_trace(go.Scatter(
                                    x=ta_data.index,
                                    y=ta_data['BB_middle'],
                                    line=dict(color='orange', width=1),
                                    name="Middle Band (20 SMA)",
                                    opacity=0.5
                                ))
                                fig_bb.add_trace(go.Scatter(
                                    x=ta_data.index,
                                    y=ta_data['BB_upper'],
                                    line=dict(color='red', width=1),
                                    name="Upper Band",
                                    fill='tonexty',  
                                    fillcolor='rgba(255,0,0,0.1)',
                                    opacity=0.5
                                ))
                                fig_bb.add_trace(go.Scatter(
                                    x=ta_data.index,
                                    y=ta_data['BB_middle'],
                                    line=dict(color='orange', width=1),
                                    name="Middle Band (20 SMA)",
                                    opacity=0,
                                    showlegend=False
                                ))
                                fig_bb.add_trace(go.Scatter(
                                    x=ta_data.index,
                                    y=ta_data['BB_lower'],
                                    line=dict(color='green', width=1),
                                    name="Lower Band",
                                    fill='tonexty',
                                    fillcolor='rgba(0,255,0,0.1)',
                                    opacity=0.5
                                ))
                                tick_vals = data.index[::30]
                                tick_text = [date.strftime("%b %Y") for date in tick_vals]
                                fig_bb.update_layout(
                                    title={"text": "Price Data with Bollinger Bands", "font": {"size": 30}},
                                    xaxis_rangeslider_visible=False,
                                    xaxis=dict(
                                        type="category",
                                        showgrid=True,
                                        ticktext=tick_text,
                                        tickvals=tick_vals
                                    ),
                                    yaxis=dict(
                                        title="Price (USD)",
                                        side="left",
                                        showgrid=True
                                    ),
                                    margin=dict(l=0, r=0, t=None, b=0),
                                    height=None,
                                    legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.010)
                                )
                                #
                                current_price = data['Close'].iloc[-1]
                                support_levels, resistance_levels = find_support_resistance(data)
                                fig_sr.add_trace(go.Candlestick(
                                    x=data.index,
                                    open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                                    name="Price",
                                    showlegend=False,
                                    increasing_line_width=0.5, decreasing_line_width=0.5,
                                    increasing_line_color='rgba(0,150,0,1)',
                                    decreasing_line_color='rgba(150,0,0,1)',
                                    opacity=1
                                ))
                                all_levels = [(date, level) for date, level in resistance_levels + support_levels]
                                all_levels = sorted(all_levels, key=lambda x: x[1], reverse=True)
                                used_positions = set()
                                label_spacing = 2  
                                for start_date, level in all_levels:
                                    is_resistance = current_price < level
                                    line_color = 'red' if is_resistance else 'green'
                                    label_prefix = 'R: ' if is_resistance else 'S: '
                                    fig_sr.add_trace(go.Scatter(
                                        x=[start_date, data.index[-1]],
                                        y=[level, level],
                                        mode='lines',
                                        line=dict(color=line_color, width=1, dash='dot'),
                                        opacity=0.7,
                                        showlegend=False,
                                        name=f"{'Resistance' if is_resistance else 'Support'} {level:.2f}"
                                    ))
                                    label_y = level
                                    while label_y in used_positions:
                                        label_y += label_spacing if is_resistance else -label_spacing
                                    used_positions.add(label_y)
                                    fig_sr.add_annotation(
                                        x=data.index[-1],
                                        y=label_y,
                                        text=f"{label_prefix}{level:.2f}",
                                        showarrow=True if label_y != level else False,
                                        arrowhead=2,
                                        arrowsize=1,
                                        arrowwidth=1,
                                        arrowcolor=line_color,
                                        xshift=50,
                                        font=dict(size=10, color=line_color),
                                        xanchor='left'
                                    )
                                fig_sr.update_layout(
                                    title={"text": "Support and Resistance Levels", "font": {"size": 30}},
                                    xaxis_rangeslider_visible=False,
                                    xaxis=dict(
                                        type="category",
                                        showgrid=True,
                                        ticktext=tick_text,
                                        tickvals=tick_vals
                                    ),
                                    yaxis=dict(
                                        title="Price (USD)",
                                        side="left",
                                        showgrid=True
                                    ),
                                    margin=dict(l=0, r=0, t=None, b=0),
                                    height=None,
                                    legend=dict(
                                        yanchor="top",
                                        y=0.99,
                                        xanchor="left",
                                        x=0.010
                                    )
                                )
                                #
                                if macd_latest > signal_latest:
                                    macd_signal = f"🟢  The Moving Averages Convergence Divergence (MACD) indicator for {ticker} is {macd_latest:.2f} and the signal line is at {signal_latest:.2f}, suggesting it is a BUY signal."
                                    macd_shortsignal = "Buy"
                                else:
                                    macd_signal = f"🔴  The Moving Averages Convergence Divergence (MACD) indicator for {ticker} is {macd_latest:.2f} and the signal line is at {signal_latest:.2f}, suggesting it is a SELL signal."
                                    macd_shortsignal = "Sell"
                                if macd_latest_bullish:
                                    crossover_signal = "🟢  Bullish Crossover: MACD line crossed above the signal line."
                                elif macd_latest_bearish:
                                    crossover_signal = "🔴  Bearish Crossover: MACD line crossed below the signal line."
                                else:
                                    crossover_signal = "🔵  No recent crossover detected."
                                #
                                if latest_rsi < 30:
                                    rsi_signal = f"🟢  {ticker}'s Relative Strength Index (RSI) is {latest_rsi:.2f}, suggesting a BUY signal."
                                    rsi_shortsignal = "Buy"
                                elif 30 <= latest_rsi <= 70:
                                    rsi_signal = f"🔵  {ticker}'s Relative Strength Index (RSI) is {latest_rsi:.2f}, suggesting a NEUTRAL."
                                    rsi_shortsignal = "Neutral"
                                else:
                                    rsi_signal = f"🔴  {ticker}'s Relative Strength Index (RSI) is {latest_rsi:.2f}, suggesting a SELL signal."
                                    rsi_shortsignal = "Sell"
            
                                try:
                                    if latest_rsi > 70:
                                        if prev_rsi >= latest_rsi:
                                            trend_analysis = "🔴  The RSI is above 70 and declining, indicating a potential reversal from overbought."
                                        else:
                                            trend_analysis = "🔴  The RSI is above 70 and holding, indicating continued overbought conditions."
                                    elif latest_rsi < 30:
                                        if prev_rsi <= latest_rsi:
                                            trend_analysis = "🟢  The RSI is below 30 and rising, indicating a potential reversal from oversold."
                                        else:
                                            trend_analysis = "🟢  The RSI is below 30 and holding, indicating continued oversold conditions."
                                    elif 30 < latest_rsi < 70:
                                        if latest_rsi < 50 and prev_rsi < latest_rsi:
                                            trend_analysis = "🔺  The RSI is approaching 50 from below, indicating strengthening momentum toward neutral."
                                        elif latest_rsi < 50 and prev_rsi > latest_rsi:
                                            trend_analysis = "🔻  The RSI is approaching 30 from neutral, indicating weakening momentum toward oversold condition."
                                        elif latest_rsi < 50 and prev_rsi == latest_rsi:
                                            trend_analysis = "🔻  The RSI is approaching 30 from neutral, indicating weakening momentum toward oversold condition."
                                        elif latest_rsi > 50 and prev_rsi > latest_rsi:
                                            trend_analysis = "🔻  The RSI is approaching 50 from above, indicating weakening momentum."
                                        elif latest_rsi > 50 and prev_rsi < latest_rsi:
                                            trend_analysis = "🔺  The RSI is approaching 70 from neutral, indicating strengthening momentum toward overbought condition."
                                        elif latest_rsi > 50 and prev_rsi == latest_rsi:
                                            trend_analysis = "🔺  The RSI is approaching 70 from neutral, indicating strengthening momentum toward overbought condition."
                                except: trend_analysis = ""
                                #
                                overall_col1, overall_col2 = st.columns ([2,3])
                                with overall_col1:
                                    st.plotly_chart(create_gauge("Overall Consensus", overall_score))
                                with overall_col2:
                                    latest_data =  ta_data[['STOCH', 'ADX', 'Williams %R', 'CCI', 'ROC', 'UO', 
                                    'STOCH Consensus', 'ADX Consensus', 'Williams %R Consensus', 
                                    'CCI Consensus', 'ROC Consensus', 'UO Consensus']].iloc[-1]
                                    indicator_names = ['SMA20','SMA50', 'SMA200', 'RSI', 'MACD', 'Stochastic Oscillator (STOCH)', 'Average Directional Index (ADX)', 'Williams %R', 'Commodity Channel Index (CCI)', 'Rate of Change (ROC)', 'Ultimate Oscillator (UO)']
                                    indicator_values = extended_data['SMA20'].iloc[-1], extended_data['SMA50'].iloc[-1], extended_data['SMA200'].iloc[-1], latest_rsi, macd_latest, *latest_data[['STOCH', 'ADX', 'Williams %R', 'CCI', 'ROC', 'UO']].values
                                    indicator_signals = get_shortsignal(price, data['SMA20'][-1], 20), get_shortsignal(price, data['SMA50'][-1], 50), get_shortsignal(price, data['SMA200'][-1], 200), rsi_shortsignal, macd_shortsignal, *latest_data[['STOCH Consensus', 'ADX Consensus', 'Williams %R Consensus', 
                                                                    'CCI Consensus', 'ROC Consensus', 'UO Consensus']].values
                                    formatted_values = [f"{value:.2f}" for value in indicator_values]
                                    summary_df = pd.DataFrame({
                                        'Technical Indicator': indicator_names,
                                        'Value': formatted_values,
                                        'Signal': indicator_signals
                                    })
                                    st.dataframe(summary_df,hide_index=True,use_container_width=True, height=300)
                                #st.subheader("",divider = 'gray')
                                gauge_col1, gauge_col2, gauge_col3 = st.columns([3,3,3])
                                with gauge_col1:
                                    st.plotly_chart(create_gauge("Moving Average Consensus", ma_score))
                                with gauge_col2:
                                    st.plotly_chart(create_gauge("MACD Consensus", macd_score))
                                with gauge_col3:
                                    st.plotly_chart(create_gauge("RSI Consensus", rsi_score))
                                
                                st.plotly_chart(fig)
                                st.plotly_chart(fig_rsi)
                                ''
                                ma_tcol1, ma_tcol2 = st.columns([3,3])
                                with ma_tcol1:
                                    st.write(get_signal(price, data['SMA20'][-1], 20))
                                    st.write(get_signal(price, data['SMA50'][-1], 50))
                                    st.write(get_signal(price, data['SMA200'][-1], 200))
                                    if cross_20_50:
                                        st.write(cross_20_50)
                                    else:
                                        st.write("🔵  No recent 20-50 SMAs crossover detected.")
                                    if cross_50_200:
                                        st.write(cross_50_200)
                                    else:
                                        st.write("🔵  No recent 50-200 SMAs crossover detected.")
                                with ma_tcol2:
                                    st.info("SMAs calculate the average price over a period, treating all past prices equally. If the current stock price is above the SMA, it suggests a buy signal, as the price is above the historical average for that period. A sell signal is suggested when the current price is below the SMA.")
                                ''
                                rsi_tcol1, rsi_tcol2 = st.columns([3,3])
                                with rsi_tcol1:
                                    st.write(rsi_signal)
                                    st.write(trend_analysis)
                                with rsi_tcol2:
                                    st.info("If RSI > 70, it generally indicates an Overbought condition. If RSI < 30, it generally indicates an Oversold condition. If RSI is between 30 and 70, it indicates a Neutral condition.")
                                st.subheader("",divider = 'gray')
                                
                                st.plotly_chart(fig_macd)
                                md_tcol1, md_tcol2 = st.columns([3,3])
                                with md_tcol1:
                                    st.write(macd_signal)
                                    st.write(crossover_signal)
                                with md_tcol2:
                                    st.info("The MACD Line is above the Signal Line, indicating a bullish crossover and the stock might be trending upward, so we interpret this as a Buy signal. When the MACD Line is below the Signal Line, it means bearish crossover and the stock might be trending downward, so we interpret this as a Sell signal.")
                                st.subheader("",divider = 'gray')
            
                                st.plotly_chart(fig_bb)
                                bb_col1, bb_col2 = st.columns([3,3])
                                with bb_col1:
                                    try:
                                        current_price = data['Close'].iloc[-1]
                                        upper_band = ta_data['BB_upper'].iloc[-1]
                                        lower_band = ta_data['BB_lower'].iloc[-1]
                                        if current_price >= upper_band:
                                            bb_signal = f"🔴 Current price is above the upper band {upper_band:.2f}, suggesting OVERBOUGHT conditions."
                                        elif current_price <= lower_band:
                                            bb_signal = f"🟢 Current price is below the lower band {lower_band:.2f}, suggesting OVERSOLD conditions."
                                        else:
                                            bb_signal = f"🔵 Current price is within the bands ({lower_band:.2f} - {upper_band:.2f}), suggesting NEUTRAL conditions."
                                    except Exception as e:
                                        bb_signal=""
                                    ''
                                    ''
                                    st.write(bb_signal)
                                with bb_col2:
                                    ''
                                    ''
                                    st.info("Bollinger Bands consist of a middle band (20-day SMA) and two outer bands (±2 standard deviations). When price moves outside these bands, it suggests potential overbought or oversold conditions. The bands also help identify volatility, as they widen during high volatility and narrow during low volatility periods.")
                                st.subheader("",divider = 'gray')
                                
                                st.plotly_chart(fig_sr)
                                sr_col1, sr_col2 = st.columns([3,3])
                                with sr_col1:
                                    msr_col1,msr_col2 = st.columns([3,3])
                                    all_levels = [(date, level) for date, level in resistance_levels + support_levels]
                                    all_levels = sorted(all_levels, key=lambda x: x[1], reverse=True)
                                    with msr_col1:
                                        ''
                                        ''
                                        st.subheader("Resistance Levels:")
                                        resistance_count = 1
                                        for date, level in all_levels:
                                            if current_price < level: 
                                                st.write(f"R{resistance_count}: ${level:.2f}")
                                                resistance_count += 1
                                    with msr_col2:
                                        ''
                                        ''
                                        st.subheader("Support Levels:")
                                        support_count = 1
                                        for date, level in all_levels:
                                            if current_price > level:  
                                                st.write(f"S{support_count}: ${level:.2f}")
                                                support_count += 1
                                with sr_col2:
                                    ''
                                    ''
                                    st.info("Support and Resistance levels are key price levels where the stock has historically found support (price stops falling) or resistance (price stops rising). These levels are important as they often act as psychological barriers and can indicate potential reversal points in price movement.")
                                
            except Exception as e: 
                st.write(e)
                st.warning("Failed to request historical price data.")

            ###Finviz picture
            # st.subheader("Price Data", divider ='gray')
            # pcol1, pcol2, pcol3 = st.columns ([0.5,3,0.5])
            # with pcol2:
            #     st.image(f'https://finviz.com/chart.ashx?t={ticker}')
            #     st.caption("Chart picture is obtained from finviz.com.")


#############################################           #############################################
############################################# News Data #############################################
#############################################           #############################################

        with news_data:
            try:
                st.caption("News data is sourced from Stockanalysis.com.")
                news_url = f'https://stockanalysis.com/stocks/{ticker}/'
                news_page = requests.get(news_url)
                soup = BeautifulSoup(news_page.text, "html.parser")
                img_elements = soup.findAll("img", attrs={"class": "h-full w-full rounded object-cover"})
                img_urls = [img.get('src') for img in img_elements]
                titles = soup.findAll("h3", attrs={"class":"mb-2 mt-3 text-xl font-bold leading-snug sm:order-2 sm:mt-0 sm:leading-tight"})
                links = [title.find('a').get('href') for title in titles]
                paragraphs = soup.findAll("p", attrs={"class":"overflow-auto text-[0.95rem] text-light sm:order-3"})
                sources = soup.findAll("div", attrs={"class":"mt-1 text-sm text-faded sm:order-1 sm:mt-0"})
                news = []
                for i in range(len(titles)):
                    news_item = {
                        'title': titles[i].get_text(strip=True),
                        'link': links[i],
                        'thumbnail': {'resolutions': [{'url': img_urls[i]}]} if i < len(img_urls) else {},
                        'publisher': sources[i].get_text(strip=True) if i < len(sources) else 'Unknown Publisher'
                    }
                    news.append(news_item)
                num_columns = 3
                columns = st.columns(num_columns)
                for i, news_item in enumerate(news):
                    title = news_item.get('title', 'No Title')
                    publisher = news_item.get('publisher', 'No Publisher')
                    link = news_item.get('link', '#')
                    provider_publish_time = news_item.get('providerPublishTime', 0)
                    thumbnails = news_item.get('thumbnail', {}).get('resolutions', [])
                    thumbnail_url = thumbnails[0]['url'] if thumbnails else None
                    column_index = i % num_columns
                    with columns[column_index]:
                        if thumbnail_url:
                            st.image(thumbnail_url, width=200)
                        st.subheader(title)
                        st.write(f"{publisher}")
                        st.write(f"**Link**: [Read more from this link]({link})")
                        st.write("---")
                    if column_index == (num_columns - 1):
                        st.write("")
            except: st.warning("Failed to get news.")
            ''

#############################################             #############################################
############################################# AI Analysis #############################################
#############################################             #############################################

        with ai_analysis:
                st.subheader("AI Stock Analysis", divider ='gray')
                if upper_ticker:
                    with st.spinner('Analyzing stock data...'):
                        st.markdown(analysis)
                        st.warning("This analysis, generated by AI, should not be the sole basis for investment decisions.")
            
    except Exception as e:
        st.error(f"Failed to fetch data. Please check your ticker again.")
        st.warning("This tool supports only tickers from the U.S. stock market. Please note that ETFs and cryptocurrencies are not available for analysis. If the entered ticker is valid but the tool does not display results, it may be due to missing data or a technical issue. Kindly try again later. If the issue persists, please contact the developer for further assistance.")
''

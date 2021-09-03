import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Merge with stock info
    df_qr = pd.read_csv('datasets/historical_qrs.csv')
    df_stock = pd.read_csv('datasets/historical_stock_data.csv')

    df_qr['Quarter end'] = pd.to_datetime(df_qr['Quarter end'], errors='coerce')
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')

    df_qr = df_qr.rename({'Stock': 'Ticker'}, axis=1)

    df_qr = df_qr.set_index('Ticker', 'Quarter end')
    df_stock = df_stock.set_index('Ticker', 'Date')

    df_qr = df_qr.sort_index(level=df_qr.index.names, kind='mergesort')
    df_stock = df_stock.sort_index(level=df_stock.index.names, kind='mergesort')

    quarter_ends = df_qr.index.get_level_values('Quarter end').unique()

    # Merge with stock info
    df_qr = pd.read_csv('datasets/historical_qrs.csv', index_col=['Stock'])
    df_info = pd.read_csv('datasets/stock_info.csv', index_col=['ticker'])

    constants = ['sector', 'state', 'country', 'industry', 'exchange', 'market', 'dividendRate', 'dividendYield']

    for ticker in df_qr.index.unique():
        for col in constants:
            df_qr.loc[ticker, col] = df_info.loc[ticker, col]

    df_qr.to_csv('datasets/historical_qrs.csv')


# ['zip', 'sector', 'fullTimeEmployees', 'longBusinessSummary', 'city', 'phone', 'state', 'country',
#  'companyOfficers', 'website', 'maxAge', 'address1', 'fax', 'industry', 'address2', 'ebitdaMargins',
#  'profitMargins', 'grossMargins', 'operatingCashflow', 'revenueGrowth', 'operatingMargins', 'ebitda',
#  'targetLowPrice', 'recommendationKey', 'grossProfits', 'freeCashflow', 'targetMedianPrice', 'currentPrice',
#  'earningsGrowth', 'currentRatio', 'returnOnAssets', 'numberOfAnalystOpinions', 'targetMeanPrice',
#  'debtToEquity', 'returnOnEquity', 'targetHighPrice', 'totalCash', 'totalDebt', 'totalRevenue',
#  'totalCashPerShare', 'financialCurrency', 'revenuePerShare', 'quickRatio', 'recommendationMean', 'exchange',
#  'shortName', 'longName', 'exchangeTimezoneName', 'exchangeTimezoneShortName', 'isEsgPopulated',
#  'gmtOffSetMilliseconds', 'underlyingSymbol', 'quoteType', 'symbol', 'underlyingExchangeSymbol', 'headSymbol',
#  'messageBoardId', 'uuid', 'market', 'annualHoldingsTurnover', 'enterpriseToRevenue', 'beta3Year',
#  'enterpriseToEbitda', '52WeekChange', 'morningStarRiskRating', 'forwardEps', 'revenueQuarterlyGrowth',
#  'sharesOutstanding', 'fundInceptionDate', 'annualReportExpenseRatio', 'totalAssets', 'bookValue',
#  'sharesShort', 'sharesPercentSharesOut', 'fundFamily', 'lastFiscalYearEnd', 'heldPercentInstitutions',
#  'netIncomeToCommon', 'trailingEps', 'lastDividendValue', 'SandP52WeekChange', 'priceToBook',
#  'heldPercentInsiders', 'nextFiscalYearEnd', 'yield', 'mostRecentQuarter', 'shortRatio',
#  'sharesShortPreviousMonthDate', 'floatShares', 'beta', 'enterpriseValue', 'priceHint',
#  'threeYearAverageReturn', 'lastSplitDate', 'lastSplitFactor', 'legalType', 'lastDividendDate',
#  'morningStarOverallRating', 'earningsQuarterlyGrowth', 'priceToSalesTrailing12Months', 'dateShortInterest',
#  'pegRatio', 'ytdReturn', 'forwardPE', 'lastCapGain', 'shortPercentOfFloat', 'sharesShortPriorMonth',
#  'impliedSharesOutstanding', 'category', 'fiveYearAverageReturn', 'previousClose', 'regularMarketOpen',
#  'twoHundredDayAverage', 'trailingAnnualDividendYield', 'payoutRatio', 'volume24Hr', 'regularMarketDayHigh',
#  'navPrice', 'averageDailyVolume10Day', 'regularMarketPreviousClose', 'fiftyDayAverage',
#  'trailingAnnualDividendRate', 'open', 'averageVolume10days', 'expireDate', 'algorithm', 'dividendRate',
#  'exDividendDate', 'circulatingSupply', 'startDate', 'regularMarketDayLow', 'currency', 'trailingPE',
#  'regularMarketVolume', 'lastMarket', 'maxSupply', 'openInterest', 'marketCap', 'volumeAllCurrencies',
#  'strikePrice', 'averageVolume', 'dayLow', 'ask', 'askSize', 'volume', 'fiftyTwoWeekHigh', 'fromCurrency',
#  'fiveYearAvgDividendYield', 'fiftyTwoWeekLow', 'bid', 'tradeable', 'dividendYield', 'bidSize', 'dayHigh',
#  'regularMarketPrice', 'logo_url']

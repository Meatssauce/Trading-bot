from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yf
yf.pdr_override()


if __name__ == '__main__':
    # Read and print the stock tickers that make up S&P500
    tickers = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    print(tickers.head())

    # Get the data for this tickers from yahoo finance
    df = pdr.get_data_yahoo(tickers.Symbol.to_list(), start='1993-06-29', group_by='Ticker', threads=True,
                            auto_adjust=True)

    # turn multilevel columns grouped by ticker into flat dataframe with an additional column called ticker
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

    df.to_csv('datasets/sp500_info.csv')
    print(df.head())

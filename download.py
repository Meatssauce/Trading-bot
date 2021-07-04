import requests
import re
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from urllib.request import urlopen
from tqdm import tqdm
from joblib import dump


html_page = urlopen('https://web.archive.org/web/20200123214339/http://www.stockpup.com/data')
soup = BeautifulSoup(html_page)
qr_by_tickers = {}

for tag in tqdm(soup.findAll('a', attrs={'href': re.compile('^/web/'), 'title': re.compile(r'\.csv$')})[-3:]):
    url = 'https://web.archive.org/' + tag.get('href')
    data = StringIO(requests.get(url).text)

    ticker = re.search(r'^fundamental_data_excel_(.*)\.csv$', tag.get('title'), re.IGNORECASE).group(1)

    qr = pd.read_csv(data)
    qr_by_tickers[ticker] = qr

# historical_qrs = pd.concat(qrs, keys=tickers, names=['Ticker', 'Row ID'])
# historical_qrs.to_csv('historical_qrs.csv', index=True)

dump(qr_by_tickers, 'qr_by_tickers.joblib')

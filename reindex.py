import requests
import re
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from urllib.request import urlopen
from tqdm import tqdm


html_page = urlopen('https://web.archive.org/web/20200123214339/http://www.stockpup.com/data')
soup = BeautifulSoup(html_page)

df = pd.read_csv('historical_qrs.csv')
df['Quarter end'] = pd.to_datetime(df['Quarter end'])
tickers = []
for tag in tqdm(soup.findAll('a', attrs={'href': re.compile('^/web/'), 'title': re.compile(r'\.csv$')})):
    tickers.append(re.search(r'^fundamental_data_excel_(.*)\.csv$', tag.get('title'), re.IGNORECASE).group(1))

qrs = []
last_date = df.loc[0, 'Quarter end']
last_i = 0
for i in df.index:
    date = df.loc[i, 'Quarter end']
    if date > last_date:
        qrs.append(df.iloc[last_i:i, :])
        last_i = i
    last_date = date
print()
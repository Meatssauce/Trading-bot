# 'input', id='company_search_text_box'  # data table header
# 'input', type_='submit', name='CF', value='Cash Flow'
# 'input', type_='checkbox', name='Total Operating Expenses', value='True'
# 'tr', id='finiancials-table-header'
# 'tbody', id='financials-table-body'  # data table
# <tr><th scope='row'>Net Income</th>
# <td>2323423</td>
# </tr>
#
# 'h1', id='title-banner'
from collections import defaultdict

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
from selenium.webdriver.support.wait import WebDriverWait
from tqdm import tqdm


def find(driver):
    element = driver.find_elements_by_id("data")
    if element:
        return element
    else:
        return False


if __name__ == '__main__':
    # Get list of sp500 stocks
    df = pd.read_csv('datasets/sp500_info.csv')
    df = df.dropna(subset=['Ticker'], how='any', axis=0)
    tickers = df['Ticker'].unique().tolist()

    # Scrape fundamentals for all
    # options = webdriver.ChromeOptions()
    # options.add_argument('headless')
    browser = webdriver.Chrome('assets/chromedriver')
    browser.get('http://graphfundamentals.com/')

    all_data = {'Balance Sheet': defaultdict(list), 'Income Statement': defaultdict(list),
                'Cash Flow': defaultdict(list)}
    for ticker in tqdm(tickers):
        # Search a ticker
        search_bar = browser.find_element_by_id('company_search_text_box')
        search_bar.clear()
        search_bar.send_keys(ticker + Keys.RETURN)

        # doc_buttons = browser.find_elements_by_css_selector("input[type='submit']")[:3]
        # for button, doc_title in zip(doc_buttons, all_data):
        for doc_title in tqdm(all_data, leave=False, desc='Scraping document'):
            button = browser.find_element_by_css_selector(f"input[type='submit'][value='{doc_title}']")

            # assert button.get_attribute('value') == doc_title

            button.click()
            check_boxes = browser.find_elements_by_css_selector("input[type='checkbox']")
            [check_box.click() for check_box in check_boxes]

            data = all_data[doc_title]

            table_headers = browser.find_element_by_css_selector("tr#financials-table-header")
            dates = table_headers.find_elements_by_css_selector("th[scope='row']")
            dates = [date.text for date in dates]
            data['Date'] += dates
            data['Ticker'] += len(dates) * [ticker]

            table_body = browser.find_element_by_css_selector("tbody#financials-table-body")
            for row in table_body.find_elements_by_css_selector('tr'):
                row_header = row.find_element_by_css_selector("th[scope='row']")
                elements = [e.text for e in row.find_elements_by_css_selector('td')]
                data[row_header] += elements

                # assert len(elements) == len(dates)

    # Save as dataframes
    for title, data in all_data.items():
        df = pd.DataFrame.from_dict(data)
        df.to_csv(f"datasets/sp500_{title.replace(' ', '')}.csv")

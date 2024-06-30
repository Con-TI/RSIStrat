from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from io import StringIO
import yfinance as yf

def find_stock_codes():
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver,10)
    idx = "https://www.idx.co.id/en/market-data/stocks-data/stock-list"
    driver.get(idx)
    select_element = wait.until(EC.presence_of_element_located((By.TAG_NAME,"select")))
    select = Select(select_element)
    select.select_by_visible_text("All")
    table_element = wait.until(EC.presence_of_element_located((By.TAG_NAME,"table")))
    table_html = table_element.get_attribute('outerHTML')
    df = pd.read_html(StringIO(table_html))[0]
    stock_codes = df.iloc[:,1].to_list()
    listing_years = [int(string[-4:]) for string in df.iloc[:,3].to_list()]
    return pd.Series(stock_codes), pd.Series(listing_years)

stock_codes, listing_years = find_stock_codes()
stock_codes = stock_codes[listing_years<2001]

stock_codes = [f"{code}.JK" for code in stock_codes]

df = yf.download(tickers=stock_codes,period="10y",interval='1d').drop(columns=['Adj Close'])
df.to_pickle('data.pkl')
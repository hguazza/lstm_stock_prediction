import yfinance as yf 
import pandas as pd


symbol = 'NVDA'
start_date = '2020-01-01'
end_date = '2025-12-01'

df = yf.download(symbol, start=start_date,end=end_date)

df.head()
# Lista exata das colunas que queremos (sem Adj Close)

features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Criando o novo dataframe filtrado
df_features = df[features]

df.to_csv('data/raw/nvda.csv', index=True)
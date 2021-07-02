from yahoofinancials import YahooFinancials

ticker = 'AAPL'
data = YahooFinancials(ticker)

quarterly_balance_sheet = data.get_financial_stmts('quarterly', 'balance')


print(data.get_financial_stmts('quarterly', 'balance'))

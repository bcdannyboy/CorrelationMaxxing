import requests
import yfinance as yf
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import minimize
import argparse
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.ERROR)

# Reduce logging for yfinance and urllib3
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

CBOE_INDICES = {
    "VIX": "^VIX",
    "SPX": "^GSPC",
    "DJIA": "^DJI",
    "NDX": "^NDX",
    "RUT": "^RUT",
    "VXD": "^VXD",
    "RVX": "^RVX",
    "VXAPL": "^VXAPL",
    "VXGOG": "^VXGOG",
    "VXIBM": "^VXIBM",
    "OVX": "^OVX",
    "GVZ": "^GVZ",
    "VXEWZ": "^VXEWZ",
    "VXEFA": "^VXEFA",
    "VXEEM": "^VXEEM",
    "VXX": "^VXX",
    "VXZ": "^VXZ",
    "VXAZN": "^VXAZN",
    "VXGS": "^VXGS"
}

def fetch_commodity_data(api_key, symbol):
    logging.info(f"Fetching commodity data for {symbol} from Financial Modeling Prep API")
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch data for {symbol}, Status code: {response.status_code}")
        return None

def fetch_stock_data(ticker, period="1y"):
    logging.info(f"Fetching stock data for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            logging.error(f"No data found for {ticker}, symbol may be delisted or incorrect")
            return None
        hist.reset_index(inplace=True)
        hist['Date'] = hist['Date'].dt.tz_localize(None)  # Ensure datetime format consistency
        return hist[['Date', 'Close']]
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_etf_holdings(api_key, etf_symbol, date="2023-09-30"):
    logging.info(f"Fetching holdings for ETF {etf_symbol}")
    url = f"https://financialmodelingprep.com/api/v4/etf-holdings?symbol={etf_symbol}&date={date}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        holdings = response.json()
        return [(holding['symbol'], holding['name']) for holding in holdings]
    else:
        logging.error(f"Failed to fetch ETF holdings for {etf_symbol}, Status code: {response.status_code}")
        return None

def fetch_commodities_list(api_key):
    logging.info("Fetching commodities list")
    url = f"https://financialmodelingprep.com/api/v3/symbol/available-commodities?apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        commodities = response.json()
        return {commodity['symbol']: commodity['name'] for commodity in commodities}
    else:
        logging.error(f"Failed to fetch commodities list, Status code: {response.status_code}")
        return None

def calculate_correlations(stock_data, commodities_data):
    correlations = {}
    for symbol, data in commodities_data.items():
        if data is not None:
            df = pd.DataFrame(data['historical'])
            if 'date' not in df.columns:
                logging.error(f"No 'date' column found in data for {symbol}")
                continue
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index = df.index.tz_localize(None)  # Ensure datetime format consistency
            combined = pd.merge(stock_data, df[['close']], left_on='Date', right_index=True, how='inner')
            if not combined.empty and 'Close' in combined.columns and 'close' in combined.columns:
                corr = combined['Close'].corr(combined['close'])
                correlations[symbol] = corr
    return correlations

def fetch_cboe_index_data():
    cboe_data = {}
    for name, ticker in CBOE_INDICES.items():
        period = "5d" if ticker.startswith("^VX") else "1y"
        data = fetch_stock_data(ticker, period=period)
        if data is not None:
            cboe_data[name] = data
    return cboe_data

def calculate_cboe_correlations(stock_data, cboe_data):
    correlations = {}
    for name, data in cboe_data.items():
        combined = pd.merge(stock_data, data[['Date', 'Close']], on='Date', suffixes=('', f'_{name}'))
        if not combined.empty:
            corr = combined['Close'].corr(combined[f'Close_{name}'])
            correlations[name] = corr
    return correlations

def optimize_portfolio(stock_data, num_stocks=10):
    returns = stock_data.pct_change().dropna()
    corr_matrix = returns.corr()

    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(corr_matrix, weights))
        return portfolio_variance

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(returns.columns)))
    initial_weights = np.array([1 / len(returns.columns)] * len(returns.columns))

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        logging.error("Optimization failed")
        return None

def calculate_internal_correlation(corr_matrix, weights):
    return np.dot(weights.T, np.dot(corr_matrix, weights))

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.01):
    excess_returns = portfolio_returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe_ratio

def backtest_portfolio(stock_data, weights):
    portfolio_returns = stock_data.pct_change().dropna().dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return portfolio_returns, cumulative_returns

def main(api_key, stock_tickers=None, etf_symbol=None, num_stocks=10):
    logging.info("Starting main function")
    
    # If ETF symbol is provided, fetch its holdings
    if etf_symbol:
        holdings = fetch_etf_holdings(api_key, etf_symbol)
        if holdings:
            stock_tickers = [ticker for ticker, name in holdings]
            stock_names = {ticker: name for ticker, name in holdings}
        else:
            logging.error(f"Failed to fetch holdings for ETF {etf_symbol}")
            return
    else:
        stock_names = {ticker: ticker for ticker in stock_tickers}

    if not stock_tickers:
        logging.error("No stock tickers provided")
        return

    # Fetch commodities list and their names
    commodities_names = fetch_commodities_list(api_key)
    if not commodities_names:
        logging.error("No commodities fetched")
        return
    
    # Fetch data for all stocks and commodities concurrently
    with ThreadPoolExecutor(max_workers=20) as executor:
        stock_futures = {executor.submit(fetch_stock_data, ticker.strip(), period="1y"): ticker.strip() for ticker in stock_tickers}
        commodity_futures = {executor.submit(fetch_commodity_data, api_key, commodity): commodity for commodity in commodities_names.keys()}
        
        stocks_data = {ticker: future.result() for future, ticker in stock_futures.items() if future.result() is not None}
        commodities_data = {symbol: future.result() for future, symbol in commodity_futures.items() if future.result() is not None}
    
    # Fetch CBOE index data
    cboe_data = fetch_cboe_index_data()
    
    # Prepare stock data for optimization
    combined_stock_data = pd.concat([data.set_index('Date')['Close'].rename(ticker) for ticker, data in stocks_data.items() if data is not None], axis=1).dropna()
    
    logging.info("Optimizing portfolio")
    weights = optimize_portfolio(combined_stock_data, num_stocks=num_stocks)

    if weights is not None:
        selected_stocks = combined_stock_data.columns[np.argsort(weights)[-num_stocks:]]
        selected_weights = weights[np.argsort(weights)[-num_stocks:]]
        
        # Normalize the selected weights
        normalized_weights = selected_weights / np.sum(selected_weights)
        
        portfolio = {stock_names[stock]: weight for stock, weight in zip(selected_stocks, normalized_weights)}
        logging.info(f"Optimized portfolio: {portfolio}")

        # Format portfolio weights to percentages and include names
        portfolio_percentages = {f"{stock_names[stock]} ({stock})": f"{weight * 100:.2f}%" for stock, weight in zip(selected_stocks, normalized_weights)}
        print("Optimized portfolio (weights as percentages):", dict(sorted(portfolio_percentages.items(), key=lambda item: float(item[1][:-1]), reverse=True)))

        # Calculate internal correlation score
        selected_data = combined_stock_data[selected_stocks]
        corr_matrix = selected_data.pct_change().dropna().corr()
        internal_correlation = calculate_internal_correlation(corr_matrix, normalized_weights)
        logging.info(f"Internal correlation score: {internal_correlation:.4f}")
        print(f"Internal correlation score: {internal_correlation:.4f}")

        # Calculate correlations with commodities
        portfolio_data = selected_data.dot(normalized_weights).reset_index().rename(columns={0: 'Portfolio'})
        portfolio_data = portfolio_data.rename(columns={portfolio_data.columns[1]: 'Close'})
        commodity_correlations = calculate_correlations(portfolio_data, commodities_data)
        formatted_correlations = {f"{commodities_names[commodity]} ({commodity})": f"{corr:.4f}" for commodity, corr in commodity_correlations.items()}
        print("Commodity correlations:", dict(sorted(formatted_correlations.items(), key=lambda item: float(item[1]), reverse=True)))

        # Calculate correlations with CBOE indices
        cboe_correlations = calculate_cboe_correlations(portfolio_data, cboe_data)
        formatted_cboe_correlations = {f"{name} ({CBOE_INDICES[name]})": f"{corr:.4f}" for name, corr in cboe_correlations.items()}
        print("CBOE index correlations:", dict(sorted(formatted_cboe_correlations.items(), key=lambda item: float(item[1]), reverse=True)))

        # Backtest the portfolio for 1 year
        portfolio_returns, cumulative_returns = backtest_portfolio(selected_data, normalized_weights)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
        logging.info(f"Sharpe ratio: {sharpe_ratio:.4f}")
        print(f"Sharpe ratio: {sharpe_ratio:.4f}")

        # Plot cumulative returns
        plt.figure(figsize=(10, 6))
        cumulative_returns.plot(title='Portfolio Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.show()
        
    else:
        logging.error("Failed to optimize portfolio")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and analyze stock and commodity data.')
    parser.add_argument('api_key', help='API key for financialmodelingprep.com')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of stock tickers')
    parser.add_argument('--etf', type=str, default='SPY', help='ETF symbol to fetch holdings (default: SPY)')
    parser.add_argument('--num_stocks', type=int, default=10, help='Number of stocks in the optimized portfolio (default: 10)')

    args = parser.parse_args()
    stock_tickers = args.tickers.split(',') if args.tickers else None
    
    if not args.api_key:
        logging.error("API key is required")
    
    if not stock_tickers and not args.etf:
        logging.error("No stock tickers or ETF symbol provided")
    
    if args.tickers:
        print(f"Fetching data for stocks: {stock_tickers}")
    else:
        print(f"Fetching data for positions within ETF: {args.etf}")
    
    main(args.api_key, stock_tickers, args.etf, args.num_stocks)

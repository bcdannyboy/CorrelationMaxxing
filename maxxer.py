import requests
import yfinance as yf
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import minimize
import argparse

logging.basicConfig(level=logging.ERROR)

# Reduce logging for yfinance and urllib3
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

def fetch_commodity_data(api_key, symbol):
    logging.info(f"Fetching commodity data for {symbol} from Financial Modeling Prep API")
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch data for {symbol}, Status code: {response.status_code}")
        return None

def fetch_stock_data(ticker):
    logging.info(f"Fetching stock data for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
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
        return [holding['symbol'] for holding in holdings]
    else:
        logging.error(f"Failed to fetch ETF holdings for {etf_symbol}, Status code: {response.status_code}")
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

def main(api_key, stock_tickers=None, etf_symbol=None, num_stocks=10):
    logging.info("Starting main function")
    
    # If ETF symbol is provided, fetch its holdings
    if etf_symbol:
        holdings = fetch_etf_holdings(api_key, etf_symbol)
        if holdings:
            stock_tickers = holdings
        else:
            logging.error(f"Failed to fetch holdings for ETF {etf_symbol}")
            return

    if not stock_tickers:
        logging.error("No stock tickers provided")
        return

    # Fetch commodities list
    commodities_url = f"https://financialmodelingprep.com/api/v3/symbol/available-commodities?apikey={api_key}"
    commodities_response = requests.get(commodities_url)
    if commodities_response.status_code != 200:
        logging.error("Failed to fetch commodities list")
        return

    commodities = commodities_response.json()
    
    # Fetch data for all stocks and commodities concurrently
    with ThreadPoolExecutor(max_workers=20) as executor:
        stock_futures = {executor.submit(fetch_stock_data, ticker.strip()): ticker.strip() for ticker in stock_tickers}
        commodity_futures = {executor.submit(fetch_commodity_data, api_key, commodity['symbol']): commodity['symbol'] for commodity in commodities}
        
        stocks_data = {ticker: future.result() for future, ticker in stock_futures.items() if future.result() is not None}
        commodities_data = {symbol: future.result() for future, symbol in commodity_futures.items() if future.result() is not None}
    
    # Prepare stock data for optimization
    combined_stock_data = pd.concat([data.set_index('Date')['Close'].rename(ticker) for ticker, data in stocks_data.items() if data is not None], axis=1).dropna()
    
    logging.info("Optimizing portfolio")
    weights = optimize_portfolio(combined_stock_data, num_stocks=num_stocks)

    if weights is not None:
        selected_stocks = combined_stock_data.columns[np.argsort(weights)[-num_stocks:]]
        selected_weights = weights[np.argsort(weights)[-num_stocks:]]
        
        # Normalize the selected weights
        normalized_weights = selected_weights / np.sum(selected_weights)
        
        portfolio = {stock: weight for stock, weight in zip(selected_stocks, normalized_weights)}
        logging.info(f"Optimized portfolio: {portfolio}")

        # Format portfolio weights to percentages
        portfolio_percentages = {stock: f"{weight * 100:.2f}%" for stock, weight in portfolio.items()}
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
        formatted_correlations = {commodity: f"{corr:.4f}" for commodity, corr in commodity_correlations.items()}
        print("Commodity correlations:", dict(sorted(formatted_correlations.items(), key=lambda item: float(item[1]), reverse=True)))

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

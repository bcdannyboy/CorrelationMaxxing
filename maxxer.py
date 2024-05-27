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
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch data for {symbol}, Status code: {response.status_code}")
        return None

def fetch_stock_data(ticker, period="5y"):
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
    url = f"https://financialmodelingprep.com/api/v4/etf-holdings?symbol={etf_symbol}&date={date}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        holdings = response.json()
        return [(holding['symbol'], holding['name']) for holding in holdings]
    else:
        logging.error(f"Failed to fetch ETF holdings for {etf_symbol}, Status code: {response.status_code}")
        return None

def fetch_commodities_list(api_key):
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
        if 'Date' in data.columns and 'Close' in data.columns:
            combined = pd.merge(stock_data, data[['Date', 'Close']], on='Date', suffixes=('', f'_{name}'))
            if not combined.empty and 'Close' in combined.columns and f'Close_{name}' in combined.columns:
                corr = combined['Close'].corr(combined[f'Close_{name}'])
                if pd.notnull(corr):
                    correlations[name] = corr
    return correlations

def optimize_portfolio_mvo(stock_data, num_stocks=10):
    # Calculate daily returns
    returns = stock_data.pct_change().dropna()

    # Calculate the mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def objective(weights):
        # Portfolio variance
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        # Portfolio return
        portfolio_return = np.dot(weights, mean_returns)
        # Objective: Minimize variance and maximize return (sharpe ratio)
        return -portfolio_return / np.sqrt(portfolio_variance)

    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(mean_returns)))
    initial_weights = np.array([1 / len(mean_returns)] * len(mean_returns))

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        indices = np.argsort(result.x)[-num_stocks:]
        weights = result.x[indices]
        return indices, weights / np.sum(weights)  # Normalize the weights
    else:
        logging.error("Optimization failed")
        return None, None

def optimize_portfolio_min_corr(stock_data, num_stocks=10):
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
        indices = np.argsort(result.x)[-num_stocks:]
        weights = result.x[indices]
        return indices, weights / np.sum(weights)  # Normalize the weights
    else:
        logging.error("Optimization failed")
        return None, None

def calculate_internal_correlation(corr_matrix, weights):
    return np.dot(weights.T, np.dot(corr_matrix, weights))

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.0446):
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
    if (etf_symbol):
        holdings = fetch_etf_holdings(api_key, etf_symbol)
        if (holdings):
            stock_tickers = [ticker for ticker, name in holdings]
            stock_names = {ticker: name for ticker, name in holdings}
        else:
            logging.error(f"Failed to fetch holdings for ETF {etf_symbol}")
            return
    else:
        stock_names = {ticker: ticker for ticker in stock_tickers}

    if (not stock_tickers):
        logging.error("No stock tickers provided")
        return

    # Fetch commodities list and their names
    commodities_names = fetch_commodities_list(api_key)
    if (not commodities_names):
        logging.error("No commodities fetched")
        return
    
    # Fetch data for all stocks and commodities concurrently
    with ThreadPoolExecutor(max_workers=20) as executor:
        stock_futures = {executor.submit(fetch_stock_data, ticker.strip(), period="5y"): ticker.strip() for ticker in stock_tickers}
        commodity_futures = {executor.submit(fetch_commodity_data, api_key, commodity): commodity for commodity in commodities_names.keys()}
        
        stocks_data = {ticker: future.result() for future, ticker in stock_futures.items() if future.result() is not None}
        commodities_data = {symbol: future.result() for future, symbol in commodity_futures.items() if future.result() is not None}
    
    # Filter out tickers with missing data
    valid_tickers = [ticker for ticker, data in stocks_data.items() if data is not None]
    combined_stock_data = pd.concat([stocks_data[ticker].set_index('Date')['Close'].rename(ticker) for ticker in valid_tickers], axis=1).dropna()

    if (combined_stock_data.empty):
        logging.error("No valid stock data available after filtering")
        return
    
    # Fetch CBOE index data
    cboe_data = fetch_cboe_index_data()
    
    # Calculate correlations
    commodity_correlations = calculate_correlations(combined_stock_data, commodities_data)
    cboe_correlations = calculate_cboe_correlations(combined_stock_data, cboe_data)
    
    # Optimize portfolios using both strategies
    logging.info("Optimizing portfolio using Mean-Variance Optimization")
    indices_mvo, weights_mvo = optimize_portfolio_mvo(combined_stock_data, num_stocks=num_stocks)
    
    logging.info("Optimizing portfolio using minimum correlation")
    indices_min_corr, weights_min_corr = optimize_portfolio_min_corr(combined_stock_data, num_stocks=num_stocks)

    if weights_mvo is not None:
        selected_stocks_mvo = combined_stock_data.columns[indices_mvo]
        
        portfolio_mvo = {stock_names[stock]: weight for stock, weight in zip(selected_stocks_mvo, weights_mvo)}
        logging.info(f"Optimized portfolio (MVO): {portfolio_mvo}")

        # Format portfolio weights to percentages and include names
        portfolio_percentages_mvo = {f"{stock_names[stock]} ({stock})": f"{weight * 100:.2f}%" for stock, weight in zip(selected_stocks_mvo, weights_mvo)}
        print("Optimized portfolio (MVO) (weights as percentages):", dict(sorted(portfolio_percentages_mvo.items(), key=lambda item: float(item[1][:-1]), reverse=True)))

        # Calculate internal correlation score
        selected_data_mvo = combined_stock_data[selected_stocks_mvo]
        corr_matrix_mvo = selected_data_mvo.pct_change().dropna().corr()
        internal_correlation_mvo = calculate_internal_correlation(corr_matrix_mvo, weights_mvo)
        logging.info(f"Internal correlation score (MVO): {internal_correlation_mvo:.4f}")
        print(f"Internal correlation score (MVO): {internal_correlation_mvo:.4f}")

        # Backtest the portfolio for 1 year
        portfolio_returns_mvo, cumulative_returns_mvo = backtest_portfolio(selected_data_mvo, weights_mvo)
        sharpe_ratio_mvo = calculate_sharpe_ratio(portfolio_returns_mvo)
        logging.info(f"Sharpe ratio (MVO): {sharpe_ratio_mvo:.4f}")
        print(f"Sharpe ratio (MVO): {sharpe_ratio_mvo:.4f}")

    if weights_min_corr is not None:
        selected_stocks_min_corr = combined_stock_data.columns[indices_min_corr]
        
        portfolio_min_corr = {stock_names[stock]: weight for stock, weight in zip(selected_stocks_min_corr, weights_min_corr)}
        logging.info(f"Optimized portfolio (Min Correlation): {portfolio_min_corr}")

        # Format portfolio weights to percentages and include names
        portfolio_percentages_min_corr = {f"{stock_names[stock]} ({stock})": f"{weight * 100:.2f}%" for stock, weight in zip(selected_stocks_min_corr, weights_min_corr)}
        print("Optimized portfolio (Min Correlation) (weights as percentages):", dict(sorted(portfolio_percentages_min_corr.items(), key=lambda item: float(item[1][:-1]), reverse=True)))

        # Calculate internal correlation score
        selected_data_min_corr = combined_stock_data[selected_stocks_min_corr]
        corr_matrix_min_corr = selected_data_min_corr.pct_change().dropna().corr()
        internal_correlation_min_corr = calculate_internal_correlation(corr_matrix_min_corr, weights_min_corr)
        logging.info(f"Internal correlation score (Min Correlation): {internal_correlation_min_corr:.4f}")
        print(f"Internal correlation score (Min Correlation): {internal_correlation_min_corr:.4f}")

        # Backtest the portfolio for 1 year
        portfolio_returns_min_corr, cumulative_returns_min_corr = backtest_portfolio(selected_data_min_corr, weights_min_corr)
        sharpe_ratio_min_corr = calculate_sharpe_ratio(portfolio_returns_min_corr)
        logging.info(f"Sharpe ratio (Min Correlation): {sharpe_ratio_min_corr:.4f}")
        print(f"Sharpe ratio (Min Correlation): {sharpe_ratio_min_corr:.4f}")

    # Plot cumulative returns for both strategies
    plt.figure(figsize=(10, 6))
    if weights_mvo is not None:
        cumulative_returns_mvo.plot(label='MVO Strategy')
    if weights_min_corr is not None:
        cumulative_returns_min_corr.plot(label='Min Correlation Strategy')
    plt.title('Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()

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

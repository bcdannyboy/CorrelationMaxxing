# CorrelationMaxxing

**Identify a portfolio that maximizes correlation to commodities while minimizing inter-portfolio correlation**

## Description

`CorrelationMaxxing` is a Python script that fetches historical price data for stocks and commodities to construct an optimized portfolio. The goal is to maximize the correlation of the portfolio with commodities while minimizing the internal correlation within the portfolio. The script can either take a list of stock tickers or use the holdings of a provided ETF (SPY by default).

## Features

- Fetch historical price data for stocks, commodities, and indices.
- Optimize a portfolio to balance correlation to commodities/indices and inter-stock correlation using Mean-Variance Optimization (MVO) or minimization of internal correlation.
- Option to provide individual stock tickers or use holdings from an ETF.
- Outputs the optimized portfolio weights, internal correlation score, and commodity correlations.

## Requirements

- Python 3.6+
- [A Financial Modeling Prep API Key](https://site.financialmodelingprep.com/)
- `requests`
- `yfinance`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/bcdannyboy/CorrelationMaxxing.git
   cd CorrelationMaxxing
   ```

2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Command Line Arguments

- `api_key`: API key for `financialmodelingprep.com`.
- `--tickers`: (Optional) Comma-separated list of stock tickers.
- `--etf`: (Optional) ETF symbol to fetch holdings (default: SPY).
- `--num_stocks`: (Optional) Number of stocks in the optimized portfolio (default: 10).

### Examples

1. **Using a list of stock tickers:**
   ```sh
   python3 maxxer.py your_api_key --tickers "AAPL, MSFT, AMZN, GOOG, FB, TSLA, NVDA, JPM, JNJ, V"
   ```

2. **Using holdings from an ETF (SPY by default):**
   ```sh
   python3 maxxer.py your_api_key
   ```

3. **Using a different ETF:**
   ```sh
   python3 maxxer.py your_api_key --etf QQQ
   ```

4. **Specifying the number of stocks in the optimized portfolio:**
   ```sh
   python3 maxxer.py your_api_key --num_stocks 15
   ```

## Output

The script provides the following outputs:

- **Optimized Portfolio Weights:** List of stocks with their corresponding weights in the optimized portfolio.
- **Internal Correlation Score:** A score representing the internal correlation of the portfolio.
- **Commodity Correlations:** Correlation values of the portfolio with various commodities.
- **Sharpe Ratio:** The Sharpe ratio of the optimized portfolio.

## Example Output

```sh
Fetching data for positions within ETF: SPY
Optimized portfolio (MVO) (weights as percentages): {'AAPL': '12.34%', 'MSFT': '11.56%', 'GOOG': '10.78%', ...}
Internal correlation score (MVO): 0.4230
Sharpe ratio (MVO): 1.2345
Commodity correlations: {'Gold': '0.1234', 'Oil': '0.2345', ...}
```

```sh
Fetching data for positions within ETF: SPY
Optimized portfolio (Min Correlation) (weights as percentages): {'AAPL': '12.34%', 'MSFT': '11.56%', 'GOOG': '10.78%', ...}
Internal correlation score (Min Correlation): 0.4230
Sharpe ratio (Min Correlation): 1.2345
Commodity correlations: {'Gold': '0.1234', 'Oil': '0.2345', ...}
```
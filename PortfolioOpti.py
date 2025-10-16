import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from scipy.optimize import minimize

st.set_page_config(page_title="Live Trading Analytics", layout="wide")

st.title("Real-Time Trading Analytics App")

tickers = st.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT, NVDA")
risk_free_rate = st.text_input("Enter risk free rate", "0.00")
period = st.selectbox("Select time period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])

if st.button("Fetch Data"):
    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    # --- PRICE DATA ---
    data = yf.download(ticker_list, period=period, auto_adjust=True)['Close']

    # --- PRICE CHART ---
    st.subheader("Price Chart")
    fig = go.Figure()
    for t in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[t], mode='lines', name=t))
    st.plotly_chart(fig, use_container_width=True)

    # --- PERFORMANCE METRICS ---
    st.subheader("Performance Metrics")
    returns = data.pct_change().dropna()
    daily_rf_rate = float(risk_free_rate) / 252
    sharpe_ratios = (((returns.mean() - daily_rf_rate) * 252) / (returns.std() * np.sqrt(252))).sort_values(ascending=False)
    st.write("**Annualized Sharpe Ratios:**")
    st.dataframe(sharpe_ratios.round(3))

    # --- CORRELATION MATRIX ---
    st.subheader("Correlation Matrix")
    corr = returns.corr()
    st.dataframe(corr)

    st.subheader("Heatmap")
    st.plotly_chart(
        go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            )
        ),
        use_container_width=True
    )

    # --- VALUATION METRICS ---
    st.subheader("Valuation Metrics (Fundamentals)")
    ratios = []
    for t in ticker_list:
        try:
            info = yf.Ticker(t).info
            ratios.append({
                "Ticker": t,
                "Current Price": info.get("currentPrice", np.nan),
                "P/E (Trailing)": info.get("trailingPE", np.nan),
                "P/S (TTM)": info.get("priceToSalesTrailing12Months", np.nan),
                "Market Cap": info.get("marketCap", np.nan)
            })
        except Exception as e:
            st.warning(f"Could not fetch fundamentals for {t}: {e}")

    df_ratios = pd.DataFrame(ratios)
    df_ratios.set_index("Ticker", inplace=True)
    st.dataframe(df_ratios.style.format("{:.2f}"))

    st.caption("ℹ️ P/E = Price/Earnings, P/S = Price/Sales. Data from Yahoo Finance (yfinance).")

    # --- PORTFOLIO RISK ANALYSIS ---
    st.subheader("Portfolio Risk Analysis")

    # Annualized volatilities
    volatilities = returns.std() * np.sqrt(252)
    st.write("**Annualized Volatility (per stock):**")
    st.dataframe(volatilities.round(3))

    # Covariance matrix (annualized)
    cov_matrix = returns.cov() * 252

    # Equal weights (for comparison)
    n_assets = len(ticker_list)
    equal_weights = np.ones(n_assets) / n_assets

    # Portfolio variance and risk
    port_variance = np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights))
    port_volatility = np.sqrt(port_variance)

    st.write(f"**Portfolio Volatility (Equal Weights):** {port_volatility:.2%}")

    # Compare to sum of individual risks (no diversification)
    sum_risks = np.sum(equal_weights * volatilities)
    diversification_effect = sum_risks - port_volatility

    st.write(f"**Sum of Individual Risks:** {sum_risks:.2%}")
    st.write(f"**Diversification Benefit (Risk Reduction):** {diversification_effect:.2%}")

    if diversification_effect > 0:
        st.success("Your portfolio benefits from diversification — total risk is lower than sum of individual risks.")
    else:
        st.warning("Little or no diversification benefit — assets may be highly correlated.")

    # --- CORRELATION RANKING ---
    st.subheader("🔗 Asset Correlation Ranking")
    corr_mean = corr.mean().sort_values(ascending=False)
    st.write("**Average Correlation per Asset (highest to lowest):**")
    st.dataframe(corr_mean.round(3))

    most_corr = corr_mean.idxmax()
    least_corr = corr_mean.idxmin()
    st.write(f"**Most correlated asset overall:** {most_corr}")
    st.write(f"**Least correlated asset overall:** {least_corr}")

    # --- OPTIMAL PORTFOLIO WEIGHTS (MPT) ---
    st.subheader("Optimal Portfolio Weights (Markowitz Optimization)")

    mean_returns = returns.mean() * 252
    rf = float(risk_free_rate)

    def portfolio_performance(weights, mean_returns, cov_matrix, rf):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - rf) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
        return -portfolio_performance(weights, mean_returns, cov_matrix, rf)[2]

    # Constraints: sum(weights) = 1, weights >= 0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array(n_assets * [1. / n_assets])

    # Optimize for maximum Sharpe ratio
    result = minimize(neg_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix, rf),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x

    # Display results
    opt_return, opt_vol, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix, rf)

    st.write("**Optimal Weights (Max Sharpe Ratio):**")
    opt_df = pd.DataFrame({'Ticker': ticker_list, 'Weight': optimal_weights})
    st.dataframe(opt_df.set_index("Ticker").style.format("{:.2%}"))

    st.write(f"**Expected Annual Return:** {opt_return:.2%}")
    st.write(f"**Expected Volatility:** {opt_vol:.2%}")
    st.write(f"**Sharpe Ratio:** {opt_sharpe:.2f}")

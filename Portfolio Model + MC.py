import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize
from arch import arch_model  # For GARCH modeling
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from joblib import Parallel, delayed  # For parallel processing
from scipy.stats import t

# Define constants
ANNUALIZATION_FACTOR = 252

# User inputs
initial_portfolio_value = float(input("Enter initial portfolio value (default: 10000): ") or 10000)
mc_sims = int(input("Enter number of Monte Carlo simulations (default: 100): ") or 100)
T = int(input("Enter investment horizon in years (default: 10): ") or 10)  # Number of years for investment horizon
risk_free_rate = float(input("Enter risk-free rate (e.g., 0.02 for 2%, default: 0.02): ") or 0.02)
tickers = input("Enter tickers separated by commas (e.g., 'SPY,AGG,GLD', default: 'SPY,AGG,GLD'): ") or 'SPY,AGG,GLD'
tickers = tickers.split(',')
benchmarks = input("Enter benchmark tickers separated by commas (e.g., 'SPY,DIA,QQQ', default: 'SPY'): ") or 'SPY'
benchmarks = benchmarks.split(',')

# Download data for multiple assets
try:
    data = yf.download(tickers, period="10y")['Adj Close']
    if data.isnull().values.any():
        raise ValueError("Downloaded data contains missing values. Please check the data or use different tickers.")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# Calculate returns
returns = data.pct_change().dropna()

# Fit GARCH models for volatility estimation
def fit_garch(ticker):
    model = arch_model(returns[ticker] * 100, vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')
    return garch_fit.conditional_volatility / 100

# Get GARCH volatility estimates in parallel
garch_vols = Parallel(n_jobs=-1)(delayed(fit_garch)(ticker) for ticker in tickers)
volatility_df = pd.concat(garch_vols, axis=1)
volatility_df.columns = tickers
cov_matrix_garch = volatility_df.cov() * ANNUALIZATION_FACTOR

# Calculate mean returns and covariance matrix using GARCH volatility
mean_returns = returns.mean()
cov_matrix = cov_matrix_garch

# Function to calculate portfolio return and volatility
def portfolio_performance(weights, mean_returns, cov_matrix):
    annual_return = mean_returns.dot(weights) * ANNUALIZATION_FACTOR
    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return annual_return, annual_volatility

# Risk-adjusted return: Sharpe Ratio calculation
def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return (p_return - risk_free_rate) / p_volatility

# Portfolio optimization function (maximizing the Sharpe Ratio)
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum to 1
    bounds = tuple((0, 1) for asset in range(num_assets))  # Weights between 0 and 1
    initial_guess = num_assets * [1. / num_assets]  # Equal allocation as initial guess
    
    result = minimize(lambda x: -sharpe_ratio(x, *args),
                      initial_guess,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'disp': False, 'maxiter': 1000})  # Improved optimization options
    return result

# Optimize portfolio
optimized_result = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)

# Check if optimization was successful
if not optimized_result.success:
    print(f"Optimization failed: {optimized_result.message}")
    exit()

optimized_weights = optimized_result.x

# Calculate optimized portfolio metrics
opt_return, opt_volatility = portfolio_performance(optimized_weights, mean_returns, cov_matrix)
opt_sharpe = sharpe_ratio(optimized_weights, mean_returns, cov_matrix, risk_free_rate)

print("\n--- Portfolio Optimization Results ---")
print(f"Optimized Portfolio Weights: {np.array2string(optimized_weights, precision=2, separator=', ')}")
weights_percentages = [round(w * 100) for w in optimized_weights]
print(f"Portfolio Allocation as Percentages: {', '.join(map(str, weights_percentages))}%")
print(f"Expected Annual Return: {opt_return:.2%}")
print(f"Annual Volatility: {opt_volatility:.2%}")
print(f"Sharpe Ratio: {opt_sharpe:.2f}")

# Monte Carlo simulation function
def monte_carlo_simulation(weights, mean_returns, cov_matrix, mc_sims, T, initial_value):
    df = 5  # Degrees of freedom for Student's t distribution
    n_assets = len(mean_returns)

    # Generate random samples for each asset
    standard_t_samples = t.rvs(df, size=(T, mc_sims, n_assets))  # Shape (T, mc_sims, n_assets)
    
    # Expand mean returns and volatility for broadcasting
    mean_returns = mean_returns.values[np.newaxis, np.newaxis, :]  # Shape (1, 1, n_assets)
    volatility = np.sqrt(np.diag(cov_matrix * ANNUALIZATION_FACTOR))[np.newaxis, np.newaxis, :] # Shape (1, 1, n_assets)

    # Simulated returns for each asset
    simulated_returns = mean_returns * ANNUALIZATION_FACTOR + standard_t_samples * volatility  # Shape (T, mc_sims, n_assets)
    
    # Portfolio log returns
    portfolio_returns = np.dot(simulated_returns, weights)  # Shape (T, mc_sims)
    log_returns = np.log1p(portfolio_returns)
    
    # Cumulative log returns and portfolio valu
    cumulative_log_returns = np.cumsum(log_returns, axis=0)
    cumulative_log_returns = np.clip(cumulative_log_returns, a_min=None, a_max=700)  # Clipping to prevent overflow
    sim_results = initial_value * np.exp(cumulative_log_returns)

    return sim_results

# Run Monte Carlo simulation
# Note: VaR (Value at Risk) is a commonly used risk metric to estimate the maximum potential loss over a given time period with a certain confidence level.
# Typically, VaR is calculated at 1%, 5%, or 10%. Here, we are using 10% (confidence level) to assess the most extreme scenarios.
# CVaR (Conditional Value at Risk) represents the expected loss in the worst-case scenarios beyond the VaR threshold (aka all the scenarios belonging to the worst 10%).
final_portfolio_values = monte_carlo_simulation(optimized_weights, mean_returns, cov_matrix, mc_sims, T, initial_portfolio_value)
mean_final_value = np.mean(final_portfolio_values[-1, :])
var_90 = np.percentile(final_portfolio_values[-1, :], 10) # 10th percentile representing the worst 10% VaR scenarios
cvar_90 = np.mean(final_portfolio_values[-1, :][final_portfolio_values[-1, :] <= var_90])

print("\n--- Monte Carlo Simulation Results ---")
print(f"Expected Portfolio Value (after {T} years): ${mean_final_value:,.2f}")
print(f"10% Value at Risk (VaR): ${var_90:,.2f}")
print(f"10% Conditional Value at Risk (CVaR): ${cvar_90:,.2f}")

# Maximum Drawdown calculation
def max_drawdown(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    return np.max(drawdown)

max_dd = max_drawdown(final_portfolio_values[-1, :])

print("\n--- Risk Metrics ---")
print(f"Maximum Drawdown: {max_dd:.2%}")

# GARCH model for time-varying volatility
# Parallelize GARCH fitting for multiple tickers
def fit_garch(ticker):
    model = arch_model(returns[ticker] * 100, vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')
    return f"{ticker} GARCH(1,1) Model Summary:\n{garch_fit.summary()}\n"

results = Parallel(n_jobs=-1)(delayed(fit_garch)(ticker) for ticker in tickers)
for result in results:
    print(result)

# Enhanced Monte Carlo Simulation Chart
import matplotlib.table as tbl
fig, ax = plt.subplots(figsize=(12, 7))
for i in range(min(mc_sims, 100)):  # Plot a subset for clarity
    ax.plot(np.arange(1, T + 1), final_portfolio_values[:, i], color='lightblue', alpha=0.1)

# Calculate key statistics for Monte Carlo results
best_result = np.max(final_portfolio_values[-1, :])
worst_result = np.min(final_portfolio_values[-1, :])
median_result = np.median(final_portfolio_values[-1, :])
mode_result = pd.Series(final_portfolio_values[-1, :]).mode()[0]
mean_result = np.mean(final_portfolio_values[-1, :])

# Function to plot key trajectories
def plot_key_trajectory(ax, x_values, y_values, color, label, linestyle='-', linewidth=2):
    ax.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

# Plot key lines with distinct styles
plot_key_trajectory(ax, np.arange(1, T + 1), np.full(T, best_result), color='green', label=f'Best Result: ${best_result:,.2f}')
plot_key_trajectory(ax, np.arange(1, T + 1), np.full(T, worst_result), color='red', label=f'Worst Result: ${worst_result:,.2f}')
plot_key_trajectory(ax, np.arange(1, T + 1), np.full(T, median_result), color='purple', label=f'Median Result: ${median_result:,.2f}', linestyle='--')
plot_key_trajectory(ax, np.arange(1, T + 1), np.full(T, mean_result), color='blue', label=f'Mean Result: ${mean_result:,.2f}', linestyle='--')
plot_key_trajectory(ax, np.arange(1, T + 1), np.full(T, mode_result), color='orange', label=f'Mode Result: ${mode_result:,.2f}', linestyle='-.')

# Highlight best, worst, median, mode, and mean trajectories
best_trajectory = final_portfolio_values[:, np.argmax(final_portfolio_values[-1, :])]
worst_trajectory = final_portfolio_values[:, np.argmin(final_portfolio_values[-1, :])]
median_trajectory = final_portfolio_values[:, np.argsort(final_portfolio_values[-1, :])[mc_sims // 2]]

# Plot highlighted trajectories
plot_key_trajectory(plt, np.arange(1, T + 1), best_trajectory, color='green', label=f'Best Result: ${best_result:,.2f}')
plot_key_trajectory(plt, np.arange(1, T + 1), worst_trajectory, color='red', label=f'Worst Result: ${worst_result:,.2f}')
plot_key_trajectory(plt, np.arange(1, T + 1), median_trajectory, color='purple', label=f'Median Result: ${median_result:,.2f}')
plt.axhline(y=mean_result, color='blue', linestyle='--', linewidth=2, label=f'Mean Result: ${mean_result:,.2f}')
plt.title('Monte Carlo Simulation of Portfolio Growth')

# Assuming `best_result`, `worst_result`, `median_result`, `mode_result`, and `mean_result` are defined
data = [
    [f'${best_result:,.2f}'], 
    [f'${worst_result:,.2f}'], 
    [f'${median_result:,.2f}'], 
    [f'${mode_result:,.2f}'], 
    [f'${mean_result:,.2f}']
]
row_labels = ['Best', 'Worst', 'Median', 'Mode', 'Mean']
col_labels = ['Value']

# Create table
table = ax.table(
    cellText=data,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc='upper right',
    bbox=[0.7, 0.5, 0.25, 0.3]  # Adjusted positioning
)

# Styling the table
table.auto_set_font_size(False)
table.set_fontsize(9)  # Reduced for compactness
for key, cell in table.get_celld().items():
    cell.set_alpha(0.8)  # Add transparency to make it lighter
    cell.set_edgecolor('lightgrey')  # Subtle border for better visibility

# Add chart details
plt.xlabel('Years', fontsize=9)
plt.ylabel('Portfolio Value', fontsize=9)
plt.title('Portfolio Value Over Time', fontsize=10)
plt.legend(loc='best', fontsize=8)  # Ensure legend does not overlap the table
plt.grid(alpha=0.3)  # Lighter grid lines for less visual clutter
plt.tight_layout()  # Adjust layout to ensure proper spacing
plt.show()

# Interactive Visualization with Plotly
fig = go.Figure()
for i in range(min(mc_sims, 100)):  # Limit the number of trajectories displayed
    fig.add_trace(go.Scatter(
        x=np.arange(1, T + 1),
        y=final_portfolio_values[:, i],
        mode='lines',
        line=dict(color='lightblue', width=0.5),
        showlegend=False
    ))

# Add key metrics as separate lines
fig.add_trace(go.Scatter(
    x=np.arange(1, T + 1),
    y=np.full(T, mean_final_value),
    mode='lines',
    name='Mean Final Value',
    line=dict(color='blue', dash='dash')
))
fig.add_trace(go.Scatter(
    x=np.arange(1, T + 1),
    y=np.full(T, var_90),
    mode='lines',
    name='10% Value at Risk',
    line=dict(color='red', dash='dash')
))
fig.add_trace(go.Scatter(
    x=np.arange(1, T + 1),
    y=np.full(T, cvar_90),
    mode='lines',
    name='10% Conditional VaR',
    line=dict(color='orange', dash='dash')
))

fig.update_layout(
    title='Monte Carlo Simulation: Portfolio Growth',
    xaxis_title='Years',
    yaxis_title='Portfolio Value',
    template='plotly_white',
    legend=dict(x=0.02, y=0.98)
)
fig.show()

# Setting up Dash app for interactive visualization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Portfolio Optimization Dashboard", className="text-center mt-4"))),
    dbc.Row([
        dbc.Col(dcc.Graph(
            figure={
                'data': [
                    go.Histogram(
                        x=final_portfolio_values[-1, :],
                        nbinsx=50,
                        marker_color='skyblue',
                        name='Final Portfolio Values'
                    ),
                    go.Scatter(
                        x=[var_90],
                        y=[0],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name=f'10% VaR: ${var_90:.2f}'
                    )
                ],
                'layout': go.Layout(
                    title='Distribution of Final Portfolio Values',
                      yaxis=dict(
                        title='Portfolio Value',
                        tickformat='$,.0f',  # Format as currency
                        tickvals=np.arange(min(final_portfolio_values[-1, :]),
                                           max(final_portfolio_values[-1, :]), 
                                           1000000)  # Ticks every 1M
                    ),
                    xaxis=dict(
                        title='Frequency',
                        tickmode='auto',  # Automatically determine tick intervals
                    ),
                    showlegend=True,
                    gridcolor='lightgray'  # Light gridlines for readability
                )
            }
        ), width=12)
    ]),
    
    # Line Chart of Portfolio Growth
    dbc.Row([
        dbc.Col(dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=np.arange(1, T + 1),
                        y=np.mean(final_portfolio_values, axis=1),
                        mode='lines',
                        name='Optimized Portfolio'
                    )
                ],
                'layout': go.Layout(
                    title=f'{T}-Year Optimized Portfolio Growth',
                    xaxis=dict(
                        title='Years',
                        tickvals=np.arange(0, T + 1, 3),  # Ticks every 3 years
                        tickmode='array'  # Explicit tick values
                    ),
                    yaxis=dict(
                        title='Portfolio Value',
                        tickformat='$,.0f',  # Format as currency
                        gridcolor='lightgray',
                        gridwidth=0.5  # Thin gridlines
                    ),
                    showlegend=True
                )
            }
        ), width=12)
    ])
])

if __name__ == '__main__':
    # Configuration Section for modifying parameters
    print("\n--- Configuration Section ---")
    initial_portfolio_value = float(input("Modify initial portfolio value (current: 10000): ") or initial_portfolio_value)
    mc_sims = int(input("Modify number of Monte Carlo simulations (current: 1000): ") or mc_sims)
    T = int(input("Modify investment horizon in years (current: 10): ") or T)
    risk_free_rate = float(input("Modify risk-free rate (current: 0.02): ") or risk_free_rate)
    tickers = input("Modify tickers (current: 'SPY,AGG,GLD'): ") or ','.join(tickers)
    tickers = tickers.split(',')
    benchmarks = input("Modify benchmark tickers (current: 'SPY'): ") or ','.join(benchmarks)
    benchmarks = benchmarks.split(',')
    
    app.run_server(debug=True)

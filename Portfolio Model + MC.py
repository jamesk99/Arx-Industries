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
    data.dropna(axis=1, how='all', inplace=True)  # Drop tickers with no valid data
    data.fillna(method='ffill', inplace=True)  # Fill missing data with forward fill
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
garch_vols = Parallel(n_jobs=4)(delayed(fit_garch)(ticker) for ticker in tickers)
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
    df = 10  # Adjusted degrees of freedom for Student's t distribution
    n_assets = len(mean_returns)
    
    # Generate random samples for each asset
    standard_t_samples = t.rvs(df, size=(T * ANNUALIZATION_FACTOR, mc_sims, n_assets))  # Adjusted shape for daily simulations
    
    # Expand mean returns and volatility for broadcasting
    mean_returns = mean_returns.values[np.newaxis, np.newaxis, :]  # Shape (1, 1, n_assets)
    volatility = np.sqrt(np.diag(cov_matrix))[np.newaxis, np.newaxis, :]  # Removed annualization factor to avoid overestimation

    # Simulated returns for each asset
    simulated_returns = mean_returns * (1 / ANNUALIZATION_FACTOR) + standard_t_samples * volatility  # Daily returns
    
    # Portfolio log returns
    portfolio_returns = np.dot(simulated_returns, weights)  # Shape (T * ANNUALIZATION_FACTOR, mc_sims)
    log_returns = np.log1p(portfolio_returns)
    
    # Cumulative log returns and portfolio value
    cumulative_log_returns = np.cumsum(log_returns, axis=0)
    cumulative_log_returns = np.clip(cumulative_log_returns, a_min=None, a_max=10)  # Tightened clipping to avoid overflow
    sim_results = initial_value * np.exp(cumulative_log_returns)

    return sim_results

# Run Monte Carlo simulation
# Note: VaR (Value at Risk) is a commonly used risk metric to estimate the maximum potential loss over a given time period with a certain confidence level.
# Typically, VaR is calculated at 1%, 5%, or 10%. Here, we are using 10% (confidence level) to assess the most extreme scenarios.
# CVaR (Conditional Value at Risk) represents the expected loss in the worst-case scenarios beyond the VaR threshold (aka all the scenarios belonging to the worst 10%).
final_portfolio_values = monte_carlo_simulation(optimized_weights, mean_returns, cov_matrix, mc_sims, T, initial_portfolio_value)
mean_final_value = np.mean(final_portfolio_values[-1, :])
var_90 = np.percentile(final_portfolio_values[-1, :], 10)  # 10th percentile representing the worst 10% VaR scenarios
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

# Enhanced Monte Carlo Simulation Chart
import matplotlib.table as tbl
fig, ax = plt.subplots(figsize=(12, 7))
for i in range(min(mc_sims, 100)):  # Plot a subset for clarity
    ax.plot(np.arange(1, T * ANNUALIZATION_FACTOR + 1), final_portfolio_values[:, i], color='lightblue', alpha=0.1)

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
plot_key_trajectory(ax, np.arange(1, T * ANNUALIZATION_FACTOR + 1), np.full(T * ANNUALIZATION_FACTOR, best_result), color='green', label=f'Best Result: ${best_result:,.2f}')
plot_key_trajectory(ax, np.arange(1, T * ANNUALIZATION_FACTOR + 1), np.full(T * ANNUALIZATION_FACTOR, worst_result), color='red', label=f'Worst Result: ${worst_result:,.2f}')
plot_key_trajectory(ax, np.arange(1, T * ANNUALIZATION_FACTOR + 1), np.full(T * ANNUALIZATION_FACTOR, median_result), color='purple', label=f'Median Result: ${median_result:,.2f}', linestyle='--')
plot_key_trajectory(ax, np.arange(1, T * ANNUALIZATION_FACTOR + 1), np.full(T * ANNUALIZATION_FACTOR, mean_result), color='blue', label=f'Mean Result: ${mean_result:,.2f}', linestyle='--')
plot_key_trajectory(ax, np.arange(1, T * ANNUALIZATION_FACTOR + 1), np.full(T * ANNUALIZATION_FACTOR, mode_result), color='orange', label=f'Mode Result: ${mode_result:,.2f}', linestyle='-.')

plt.xlabel('Days', fontsize=9)
plt.ylabel('Portfolio Value', fontsize=9)
plt.title('Monte Carlo Simulation of Portfolio Growth (Daily)', fontsize=10)
plt.legend(loc='best', fontsize=8)  # Ensure legend does not overlap the table
plt.grid(alpha=0.3)  # Lighter grid lines for less visual clutter
plt.tight_layout()  # Adjust layout to ensure proper spacing
plt.show()


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

# Monte Carlo Simulation with Plotly
fig = go.Figure()
num_trajectories = min(mc_sims, 100)  # Add parameter to control the number of trajectories displayed
for i in range(num_trajectories):
    fig.add_trace(go.Scatter(
        x=np.linspace(1, T * ANNUALIZATION_FACTOR, num=500),  # Downsample to 500 points for efficiency
        y=np.interp(np.linspace(1, T * ANNUALIZATION_FACTOR, num=500), np.arange(1, T * ANNUALIZATION_FACTOR + 1), final_portfolio_values[:, i]),
        mode='lines',
        line=dict(color='lightblue', width=0.5),
        showlegend=False
    ))

# Add key metrics as separate lines
fig.add_trace(go.Scatter(
    x=np.linspace(1, T * ANNUALIZATION_FACTOR, num=500),
    y=np.full(500, mean_final_value),
    mode='lines',
    name='Mean Final Value',
    line=dict(color='blue', dash='dash')
))
fig.add_trace(go.Scatter(
    x=np.linspace(1, T * ANNUALIZATION_FACTOR, num=500),
    y=np.full(500, var_90),
    mode='lines',
    name='10% Value at Risk',
    line=dict(color='red', dash='dash')
))
fig.add_trace(go.Scatter(
    x=np.linspace(1, T * ANNUALIZATION_FACTOR, num=500),
    y=np.full(500, cvar_90),
    mode='lines',
    name='10% Conditional VaR',
    line=dict(color='orange', dash='dash')
))

fig.update_layout(
    title='Monte Carlo Simulation: Portfolio Growth (Daily)',
    xaxis_title='Days',
    yaxis_title='Portfolio Value',
    template='plotly_white',
    legend=dict(x=0.02, y=0.98)
)

def create_dash_app():
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
                            title='Frequency',
                            tickformat=',.0f'
                        ),
                        xaxis=dict(
                            title='Portfolio Value',
                            tickformat='$,.0f'
                        ),
                        showlegend=True
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
                            x=np.linspace(1, T * ANNUALIZATION_FACTOR, num=500),
                            y=np.interp(np.linspace(1, T * ANNUALIZATION_FACTOR, num=500), np.arange(1, T * ANNUALIZATION_FACTOR + 1), np.mean(final_portfolio_values, axis=1)),
                            mode='lines',
                            name='Optimized Portfolio'
                        )
                    ],
                    'layout': go.Layout(
                        title=f'{T}-Year Optimized Portfolio Growth (Daily)',
                        xaxis=dict(
                            title='Days',
                            tickvals=np.linspace(0, T * ANNUALIZATION_FACTOR, num=10),
                            tickmode='array'
                        ),
                        yaxis=dict(
                            title='Portfolio Value',
                            tickformat='$,.0f',
                            gridcolor='lightgray'
                        ),
                        showlegend=True
                    )
                }
            ), width=12)
        ]),
        # Add Monte Carlo Simulation as part of the Dash layout
        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=fig
            ), width=12)
        ])
    ])

    return app

if __name__ == '__main__':
    app = create_dash_app()
    app.run(debug=True)
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
    


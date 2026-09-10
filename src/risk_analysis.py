import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Metrics for risk analysis."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_return: float
    volatility: float
    daily_volatility: float
    sharpe_ratio: float
    max_loss: float
    max_gain: float


class MonteCarloSimulator:
    """Monte Carlo simulator for stock prices using Geometric Brownian Motion."""
    
    def __init__(self, historical_prices: pd.Series, risk_free_rate: float = 0.02):
        self.historical_prices = historical_prices
        self.risk_free_rate = risk_free_rate
        self.mu = 0.0
        self.sigma = 0.0
        self.daily_drift = 0.0
        self.daily_volatility = 0.0
        self.last_S0 = 0.0
        
    def calibrate(self) -> None:
        """Calculate mu (drift) and sigma (volatility) from historical log returns."""
        logger.info("Calibrating simulator parameters from historical data.")
        log_returns = np.log(self.historical_prices / self.historical_prices.shift(1)).dropna()
        
        self.daily_volatility = log_returns.std()
        self.daily_drift = log_returns.mean()
        
        # Annualized metrics assuming 252 trading days
        self.sigma = self.daily_volatility * np.sqrt(252)
        self.mu = self.daily_drift * 252
        
    def simulate(self, days: int = 30, num_simulations: int = 10000, current_price: Optional[float] = None) -> np.ndarray:
        """Run Monte Carlo simulation using Geometric Brownian Motion.
        
        Returns array of shape (num_simulations, days) with simulated price paths.
        """
        if self.sigma == 0.0:
            self.calibrate()
            
        S0 = current_price if current_price is not None else self.historical_prices.iloc[-1]
        self.last_S0 = S0
        logger.info(f"Running {num_simulations} simulations for {days} days starting at price {S0:.2f}.")
        
        dt = 1 / 252
        
        # Generate random shocks
        Z = np.random.normal(0, 1, (num_simulations, days))
        
        # Calculate daily returns matrix
        daily_returns = np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)
        
        # Compute cumulative paths
        price_paths = np.zeros_like(daily_returns)
        price_paths[:, 0] = S0 * daily_returns[:, 0]
        
        for t in range(1, days):
            price_paths[:, t] = price_paths[:, t-1] * daily_returns[:, t]
            
        return price_paths
        
    def calculate_var(self, simulated_paths: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk from terminal returns."""
        terminal_returns = simulated_paths[:, -1] / self.last_S0 - 1
        return float(np.percentile(terminal_returns, (1 - confidence) * 100))

    def calculate_cvar(self, simulated_paths: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk from terminal returns."""
        terminal_returns = simulated_paths[:, -1] / self.last_S0 - 1
        var_threshold = self.calculate_var(simulated_paths, confidence)
        return float(terminal_returns[terminal_returns <= var_threshold].mean())
        
    def get_risk_metrics(self, simulated_paths: np.ndarray) -> RiskMetrics:
        """Calculate all risk metrics from simulation results."""
        terminal_returns = simulated_paths[:, -1] / self.last_S0 - 1
        
        var_95 = self.calculate_var(simulated_paths, 0.95)
        var_99 = self.calculate_var(simulated_paths, 0.99)
        cvar_95 = self.calculate_cvar(simulated_paths, 0.95)
        cvar_99 = self.calculate_cvar(simulated_paths, 0.99)
        
        expected_return = float(terminal_returns.mean())
        
        volatility = self.sigma
        daily_vol = self.daily_volatility
        
        days = simulated_paths.shape[1]
        annualized_return = (1 + expected_return) ** (252 / days) - 1
        sharpe_ratio = float((annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0)
        
        max_loss = float(terminal_returns.min())
        max_gain = float(terminal_returns.max())
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_return=expected_return,
            volatility=volatility,
            daily_volatility=daily_vol,
            sharpe_ratio=sharpe_ratio,
            max_loss=max_loss,
            max_gain=max_gain
        )
        
    def get_percentile_paths(self, simulated_paths: np.ndarray, percentiles: List[int] = [5, 25, 50, 75, 95]) -> Dict[int, np.ndarray]:
        """Extract percentile paths for visualization."""
        return {p: np.percentile(simulated_paths, p, axis=0) for p in percentiles}


def plot_simulation_fan(simulator: MonteCarloSimulator, simulated_paths: np.ndarray, current_price: float) -> go.Figure:
    """Plot a fan chart of simulated paths with percentile bands."""
    days = simulated_paths.shape[1]
    time_axis = np.arange(1, days + 1)
    
    percentiles = [10, 25, 50, 75, 90]
    paths = simulator.get_percentile_paths(simulated_paths, percentiles)
    
    fig = go.Figure()
    
    # 10-90 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_axis, time_axis[::-1]]),
        y=np.concatenate([paths[90], paths[10][::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='10-90th Percentile'
    ))
    
    # 25-75 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_axis, time_axis[::-1]]),
        y=np.concatenate([paths[75], paths[25][::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,255,0.4)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='25-75th Percentile'
    ))
    
    # Median
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=paths[50],
        line=dict(color='rgb(0,0,150)', width=2),
        name='Median'
    ))
    
    fig.update_layout(
        title=f"Monte Carlo Simulation (Current Price: ${current_price:.2f})",
        xaxis_title="Days",
        yaxis_title="Price"
    )
    return fig


def plot_return_distribution(simulated_paths: np.ndarray) -> go.Figure:
    """Plot the distribution of terminal returns with VaR lines."""
    # Approximate initial price using first step backward
    terminal_returns = simulated_paths[:, -1] / simulated_paths[:, 0] - 1
    
    var_95 = float(np.percentile(terminal_returns, 5))
    var_99 = float(np.percentile(terminal_returns, 1))
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=terminal_returns,
        histnorm='probability density',
        name='Simulated Returns',
        marker_color='lightblue'
    ))
    
    # Overlay Normal distribution
    mu, std = norm.fit(terminal_returns)
    xmin, xmax = terminal_returns.min(), terminal_returns.max()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    fig.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Normal Fit', line=dict(color='red')))
    
    # VaR lines
    fig.add_vline(x=var_95, line_dash="dash", line_color="orange", annotation_text=f"VaR 95: {var_95:.1%}")
    fig.add_vline(x=var_99, line_dash="dash", line_color="red", annotation_text=f"VaR 99: {var_99:.1%}")
    
    fig.update_layout(title="Distribution of Terminal Returns", xaxis_title="Return", yaxis_title="Density")
    return fig


def plot_var_gauge(risk_metrics: RiskMetrics) -> go.Figure:
    """Plot gauge indicators for VaR 95% and VaR 99%."""
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_metrics.var_95 * 100,
        title={'text': "VaR 95% (%)"},
        domain={'x': [0, 0.45], 'y': [0, 1]},
        gauge={'axis': {'range': [min(-20.0, risk_metrics.var_99 * 100), 0.0]}}
    ))
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_metrics.var_99 * 100,
        title={'text': "VaR 99% (%)"},
        domain={'x': [0.55, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [min(-30.0, risk_metrics.var_99 * 100), 0.0]}}
    ))
    
    fig.update_layout(title="Value at Risk (VaR) Gauges")
    return fig


def plot_risk_return_scatter(tickers_data: Dict[str, RiskMetrics]) -> go.Figure:
    """Plot risk-return scatter with efficient frontier approximation."""
    tickers = list(tickers_data.keys())
    returns = [tickers_data[t].expected_return for t in tickers]
    vols = [tickers_data[t].volatility for t in tickers]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols,
        y=returns,
        mode='markers+text',
        text=tickers,
        textposition="top center",
        marker=dict(size=10, color=returns, colorscale='Viridis', showscale=True),
        name="Stocks"
    ))
    
    if len(tickers) > 2:
        z = np.polyfit(vols, returns, 2)
        p = np.poly1d(z)
        vols_sorted = np.linspace(min(vols), max(vols), 50)
        fig.add_trace(go.Scatter(
            x=vols_sorted,
            y=p(vols_sorted),
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name="Trend"
        ))
        
    fig.update_layout(title="Risk-Return Profile", xaxis_title="Volatility (Risk)", yaxis_title="Expected Return")
    return fig


def generate_demo_risk_analysis(ticker: str = 'AAPL') -> Tuple[MonteCarloSimulator, np.ndarray, RiskMetrics]:
    """Generate a demo risk analysis with simulated historical data."""
    logger.info(f"Generating demo risk analysis for {ticker}")
    np.random.seed(42)
    
    # Generate realistic historical prices (random walk with drift)
    days = 252 * 2
    mu = 0.0005  # daily drift
    sigma = 0.015  # daily vol
    returns = np.random.normal(mu, sigma, days)
    
    price = 150.0
    prices = [price]
    for r in returns:
        price = price * np.exp(r)
        prices.append(price)
        
    hist_prices = pd.Series(prices)
    
    simulator = MonteCarloSimulator(hist_prices)
    simulator.calibrate()
    paths = simulator.simulate(days=30, num_simulations=5000, current_price=prices[-1])
    metrics = simulator.get_risk_metrics(paths)
    
    return simulator, paths, metrics

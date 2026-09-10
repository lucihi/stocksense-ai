import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

@dataclass
class NewsEvent:
    date: Union[str, datetime]
    headline: str
    sentiment_score: float
    source: str
    z_score: float = 0.0
    abnormal_return: float = 0.0
    cumulative_abnormal_return: float = 0.0

class NewsImpactAnalyzer:
    def __init__(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame):
        """
        Initialize the NewsImpactAnalyzer.
        
        Args:
            stock_data: DataFrame with at least Date, Close, Volume.
            sentiment_data: DataFrame with at least date, mean_sentiment, headlines.
        """
        self.stock_data = stock_data.copy()
        self.sentiment_data = sentiment_data.copy()
        
        # Ensure dates are datetime
        if 'Date' in self.stock_data.columns:
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
            self.stock_data = self.stock_data.sort_values('Date').reset_index(drop=True)
        
        if 'date' in self.sentiment_data.columns:
            self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['date'])
            self.sentiment_data = self.sentiment_data.sort_values('date').reset_index(drop=True)
            
        # Calculate daily returns
        self.stock_data['Return'] = self.stock_data['Close'].pct_change()

    def detect_events(self, z_threshold: float = 1.5) -> List[NewsEvent]:
        """
        Detect days with abnormal sentiment.
        """
        if 'mean_sentiment' not in self.sentiment_data.columns:
            logger.warning("No mean_sentiment column in sentiment data.")
            return []
            
        mean_s = self.sentiment_data['mean_sentiment'].mean()
        std_s = self.sentiment_data['mean_sentiment'].std()
        
        self.sentiment_data['z_score'] = (self.sentiment_data['mean_sentiment'] - mean_s) / std_s
        
        events_df = self.sentiment_data[self.sentiment_data['z_score'].abs() > z_threshold].copy()
        events_df = events_df.sort_values(by='z_score', key=abs, ascending=False)
        
        events = []
        for _, row in events_df.iterrows():
            headline = row.get('headlines', row.get('title', 'Unknown Event'))
            source = row.get('source', 'Unknown Source')
            
            event = NewsEvent(
                date=row['date'],
                headline=headline,
                sentiment_score=row['mean_sentiment'],
                source=source,
                z_score=row['z_score']
            )
            events.append(event)
            
        return events

    def calculate_expected_returns(self, estimation_window: int = 60) -> pd.Series:
        """
        Calculate expected returns based on a rolling mean of returns.
        """
        self.stock_data['Expected_Return'] = self.stock_data['Return'].rolling(window=estimation_window, min_periods=10).mean()
        return self.stock_data['Expected_Return']

    def calculate_abnormal_returns(self, event_date: Union[str, datetime], event_window: Tuple[int, int] = (-2, 5)) -> pd.DataFrame:
        """
        Calculate abnormal returns around an event date.
        """
        event_date = pd.to_datetime(event_date)
        
        # Find the index of the closest trading day to the event date
        closest_idx_series = self.stock_data.index[self.stock_data['Date'] >= event_date]
        if len(closest_idx_series) == 0:
            return pd.DataFrame()
            
        event_idx = closest_idx_series[0]
        
        start_idx = max(0, event_idx + event_window[0])
        end_idx = min(len(self.stock_data) - 1, event_idx + event_window[1])
        
        if 'Expected_Return' not in self.stock_data.columns:
            self.calculate_expected_returns()
            
        window_data = self.stock_data.iloc[start_idx:end_idx + 1].copy()
        window_data['day_offset'] = np.arange(start_idx - event_idx, end_idx - event_idx + 1)
        
        # Avoid issues with NAs
        expected = window_data['Expected_Return'].fillna(0)
        actual = window_data['Return'].fillna(0)
        
        window_data['actual_return'] = actual
        window_data['expected_return'] = expected
        window_data['abnormal_return'] = actual - expected
        window_data['car'] = window_data['abnormal_return'].cumsum()
        
        return window_data[['Date', 'day_offset', 'actual_return', 'expected_return', 'abnormal_return', 'car']].rename(columns={'Date': 'date'})

    def build_event_study(self, events: Optional[List[NewsEvent]] = None, top_n: int = 10) -> Dict[str, Any]:
        """
        Run event study for top N events.
        """
        if events is None:
            events = self.detect_events()
            
        top_events = events[:top_n]
        results = {}
        aggregate_car = pd.DataFrame()
        
        for i, event in enumerate(top_events):
            ar_df = self.calculate_abnormal_returns(event.date)
            if ar_df.empty:
                continue
                
            # Update event metrics
            event.abnormal_return = ar_df.loc[ar_df['day_offset'] == 0, 'abnormal_return'].values[0] if 0 in ar_df['day_offset'].values else 0
            event.cumulative_abnormal_return = ar_df['car'].iloc[-1] if not ar_df.empty else 0
            
            results[f'event_{i}'] = {
                'event': event,
                'abnormal_returns': ar_df
            }
            
            # For aggregate CAR
            car_series = ar_df.set_index('day_offset')['car']
            aggregate_car = pd.concat([aggregate_car, car_series.rename(f'event_{i}')], axis=1)
            
        return {
            'events': [res['event'] for res in results.values()],
            'event_returns': results,
            'aggregate_car': aggregate_car.mean(axis=1) if not aggregate_car.empty else pd.Series()
        }

    def get_sentiment_return_correlation(self) -> Dict[str, Any]:
        """
        Calculate correlation between daily sentiment and next-day return.
        """
        merged = pd.merge(
            self.stock_data[['Date', 'Return']], 
            self.sentiment_data[['date', 'mean_sentiment']], 
            left_on='Date', 
            right_on='date'
        )
        
        if merged.empty:
            return {}
            
        merged = merged.dropna()
        merged['next_day_return'] = merged['Return'].shift(-1)
        merged = merged.dropna()
        
        if len(merged) < 2:
            return {}
            
        pearson_r, pearson_p = stats.pearsonr(merged['mean_sentiment'], merged['next_day_return'])
        spearman_r, spearman_p = stats.spearmanr(merged['mean_sentiment'], merged['next_day_return'])
        
        # Lead/lag correlations
        lags = range(-3, 4)
        lead_lag = {}
        for lag in lags:
            shifted_return = merged['Return'].shift(-lag)
            valid_idx = shifted_return.notna()
            if valid_idx.sum() > 2:
                corr, _ = stats.pearsonr(merged.loc[valid_idx, 'mean_sentiment'], shifted_return[valid_idx])
                lead_lag[f't{"+" if lag > 0 else ""}{lag}'] = corr
                
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'lead_lag_correlations': lead_lag
        }


# Visualization functions

def plot_event_timeline(stock_data: pd.DataFrame, events: List[NewsEvent]) -> go.Figure:
    fig = go.Figure()
    
    # Stock price line
    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['Close'],
        mode='lines',
        name='Stock Price',
        line=dict(color='blue')
    ))
    
    # Event markers
    for event in events:
        color = 'green' if event.sentiment_score > 0 else 'red'
        fig.add_vline(x=event.date, line_width=1, line_dash="dash", line_color=color)
        
        fig.add_trace(go.Scatter(
            x=[event.date],
            y=[stock_data.loc[stock_data['Date'] >= pd.to_datetime(event.date), 'Close'].iloc[0] if not stock_data[stock_data['Date'] >= pd.to_datetime(event.date)].empty else stock_data['Close'].iloc[-1]],
            mode='markers',
            marker=dict(color=color, size=10, symbol='star'),
            name=f"{event.date.date() if isinstance(event.date, datetime) else event.date}",
            hoverinfo='text',
            hovertext=f"{event.headline}<br>Sentiment: {event.sentiment_score:.2f}<br>Z-Score: {event.z_score:.2f}"
        ))
        
    fig.update_layout(title='Event Timeline', xaxis_title='Date', yaxis_title='Price', showlegend=False)
    return fig

def plot_car_chart(abnormal_returns_df: pd.DataFrame, event: NewsEvent) -> go.Figure:
    fig = go.Figure()
    
    colors = ['green' if r >= 0 else 'red' for r in abnormal_returns_df['abnormal_return']]
    
    fig.add_trace(go.Bar(
        x=abnormal_returns_df['day_offset'],
        y=abnormal_returns_df['abnormal_return'],
        name='Daily AR',
        marker_color=colors
    ))
    
    fig.add_trace(go.Scatter(
        x=abnormal_returns_df['day_offset'],
        y=abnormal_returns_df['car'],
        mode='lines+markers',
        name='CAR',
        line=dict(color='black', width=2)
    ))
    
    fig.update_layout(
        title=f"Event Impact: {event.headline}",
        xaxis_title='Days relative to event',
        yaxis_title='Abnormal Return',
        barmode='group'
    )
    return fig

def plot_sentiment_return_scatter(sentiment_data: pd.DataFrame, stock_data: pd.DataFrame) -> go.Figure:
    merged = pd.merge(
        stock_data[['Date', 'Return']].rename(columns={'Date': 'date'}), 
        sentiment_data[['date', 'mean_sentiment']], 
        on='date'
    )
    merged['next_day_return'] = merged['Return'].shift(-1)
    merged = merged.dropna()
    
    merged['sentiment_label'] = pd.cut(
        merged['mean_sentiment'], 
        bins=[-np.inf, -0.2, 0.2, np.inf], 
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    fig = px.scatter(
        merged, 
        x='mean_sentiment', 
        y='next_day_return', 
        color='sentiment_label',
        trendline="ols",
        title='Sentiment vs Next-Day Return',
        labels={'mean_sentiment': 'Daily Sentiment', 'next_day_return': 'Next-Day Return'}
    )
    return fig

def plot_aggregate_event_impact(event_study_results: Dict[str, Any]) -> go.Figure:
    events = event_study_results.get('events', [])
    event_returns = event_study_results.get('event_returns', {})
    
    pos_cars = []
    neg_cars = []
    
    for i, event in enumerate(events):
        event_data = event_returns.get(f'event_{i}')
        if not event_data:
            continue
            
        ar_df = event_data['abnormal_returns']
        if ar_df.empty:
            continue
            
        car_series = ar_df.set_index('day_offset')['car']
        if event.sentiment_score > 0:
            pos_cars.append(car_series)
        else:
            neg_cars.append(car_series)
            
    fig = go.Figure()
    
    if pos_cars:
        pos_avg = pd.concat(pos_cars, axis=1).mean(axis=1)
        fig.add_trace(go.Scatter(
            x=pos_avg.index, y=pos_avg.values, mode='lines+markers', name='Positive Events CAR', line=dict(color='green')
        ))
        
    if neg_cars:
        neg_avg = pd.concat(neg_cars, axis=1).mean(axis=1)
        fig.add_trace(go.Scatter(
            x=neg_avg.index, y=neg_avg.values, mode='lines+markers', name='Negative Events CAR', line=dict(color='red')
        ))
        
    fig.update_layout(
        title='Aggregate Cumulative Abnormal Returns',
        xaxis_title='Days relative to event',
        yaxis_title='Average CAR'
    )
    return fig


def generate_demo_news_impact(ticker: str = 'AAPL') -> Dict[str, Any]:
    """
    Generate realistic demo data and run analysis for dashboard display.
    """
    logger.info(f"Generating demo news impact data for {ticker}")
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=252, freq='B')
    
    # Synthetic stock prices (random walk with drift)
    returns = np.random.normal(0.0005, 0.015, len(dates))
    price = 150 * np.exp(np.cumsum(returns))
    stock_data = pd.DataFrame({
        'Date': dates,
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Synthetic sentiment
    sentiment = np.random.normal(0.1, 0.3, len(dates))
    sentiment = np.clip(sentiment, -1, 1)
    
    # Create distinct events
    event_indices = np.random.choice(range(20, 230), size=10, replace=False)
    headlines = [
        f'{ticker} reports record Q4 earnings beating estimates',
        f'Fed signals potential rate cuts in 2024',
        f'Major data breach reported at {ticker}',
        f'{ticker} announces $90B stock buyback program',
        f'Analyst downgrades {ticker} citing weak demand',
        f'New product launch from {ticker} exceeds expectations',
        f'Regulatory scrutiny increases for {ticker}',
        f'{ticker} acquires promising AI startup',
        f'Supply chain disruptions threaten {ticker} production',
        f'{ticker} CEO steps down unexpectedly'
    ]
    
    sentiments = [0.8, 0.6, -0.9, 0.7, -0.6, 0.85, -0.5, 0.65, -0.7, -0.8]
    
    sentiment_data = pd.DataFrame({
        'date': dates,
        'mean_sentiment': sentiment,
        'headlines': ['Normal trading day'] * len(dates)
    })
    
    # Inject events into sentiment data
    for idx, (h, s) in zip(event_indices, zip(headlines, sentiments)):
        sentiment_data.at[idx, 'mean_sentiment'] = s
        sentiment_data.at[idx, 'headlines'] = h
        
        # Inject price reaction
        stock_data.at[idx, 'Close'] = stock_data.at[max(0, idx-1), 'Close'] * (1 + s * 0.05)
        # Recompute subsequent prices to maintain the random walk
        for j in range(idx + 1, len(dates)):
            stock_data.at[j, 'Close'] = stock_data.at[j-1, 'Close'] * (1 + returns[j])

    # Analyze
    analyzer = NewsImpactAnalyzer(stock_data, sentiment_data)
    events = analyzer.detect_events(z_threshold=1.5)
    
    event_study_results = analyzer.build_event_study(events, top_n=10)
    correlation = analyzer.get_sentiment_return_correlation()
    
    return {
        'stock_data': stock_data,
        'sentiment_data': sentiment_data,
        'events': event_study_results['events'],
        'event_study': event_study_results,
        'correlation': correlation
    }

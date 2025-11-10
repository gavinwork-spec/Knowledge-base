"""
Trend Predictor Agent
XAgent-inspired trend prediction agent with advanced forecasting capabilities,
market analysis, and predictive modeling for manufacturing insights.
"""

import asyncio
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import warnings
warnings.filterwarnings('ignore')

# Machine learning and time series
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.cluster import KMeans
import joblib

# Time series analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Advanced analytics
from scipy import stats
from scipy.signal import find_peaks
import networkx as nx

# Import agent framework
from multi_agent_orchestrator import BaseAgent, AgentTask, AgentCapability, AgentStatus
from multi_agent_system.protocols.agent_communication import (
    MessageRouter, TaskDelegator, AgentMessage, TaskRequest, TaskResponse,
    MessageType, Priority
)
from multi_agent_system.agents.specialized_agents import EnhancedBaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrendPrediction:
    """Represents a trend prediction result"""
    prediction_id: str
    metric_name: str
    prediction_type: str
    time_horizon: str
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    trend_direction: str
    trend_strength: float
    seasonal_pattern: bool
    anomalies_detected: List[Dict[str, Any]]
    accuracy_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketSignal:
    """Represents a market signal or indicator"""
    signal_id: str
    signal_type: str
    strength: float
    description: str
    impacted_metrics: List[str]
    confidence: float
    time_sensitivity: str
    recommended_actions: List[str]
    detected_at: datetime = field(default_factory=datetime.now)


class TrendPredictorAgent(EnhancedBaseAgent):
    """Advanced trend prediction agent with ML-powered forecasting"""

    def __init__(self, orchestrator):
        super().__init__("trend_predictor", "Trend Predictor", orchestrator)
        self.capabilities.update({
            "trend_prediction",
            "time_series_forecasting",
            "market_analysis",
            "seasonal_decomposition",
            "anomaly_detection",
            "predictive_modeling",
            "market_signal_detection",
            "demand_forecasting",
            "price_trending",
            "capacity_planning"
        })
        self.prediction_models = {}
        self.historical_data = {}
        self.market_signals = deque(maxlen=1000)
        self.prediction_cache = {}
        self.analysis_history = []

    async def initialize(self):
        """Initialize trend prediction models"""
        await super().initialize()

        # Initialize prediction models
        await self._initialize_prediction_models()

        # Load historical data
        await self._load_historical_data()

        # Initialize market signal detectors
        await self._initialize_signal_detectors()

        logger.info("Trend Predictor Agent initialized with advanced forecasting models")

    async def _initialize_prediction_models(self):
        """Initialize machine learning models for trend prediction"""
        try:
            # Load pre-trained models if available
            self.price_trend_model = joblib.load('models/price_trend_predictor.pkl')
            self.demand_forecast_model = joblib.load('models/demand_forecast_predictor.pkl')
            self.anomaly_detector = joblib.load('models/trend_anomaly_detector.pkl')
        except FileNotFoundError:
            logger.warning("Pre-trained models not found, initializing new models")
            # Initialize new models
            self.price_trend_model = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'lr': LinearRegression(),
                'gbr': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            self.demand_forecast_model = {
                'arima': None,  # Will be created per time series
                'ets': None,   # Exponential smoothing
                'mlp': None    # Multi-layer perceptron
            }
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

    async def _load_historical_data(self):
        """Load historical data for trend analysis"""
        try:
            conn = sqlite3.connect('knowledge_base.db')

            # Load factory quotes data
            quotes_query = """
                SELECT fq.*, f.name as factory_name, f.location, f.industry_type
                FROM factory_quotes fq
                JOIN factories f ON fq.factory_id = f.id
                ORDER BY fq.quote_date ASC
            """
            self.historical_data['quotes'] = pd.read_sql_query(quotes_query, conn)

            # Load reminder records for activity trends
            reminders_query = """
                SELECT *, datetime(created_at) as reminder_date
                FROM reminder_records
                ORDER BY created_at ASC
            """
            self.historical_data['reminders'] = pd.read_sql_query(reminders_query, conn)

            # Load knowledge base entries for learning trends
            knowledge_query = """
                SELECT *, datetime(created_at) as entry_date
                FROM knowledge_base_entries
                ORDER BY created_at ASC
            """
            self.historical_data['knowledge'] = pd.read_sql_query(knowledge_query, conn)

            conn.close()

            # Preprocess data
            await self._preprocess_historical_data()

            logger.info(f"Loaded historical data: quotes={len(self.historical_data['quotes'])}, "
                       f"reminders={len(self.historical_data['reminders'])}, "
                       f"knowledge={len(self.historical_data['knowledge'])}")

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            # Create empty dataframes
            self.historical_data = {
                'quotes': pd.DataFrame(),
                'reminders': pd.DataFrame(),
                'knowledge': pd.DataFrame()
            }

    async def _preprocess_historical_data(self):
        """Preprocess historical data for analysis"""
        # Process quotes data
        if not self.historical_data['quotes'].empty:
            df = self.historical_data['quotes'].copy()
            df['quote_date'] = pd.to_datetime(df['quote_date'])
            df['year'] = df['quote_date'].dt.year
            df['month'] = df['quote_date'].dt.month
            df['quarter'] = df['quote_date'].dt.quarter
            df['day_of_week'] = df['quote_date'].dt.dayofweek
            self.historical_data['quotes'] = df

        # Process reminders data
        if not self.historical_data['reminders'].empty:
            df = self.historical_data['reminders'].copy()
            df['reminder_date'] = pd.to_datetime(df['reminder_date'])
            df['year'] = df['reminder_date'].dt.year
            df['month'] = df['reminder_date'].dt.month
            self.historical_data['reminders'] = df

        # Process knowledge data
        if not self.historical_data['knowledge'].empty:
            df = self.historical_data['knowledge'].copy()
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['year'] = df['entry_date'].dt.year
            df['month'] = df['entry_date'].dt.month
            self.historical_data['knowledge'] = df

    async def _execute_task_logic(self, task: TaskRequest) -> Dict[str, Any]:
        """Execute trend prediction task"""
        task_type = task.task_type
        parameters = task.parameters

        if task_type == "predict_price_trend":
            return await self._predict_price_trend(parameters)
        elif task_type == "forecast_demand":
            return await self._forecast_demand(parameters)
        elif task_type == "analyze_seasonal_patterns":
            return await self._analyze_seasonal_patterns(parameters)
        elif task_type == "detect_anomalies":
            return await self._detect_anomalies(parameters)
        elif task_type == "market_signal_analysis":
            return await self._analyze_market_signals(parameters)
        elif task_type == "capacity_planning":
            return await self._predict_capacity_needs(parameters)
        elif task_type == "competitor_trend_analysis":
            return await self._analyze_competitor_trends(parameters)
        elif task_type == "predict_market_movements":
            return await self._predict_market_movements(parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _predict_price_trend(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Predict price trends for specific products or categories"""
        try:
            product_category = parameters.get('product_category')
            time_horizon = parameters.get('time_horizon', '90d')  # Default 90 days
            confidence_level = parameters.get('confidence_level', 0.95)

            # Prepare time series data
            price_data = await self._prepare_price_time_series(product_category)

            if price_data.empty:
                return {'error': 'No historical price data available for analysis'}

            # Decompose time series
            decomposition = await self._decompose_time_series(price_data['price'])

            # Generate multiple forecasts
            forecasts = await self._generate_price_forecasts(price_data, time_horizon)

            # Combine forecasts using ensemble method
            ensemble_forecast = await self._ensemble_price_forecasts(forecasts)

            # Calculate confidence intervals
            confidence_intervals = await self._calculate_confidence_intervals(
                ensemble_forecast, price_data, confidence_level
            )

            # Detect trend direction and strength
            trend_analysis = await self._analyze_trend_characteristics(ensemble_forecast)

            # Identify market signals
            market_signals = await self._identify_price_signals(ensemble_forecast, decomposition)

            # Create prediction object
            prediction = TrendPrediction(
                prediction_id=str(uuid.uuid4()),
                metric_name=f"{product_category}_price_trend",
                prediction_type="price_forecast",
                time_horizon=time_horizon,
                predicted_values=ensemble_forecast,
                confidence_intervals=confidence_intervals,
                trend_direction=trend_analysis['direction'],
                trend_strength=trend_analysis['strength'],
                seasonal_pattern=decomposition['seasonal'],
                anomalies_detected=market_signals,
                accuracy_metrics=await self._calculate_forecast_accuracy(price_data, ensemble_forecast)
            )

            # Cache prediction
            self.prediction_cache[prediction.prediction_id] = prediction

            return {
                'prediction_id': prediction.prediction_id,
                'forecast': ensemble_forecast,
                'confidence_intervals': confidence_intervals,
                'trend_analysis': trend_analysis,
                'decomposition': decomposition,
                'market_signals': market_signals,
                'accuracy_metrics': prediction.accuracy_metrics,
                'metadata': {
                    'data_points_used': len(price_data),
                    'forecast_horizon': time_horizon,
                    'confidence_level': confidence_level,
                    'created_at': prediction.created_at.isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error predicting price trend: {e}")
            raise

    async def _prepare_price_time_series(self, product_category: Optional[str]) -> pd.DataFrame:
        """Prepare time series data for price analysis"""
        if self.historical_data['quotes'].empty:
            return pd.DataFrame()

        df = self.historical_data['quotes'].copy()

        # Filter by product category if specified
        if product_category:
            df = df[df['product_category'] == product_category]

        if df.empty:
            return pd.DataFrame()

        # Group by date and calculate statistics
        price_series = df.groupby('quote_date').agg({
            'price': ['mean', 'median', 'std', 'count']
        }).reset_index()

        # Flatten column names
        price_series.columns = ['date', 'price_mean', 'price_median', 'price_std', 'quote_count']

        # Filter dates with sufficient data
        min_quotes_per_day = 3
        price_series = price_series[price_series['quote_count'] >= min_quotes_per_day]

        # Sort by date
        price_series = price_series.sort_values('date')

        # Fill missing dates
        price_series = price_series.set_index('date')
        full_date_range = pd.date_range(
            start=price_series.index.min(),
            end=price_series.index.max(),
            freq='D'
        )
        price_series = price_series.reindex(full_date_range)
        price_series = price_series.interpolate(method='linear')

        return price_series

    async def _decompose_time_series(self, series: pd.Series) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components"""
        try:
            if len(series) < 24:  # Need at least 2 periods for seasonal decomposition
                return {
                    'trend': False,
                    'seasonal': False,
                    'residual': True,
                    'trend_component': None,
                    'seasonal_component': None,
                    'residual_component': series.values
                }

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(series, model='additive', period=7)  # Weekly seasonality

            # Test for stationarity
            adf_result = adfuller(series.dropna())

            return {
                'trend': True,
                'seasonal': True,
                'residual': True,
                'trend_component': decomposition.trend.dropna().tolist(),
                'seasonal_component': decomposition.seasonal.dropna().tolist(),
                'residual_component': decomposition.resid.dropna().tolist(),
                'stationarity': {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'is_stationary': adf_result[1] < 0.05
                }
            }

        except Exception as e:
            logger.error(f"Error in time series decomposition: {e}")
            return {
                'trend': False,
                'seasonal': False,
                'residual': True,
                'error': str(e)
            }

    async def _generate_price_forecasts(self, price_data: pd.DataFrame, time_horizon: str) -> Dict[str, List[float]]:
        """Generate multiple price forecasts using different models"""
        forecasts = {}

        # Parse time horizon
        horizon_days = self._parse_time_horizon(time_horizon)

        # Prepare features
        X, y = await self._prepare_forecasting_features(price_data)

        if len(X) < 10:  # Need sufficient data for forecasting
            return {'error': 'Insufficient historical data for forecasting'}

        # Random Forest forecast
        try:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X[:-1], y[1:])  # Use previous day to predict current
            rf_forecast = await self._recursive_forecast(rf_model, X[-1:], horizon_days)
            forecasts['random_forest'] = rf_forecast
        except Exception as e:
            logger.error(f"Random Forest forecast failed: {e}")

        # Linear Regression forecast
        try:
            lr_model = LinearRegression()
            lr_model.fit(X[:-1], y[1:])
            lr_forecast = await self._recursive_forecast(lr_model, X[-1:], horizon_days)
            forecasts['linear_regression'] = lr_forecast
        except Exception as e:
            logger.error(f"Linear Regression forecast failed: {e}")

        # ARIMA forecast
        try:
            arima_model = ARIMA(price_data['price_mean'], order=(1, 1, 1))
            arima_result = arima_model.fit()
            arima_forecast = arima_result.forecast(steps=horizon_days)
            forecasts['arima'] = arima_forecast.tolist()
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")

        # Exponential Smoothing forecast
        try:
            ets_model = ExponentialSmoothing(price_data['price_mean'], trend='add', seasonal='add', seasonal_periods=7)
            ets_result = ets_model.fit()
            ets_forecast = ets_result.forecast(steps=horizon_days)
            forecasts['exponential_smoothing'] = ets_forecast.tolist()
        except Exception as e:
            logger.error(f"Exponential Smoothing forecast failed: {e}")

        return forecasts

    async def _prepare_forecasting_features(self, price_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for machine learning forecasting"""
        features = []
        target = []

        for i in range(1, len(price_data)):
            # Current day's features
            row = price_data.iloc[i]
            features.append([
                row['price_mean'],
                row['price_median'],
                row['price_std'],
                row['quote_count'],
                i,  # Time index
                i % 7,  # Day of week
                (i // 7) % 52,  # Week of year
                (i // 30) % 12,  # Month
            ])
            target.append(row['price_mean'])

        return np.array(features), np.array(target)

    async def _recursive_forecast(self, model, last_features: np.ndarray, horizon: int) -> List[float]:
        """Generate recursive forecast using a trained model"""
        forecast = []
        current_features = last_features[-1:].copy()

        for _ in range(horizon):
            # Update time index
            current_features[0][4] += 1  # Time index
            current_features[0][5] = (current_features[0][5] + 1) % 7  # Day of week

            # Make prediction
            pred = model.predict(current_features)[0]
            forecast.append(pred)

            # Update features for next prediction
            current_features[0][0] = pred  # Update price_mean

        return forecast

    async def _ensemble_price_forecasts(self, forecasts: Dict[str, List[float]]) -> List[float]:
        """Combine multiple forecasts using ensemble method"""
        if not forecasts or 'error' in forecasts:
            return []

        forecast_values = list(forecasts.values())
        if not forecast_values:
            return []

        # Ensure all forecasts have the same length
        min_length = min(len(f) for f in forecast_values)
        aligned_forecasts = [f[:min_length] for f in forecast_values]

        # Calculate weighted average (equal weights for now, can be optimized)
        ensemble = []
        for i in range(min_length):
            values_at_t = [f[i] for f in aligned_forecasts]
            ensemble.append(statistics.mean(values_at_t))

        return ensemble

    async def _calculate_confidence_intervals(self, forecast: List[float],
                                            historical_data: pd.DataFrame,
                                            confidence_level: float) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for forecasts"""
        if not forecast or historical_data.empty:
            return []

        # Calculate historical forecast errors
        historical_volatility = historical_data['price_std'].mean()
        mean_price = historical_data['price_mean'].mean()

        # Calculate confidence bounds based on historical volatility
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        confidence_intervals = []

        for predicted_value in forecast:
            # Relative confidence based on prediction horizon
            relative_uncertainty = 0.1 + 0.05 * (len(confidence_intervals) / len(forecast))
            absolute_uncertainty = predicted_value * relative_uncertainty

            # Use both relative uncertainty and historical volatility
            total_uncertainty = max(absolute_uncertainty, z_score * historical_volatility)

            lower_bound = predicted_value - total_uncertainty
            upper_bound = predicted_value + total_uncertainty

            confidence_intervals.append((lower_bound, upper_bound))

        return confidence_intervals

    async def _analyze_trend_characteristics(self, forecast: List[float]) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        if len(forecast) < 2:
            return {'direction': 'unknown', 'strength': 0.0}

        # Calculate trend slope
        x = np.arange(len(forecast))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, forecast)

        # Determine trend direction
        if slope > 0.01:
            direction = 'increasing'
        elif slope < -0.01:
            direction = 'decreasing'
        else:
            direction = 'stable'

        # Calculate trend strength (0-1)
        strength = min(abs(r_value), 1.0)

        # Detect trend changes
        trend_changes = []
        for i in range(1, len(forecast)):
            if (forecast[i] - forecast[i-1]) / forecast[i-1] > 0.05:  # 5% increase
                trend_changes.append({'position': i, 'type': 'increase'})
            elif (forecast[i] - forecast[i-1]) / forecast[i-1] < -0.05:  # 5% decrease
                trend_changes.append({'position': i, 'type': 'decrease'})

        return {
            'direction': direction,
            'strength': strength,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'trend_changes': trend_changes,
            'volatility': np.std(forecast) / np.mean(forecast) if np.mean(forecast) != 0 else 0
        }

    async def _identify_price_signals(self, forecast: List[float], decomposition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify significant market signals in price forecast"""
        signals = []

        if not forecast:
            return signals

        # Detect price spikes or drops
        for i in range(1, len(forecast)):
            change_rate = (forecast[i] - forecast[i-1]) / forecast[i-1]

            if abs(change_rate) > 0.1:  # 10% change threshold
                signal_type = 'price_spike' if change_rate > 0 else 'price_drop'
                signals.append({
                    'type': signal_type,
                    'position': i,
                    'magnitude': abs(change_rate),
                    'description': f'{signal_type.replace("_", " ").title()} of {abs(change_rate)*100:.1f}% detected'
                })

        # Detect pattern changes
        if decomposition.get('seasonal_component'):
            seasonal_pattern = decomposition['seasonal_component']
            if len(seasonal_pattern) > 0:
                # Check for unusual seasonal behavior
                seasonal_mean = np.mean(seasonal_pattern)
                seasonal_std = np.std(seasonal_pattern)

                for i, value in enumerate(forecast[:len(seasonal_pattern)]):
                    expected = seasonal_pattern[i] if i < len(seasonal_pattern) else seasonal_mean
                    if abs(value - expected) > 2 * seasonal_std:
                        signals.append({
                            'type': 'seasonal_anomaly',
                            'position': i,
                            'expected_value': expected,
                            'actual_value': value,
                            'description': f'Value deviates significantly from expected seasonal pattern'
                        })

        return signals

    async def _forecast_demand(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast demand for products or services"""
        try:
            product_category = parameters.get('product_category')
            time_horizon = parameters.get('time_horizon', '30d')
            external_factors = parameters.get('external_factors', {})

            # Prepare demand data
            demand_data = await self._prepare_demand_time_series(product_category)

            if demand_data.empty:
                return {'error': 'No demand data available for forecasting'}

            # Analyze demand patterns
            demand_patterns = await self._analyze_demand_patterns(demand_data)

            # Generate demand forecast
            forecast_result = await self._generate_demand_forecast(demand_data, time_horizon, external_factors)

            # Calculate capacity implications
            capacity_analysis = await self._analyze_capacity_implications(forecast_result, demand_patterns)

            return {
                'demand_forecast': forecast_result,
                'demand_patterns': demand_patterns,
                'capacity_analysis': capacity_analysis,
                'recommendations': await self._generate_demand_recommendations(forecast_result, capacity_analysis)
            }

        except Exception as e:
            logger.error(f"Error forecasting demand: {e}")
            raise

    async def _prepare_demand_time_series(self, product_category: Optional[str]) -> pd.DataFrame:
        """Prepare time series data for demand analysis"""
        # Use quotes as proxy for demand (number of quotes per day)
        if self.historical_data['quotes'].empty:
            return pd.DataFrame()

        df = self.historical_data['quotes'].copy()

        # Filter by product category if specified
        if product_category:
            df = df[df['product_category'] == product_category]

        if df.empty:
            return pd.DataFrame()

        # Count quotes per day as demand indicator
        demand_series = df.groupby('quote_date').agg({
            'factory_id': 'count',  # Number of quotes = demand indicator
            'product_category': 'nunique'  # Number of different products
        }).reset_index()
        demand_series.columns = ['date', 'demand_indicator', 'product_diversity']

        # Sort and fill missing dates
        demand_series = demand_series.sort_values('date')
        demand_series = demand_series.set_index('date')
        full_date_range = pd.date_range(
            start=demand_series.index.min(),
            end=demand_series.index.max(),
            freq='D'
        )
        demand_series = demand_series.reindex(full_date_range)
        demand_series['demand_indicator'] = demand_series['demand_indicator'].fillna(0)
        demand_series['product_diversity'] = demand_series['product_diversity'].fillna(0)

        return demand_series.reset_index()

    async def _analyze_demand_patterns(self, demand_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demand patterns and seasonality"""
        patterns = {
            'weekly_pattern': {},
            'monthly_pattern': {},
            'seasonal_trends': {},
            'growth_rate': 0.0,
            'volatility': 0.0
        }

        if demand_data.empty:
            return patterns

        # Add day of week and month
        demand_data = demand_data.copy()
        demand_data['day_of_week'] = demand_data['date'].dt.dayofweek
        demand_data['month'] = demand_data['date'].dt.month

        # Weekly pattern
        weekly_avg = demand_data.groupby('day_of_week')['demand_indicator'].mean()
        patterns['weekly_pattern'] = {
            'values': weekly_avg.tolist(),
            'peak_day': weekly_avg.idxmax(),
            'lowest_day': weekly_avg.idxmin(),
            'variation_coefficient': weekly_avg.std() / weekly_avg.mean() if weekly_avg.mean() > 0 else 0
        }

        # Monthly pattern
        monthly_avg = demand_data.groupby('month')['demand_indicator'].mean()
        patterns['monthly_pattern'] = {
            'values': monthly_avg.tolist(),
            'peak_month': monthly_avg.idxmax(),
            'lowest_month': monthly_avg.idxmin(),
            'seasonal_variation': (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() if monthly_avg.mean() > 0 else 0
        }

        # Overall trends
        demand_values = demand_data['demand_indicator'].values
        if len(demand_values) > 1:
            x = np.arange(len(demand_values))
            slope, _, r_value, _, _ = stats.linregress(x, demand_values)
            patterns['growth_rate'] = slope
            patterns['volatility'] = np.std(demand_values) / np.mean(demand_values) if np.mean(demand_values) > 0 else 0
            patterns['trend_strength'] = abs(r_value)

        return patterns

    def _parse_time_horizon(self, time_horizon: str) -> int:
        """Parse time horizon string to number of days"""
        if time_horizon.endswith('d'):
            return int(time_horizon[:-1])
        elif time_horizon.endswith('w'):
            return int(time_horizon[:-1]) * 7
        elif time_horizon.endswith('m'):
            return int(time_horizon[:-1]) * 30
        elif time_horizon.endswith('y'):
            return int(time_horizon[:-1]) * 365
        else:
            return int(time_horizon)  # Assume days

    async def _calculate_forecast_accuracy(self, historical_data: pd.DataFrame, forecast: List[float]) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        if len(historical_data) < len(forecast):
            return {'error': 'Insufficient data for accuracy calculation'}

        try:
            # Use last part of historical data as test set
            actual_values = historical_data['price_mean'].tail(len(forecast)).values

            mae = mean_absolute_error(actual_values, forecast)
            mse = mean_squared_error(actual_values, forecast)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_values - forecast) / actual_values)) * 100

            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r_squared': r2_score(actual_values, forecast)
            }

        except Exception as e:
            logger.error(f"Error calculating forecast accuracy: {e}")
            return {'error': str(e)}


# Factory function
def create_trend_predictor_agent(orchestrator) -> TrendPredictorAgent:
    """Create a trend predictor agent"""
    return TrendPredictorAgent(orchestrator)


# Usage example
if __name__ == "__main__":
    from multi_agent_orchestrator import MultiAgentOrchestrator

    async def test_trend_predictor():
        # Create orchestrator
        orchestrator = MultiAgentOrchestrator()
        await orchestrator.initialize()

        # Create trend predictor
        trend_agent = create_trend_predictor_agent(orchestrator)
        await trend_agent.initialize()

        # Test price trend prediction
        price_task = TaskRequest(
            task_id="trend_test_001",
            task_type="predict_price_trend",
            parameters={
                'product_category': 'bolts',
                'time_horizon': '30d',
                'confidence_level': 0.95
            },
            priority=Priority.NORMAL
        )

        result = await trend_agent.execute_delegated_task(price_task)
        print(f"Price trend prediction: {result.success}")

        # Test demand forecasting
        demand_task = TaskRequest(
            task_id="demand_test_001",
            task_type="forecast_demand",
            parameters={
                'product_category': 'fasteners',
                'time_horizon': '60d'
            },
            priority=Priority.NORMAL
        )

        result = await trend_agent.execute_delegated_task(demand_task)
        print(f"Demand forecasting: {result.success}")

    asyncio.run(test_trend_predictor())
#!/usr/bin/env python3
"""
ARIMA Model for Air Quality Time Series Prediction
Optimized for Colab execution with seasonal patterns
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AirQualityARIMA:
    """ARIMA model for air quality time series prediction"""
    
    def __init__(self, data_path="../../phase_1_streaming_infrastructure/processed_data/"):
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.results = {}
        self.target_columns = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)']
        
    def load_data(self):
        """Load and prepare time series data"""
        try:
            logger.info("🔄 Loading data for ARIMA modeling...")
            
            # Find all CSV files
            csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            
            if not csv_files:
                logger.error(f"❌ No data files found in {self.data_path}")
                return False
                
            # Load and concatenate all dataframes
            dataframes = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {file}: {e}")
                    
            if not dataframes:
                logger.error("❌ No data loaded successfully")
                return False
                
            # Concatenate all dataframes
            self.data = pd.concat(dataframes, ignore_index=True)
            
            # Create datetime index
            if 'timestamp' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['timestamp'])
            else:
                self.data['datetime'] = pd.date_range(start='2004-03-10', periods=len(self.data), freq='H')
            
            self.data.set_index('datetime', inplace=True)
            self.data = self.data.sort_index()
            
            # Clean infinity and extreme values
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            self.data = self.data.dropna()
            
            logger.info(f"✅ Loaded {len(self.data)} records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def check_stationarity(self, series, title):
        """Check if time series is stationary"""
        try:
            logger.info(f"🔍 Checking stationarity for {title}...")
            
            # Augmented Dickey-Fuller test
            result = adfuller(series.dropna())
            
            logger.info(f"ADF Statistic: {result[0]:.6f}")
            logger.info(f"p-value: {result[1]:.6f}")
            logger.info(f"Critical Values:")
            for key, value in result[4].items():
                logger.info(f"\t{key}: {value:.6f}")
            
            if result[1] <= 0.05:
                logger.info(f"✅ {title} is stationary (p-value <= 0.05)")
                return True
            else:
                logger.info(f"⚠️ {title} is not stationary (p-value > 0.05)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking stationarity: {e}")
            return False
    
    def make_stationary(self, series):
        """Make time series stationary using differencing"""
        try:
            # First difference
            diff1 = series.diff().dropna()
            
            if self.check_stationarity(diff1, "First Difference"):
                return diff1, 1
            
            # Second difference
            diff2 = diff1.diff().dropna()
            
            if self.check_stationarity(diff2, "Second Difference"):
                return diff2, 2
            
            # If still not stationary, return first difference
            logger.warning("⚠️ Series still not stationary after second difference")
            return diff1, 1
            
        except Exception as e:
            logger.error(f"❌ Error making stationary: {e}")
            return series, 0
    
    def find_arima_params(self, series, max_p=3, max_d=2, max_q=3):
        """Find optimal ARIMA parameters using grid search"""
        try:
            logger.info("🔍 Finding optimal ARIMA parameters...")
            
            best_aic = float('inf')
            best_params = None
            
            # Grid search
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                                
                        except:
                            continue
            
            logger.info(f"✅ Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
            return best_params
            
        except Exception as e:
            logger.error(f"❌ Error finding ARIMA params: {e}")
            return (1, 1, 1)  # Default fallback
    
    def train_arima_model(self, target_col):
        """Train ARIMA model for a specific target"""
        try:
            logger.info(f"🚀 Training ARIMA model for {target_col}...")
            
            if target_col not in self.data.columns:
                logger.warning(f"⚠️ {target_col} not found in data")
                return None
            
            # Get time series data
            series = self.data[target_col].dropna()
            
            if len(series) < 50:
                logger.warning(f"⚠️ Insufficient data for {target_col}")
                return None
            
            # Check stationarity
            if not self.check_stationarity(series, target_col):
                series, d = self.make_stationary(series)
            else:
                d = 0
            
            # Find optimal parameters
            params = self.find_arima_params(series)
            
            # Temporal split: 70% train, 30% test
            split_idx = int(len(series) * 0.7)
            train_series = series.iloc[:split_idx]
            test_series = series.iloc[split_idx:]
            
            # Train ARIMA model
            model = ARIMA(train_series, order=params)
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=len(test_series))
            
            # Calculate metrics
            mae = mean_absolute_error(test_series, forecast)
            rmse = np.sqrt(mean_squared_error(test_series, forecast))
            r2 = r2_score(test_series, forecast)
            
            logger.info(f"✅ {target_col}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
            
            return {
                'model': fitted_model,
                'params': params,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': forecast,
                'actual': test_series.values,
                'train_series': train_series,
                'test_series': test_series
            }
            
        except Exception as e:
            logger.error(f"❌ Error training ARIMA for {target_col}: {e}")
            return None
    
    def create_visualizations(self):
        """Create ARIMA model visualizations"""
        try:
            logger.info("📊 Creating ARIMA visualizations...")
            
            # Create results directory
            os.makedirs("../../phase_4_final_report/Images/Models", exist_ok=True)
            
            # 1. Model Performance Comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ARIMA Model Performance', fontsize=16, fontweight='bold')
            
            for i, (target, metrics) in enumerate(self.results.items()):
                if metrics is None:
                    continue
                    
                row, col = i // 2, i % 2
                ax = axes[row, col]
                
                # Scatter plot: Actual vs Predicted
                ax.scatter(metrics['actual'], metrics['predictions'], alpha=0.6, s=20)
                ax.plot([metrics['actual'].min(), metrics['actual'].max()], 
                       [metrics['actual'].min(), metrics['actual'].max()], 'r--', lw=2)
                
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{target}\nMAE: {metrics["mae"]:.3f}, RMSE: {metrics["rmse"]:.3f}, R²: {metrics["r2"]:.3f}')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('Images/Models/arima_performance.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Time Series Plots with Predictions
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ARIMA Time Series Predictions', fontsize=16, fontweight='bold')
            
            for i, (target, metrics) in enumerate(self.results.items()):
                if metrics is None:
                    continue
                    
                row, col = i // 2, i % 2
                ax = axes[row, col]
                
                # Plot training data
                ax.plot(metrics['train_series'].index, metrics['train_series'].values, 
                       label='Training Data', alpha=0.7)
                
                # Plot test data
                ax.plot(metrics['test_series'].index, metrics['test_series'].values, 
                       label='Actual Test', alpha=0.7)
                
                # Plot predictions
                ax.plot(metrics['test_series'].index, metrics['predictions'], 
                       label='ARIMA Predictions', alpha=0.7)
                
                ax.set_title(f'{target} - ARIMA({metrics["params"][0]},{metrics["params"][1]},{metrics["params"][2]})')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('Images/Models/arima_timeseries.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ ARIMA visualizations created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {e}")
            return False
    
    def run_analysis(self):
        """Run complete ARIMA analysis"""
        logger.info("🚀 Starting ARIMA Analysis")
        logger.info("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Train models for each target
        for target in self.target_columns:
            self.results[target] = self.train_arima_model(target)
        
        # Create visualizations
        if not self.create_visualizations():
            return False
        
        logger.info("✅ ARIMA analysis completed successfully!")
        return True

def main():
    """Main function to run ARIMA analysis"""
    model = AirQualityARIMA()
    model.run_analysis()

if __name__ == "__main__":
    main()

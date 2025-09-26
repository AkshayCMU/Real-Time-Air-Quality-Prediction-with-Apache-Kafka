"""
ARIMAX Model for Air Quality Prediction with Feature Store Integration
Quick implementation to get feature importance
"""

import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

logger = logging.getLogger(__name__)

class AirQualityARIMAX:
    """
    ARIMAX model for air quality prediction with feature store integration
    """
    
    def __init__(self, data_path: str = "../../phase_1_streaming_infrastructure/processed_data/"):
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.top_features = {}
        
    def load_data(self):
        """Load processed data for ARIMAX modeling"""
        try:
            logger.info("🔄 Loading data for ARIMAX modeling...")
            
            # Load all processed CSV files
            csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No data files found in {self.data_path}")
            
            dataframes = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {file}: {e}")
            
            if not dataframes:
                raise ValueError("No data loaded successfully")
            
            self.data = pd.concat(dataframes, ignore_index=True)
            
            # Clean data
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            self.data = self.data.dropna()
            
            # Convert timestamp
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                self.data = self.data.sort_values('timestamp')
            
            logger.info(f"✅ Loaded {len(self.data)} records for ARIMAX")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def prepare_arimax_features(self, target_col: str):
        """Prepare features for ARIMAX model"""
        logger.info(f"🔧 Preparing ARIMAX features for {target_col}...")
        
        # Get target series
        target_series = self.data[target_col].values
        
        # Select top features based on correlation with target
        feature_cols = [col for col in self.data.columns 
                       if col not in ['timestamp', 'sensor_id', 'location', target_col] 
                       and self.data[col].dtype in ['int64', 'float64']]
        
        # Calculate correlations with target
        correlations = {}
        for col in feature_cols:
            try:
                corr = self.data[target_col].corr(self.data[col])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                continue
        
        # Select top 20 features
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:20]
        top_feature_names = [f[0] for f in top_features]
        
        logger.info(f"✅ Selected {len(top_feature_names)} features for {target_col}")
        
        # Prepare exogenous features
        exog_features = self.data[top_feature_names].values
        
        # Clean exogenous features
        exog_features = np.nan_to_num(exog_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return target_series, exog_features, top_feature_names
    
    def train_arimax_model(self, target_col: str, order: tuple = (2, 1, 2)):
        """Train ARIMAX model for a specific target"""
        try:
            logger.info(f"🚀 Training ARIMAX model for {target_col}...")
            
            # Prepare features
            target_series, exog_features, feature_names = self.prepare_arimax_features(target_col)
            
            # Split data (80% train, 20% test)
            split_idx = int(len(target_series) * 0.8)
            train_target = target_series[:split_idx]
            test_target = target_series[split_idx:]
            train_exog = exog_features[:split_idx]
            test_exog = exog_features[split_idx:]
            
            # Train ARIMAX model
            model = SARIMAX(train_target, 
                           exog=train_exog,
                           order=order,
                           seasonal_order=(0, 0, 0, 0),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            
            fitted_model = model.fit(disp=False)
            
            # Make predictions
            predictions = fitted_model.forecast(steps=len(test_target), exog=test_exog)
            
            # Calculate metrics
            mae = mean_absolute_error(test_target, predictions)
            rmse = np.sqrt(mean_squared_error(test_target, predictions))
            r2 = r2_score(test_target, predictions)
            
            # Store results
            self.results[target_col] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'actual': test_target,
                'predictions': predictions,
                'model': fitted_model,
                'features': feature_names
            }
            
            # Get feature importance from model coefficients
            feature_importance = {}
            if hasattr(fitted_model, 'params'):
                # Get exogenous variable coefficients
                exog_params = fitted_model.params[len(order):]  # Skip AR/MA params
                for i, feature in enumerate(feature_names):
                    if i < len(exog_params):
                        feature_importance[feature] = abs(exog_params[i])
            
            self.feature_importance[target_col] = feature_importance
            self.top_features[target_col] = feature_names
            
            logger.info(f"✅ {target_col}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training ARIMAX for {target_col}: {e}")
            return False
    
    def run_arimax_analysis(self):
        """Run ARIMAX analysis for all targets"""
        logger.info("🚀 Starting ARIMAX Analysis")
        logger.info("=" * 60)
        
        if not self.load_data():
            return False
        
        targets = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)']
        
        for target in targets:
            if target in self.data.columns:
                self.train_arimax_model(target)
            else:
                logger.warning(f"⚠️ Target {target} not found in data")
        
        # Create visualizations
        self.create_arimax_visualizations()
        
        logger.info("✅ ARIMAX analysis completed successfully!")
        return True
    
    def create_arimax_visualizations(self):
        """Create ARIMAX visualizations"""
        try:
            logger.info("📊 Creating ARIMAX visualizations...")
            
            # Create output directory
            os.makedirs("Images/Models/", exist_ok=True)
            
            # 1. Model Performance Comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ARIMAX Model Performance', fontsize=16, fontweight='bold')
            
            for i, (target, metrics) in enumerate(self.results.items()):
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
            plt.savefig('Images/Models/arimax_performance.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Feature Importance for each target
            for target, importance in self.feature_importance.items():
                if importance:
                    plt.figure(figsize=(12, 8))
                    
                    # Sort features by importance
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    features, scores = zip(*sorted_features)
                    
                    # Create bar plot
                    plt.barh(range(len(features)), scores)
                    plt.yticks(range(len(features)), features)
                    plt.xlabel('Feature Importance (|Coefficient|)')
                    plt.title(f'ARIMAX Feature Importance - {target}')
                    plt.tight_layout()
                    
                    plt.savefig(f'Images/Models/arimax_features_{target.replace("(", "").replace(")", "").replace("GT", "")}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info("✅ ARIMAX visualizations created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {e}")
            return False
    
    def get_top_features_summary(self):
        """Get summary of top features across all targets"""
        logger.info("📊 ARIMAX Feature Importance Summary:")
        logger.info("=" * 50)
        
        for target, features in self.top_features.items():
            logger.info(f"\n{target} - Top 10 Features:")
            for i, feature in enumerate(features[:10], 1):
                importance = self.feature_importance[target].get(feature, 0)
                logger.info(f"  {i:2d}. {feature:<30} (importance: {importance:.4f})")
        
        # Find common features across targets
        all_features = set()
        for features in self.top_features.values():
            all_features.update(features)
        
        logger.info(f"\n📈 Total unique features used: {len(all_features)}")
        
        return self.top_features, self.feature_importance

def main():
    """Main function to run ARIMAX analysis"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize ARIMAX model
    arimax = AirQualityARIMAX()
    
    # Run analysis
    success = arimax.run_arimax_analysis()
    
    if success:
        # Get feature importance summary
        arimax.get_top_features_summary()
        
        logger.info("🎉 ARIMAX analysis completed!")
    else:
        logger.error("❌ ARIMAX analysis failed!")

if __name__ == "__main__":
    main()

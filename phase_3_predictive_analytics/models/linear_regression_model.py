#!/usr/bin/env python3
"""
Linear Regression Model for Air Quality Prediction
Optimized for Colab execution with temporal features
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import logging
from typing import Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from proper_feature_store import AirQualityFeatureStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AirQualityLinearRegression:
    """Linear Regression model for air quality prediction"""
    
    def __init__(self, data_path="../../phase_1_streaming_infrastructure/processed_data/"):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_columns = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)']
        self.results = {}
        self.feature_store = AirQualityFeatureStore()
        
    def load_data(self):
        """Load data using feature store or fallback to processed data"""
        try:
            logger.info("🔄 Loading data using feature store...")
            
            # Try feature store first
            feature_store = AirQualityFeatureStore()
            self.data = feature_store.prepare_training_data(self.data_path)
            
            if self.data is not None:
                logger.info(f"✅ Loaded {len(self.data)} records using feature store")
                return True
            else:
                logger.warning("⚠️ Feature store failed, using direct data loading")
                return self._load_direct_data()
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return self._load_direct_data()
    
    def _load_direct_data(self):
        """Fallback: Load data directly from CSV files"""
        try:
            logger.info("🔄 Loading data directly from CSV files...")
            
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
            logger.info(f"✅ Loaded {len(self.data)} records directly")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def prepare_features(self):
        """Use existing engineered features from processed data"""
        try:
            logger.info("🔧 Preparing existing features...")
            
            # Clean infinity and extreme values
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            self.data = self.data.dropna()
            
            # Remove extreme outliers (beyond 5 standard deviations) for feature columns only
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in self.target_columns]
            
            for col in feature_cols:
                mean = self.data[col].mean()
                std = self.data[col].std()
                if std > 0:  # Avoid division by zero
                    self.data = self.data[np.abs(self.data[col] - mean) <= 5 * std]
            
            logger.info(f"✅ Feature preparation completed. Shape: {self.data.shape}")
            logger.info(f"📊 Available features: {len([col for col in self.data.columns if col not in self.target_columns])}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in feature preparation: {e}")
            return False
    
    def prepare_model_data(self):
        """Prepare features and targets for modeling"""
        try:
            logger.info("📊 Preparing model data...")
            
            # Select feature columns (exclude target columns and non-numeric)
            feature_cols = [col for col in self.data.columns 
                          if col not in self.target_columns and 
                          self.data[col].dtype in ['int64', 'float64'] and
                          not any(target in col for target in self.target_columns)]
            
            self.feature_columns = feature_cols
            X = self.data[feature_cols]
            y = self.data[self.target_columns]
            
            logger.info(f"✅ Features: {len(feature_cols)}, Targets: {len(self.target_columns)}")
            logger.info(f"📊 Data shape: X={X.shape}, y={y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error preparing model data: {e}")
            return None, None
    
    def train_model(self, X, y):
        """Train linear regression model"""
        try:
            logger.info("🚀 Training Linear Regression model...")
            
            # Temporal split: 70% train, 30% test
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics for each target
            results = {}
            for i, target in enumerate(self.target_columns):
                mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
                rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
                r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
                
                results[target] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred[:, i],
                    'actual': y_test.iloc[:, i].values
                }
                
                logger.info(f"✅ {target}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
            
            self.results = results
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")
            return False
    
    def create_visualizations(self):
        """Create model performance visualizations"""
        try:
            logger.info("📊 Creating visualizations...")
            
            # Create results directory
            os.makedirs("../../phase_4_final_report/Images/Models", exist_ok=True)
            
            # 1. Model Performance Comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Linear Regression Model Performance', fontsize=16, fontweight='bold')
            
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
            plt.savefig('Images/Models/linear_regression_performance.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Feature Importance
            if hasattr(self.model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': np.abs(self.model.coef_[0])  # Use first target for importance
                }).sort_values('importance', ascending=False).head(20)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(data=feature_importance, x='importance', y='feature')
                plt.title('Top 20 Feature Importance (Linear Regression)', fontsize=14, fontweight='bold')
                plt.xlabel('Absolute Coefficient Value')
                plt.tight_layout()
                plt.savefig('Images/Models/linear_regression_features.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("✅ Visualizations created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {e}")
            return False
    
    def predict_real_time(self, sensor_id: str, location: str, timestamp: datetime) -> Dict:
        """
        Real-time prediction using professional feature store
        """
        try:
            logger.info(f"🔮 Making real-time prediction for {sensor_id} at {location}")
            
            # Get features from feature store
            feature_response = self.feature_store.get_features_for_prediction(sensor_id, location, timestamp)
            
            # Extract feature values from the structured response
            feature_values = {}
            for feature_name, feature_data in feature_response['features'].items():
                feature_values[feature_name] = feature_data['value']
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([feature_values])
            
            # Ensure we have the right features
            missing_features = set(self.feature_columns) - set(feature_df.columns)
            if missing_features:
                logger.warning(f"⚠️ Missing features: {missing_features}")
                for feature in missing_features:
                    feature_df[feature] = 0.0
            
            # Select only the features used in training
            X = feature_df[self.feature_columns]
            
            # Make prediction
            predictions = self.model.predict(X)
            
            # Create result with feature store metadata
            result = {
                'sensor_id': sensor_id,
                'location': location,
                'timestamp': timestamp.isoformat(),
                'entity_key': feature_response['entity_key'],
                'predictions': {
                    'CO(GT)': float(predictions[0][0]),
                    'NOx(GT)': float(predictions[0][1]),
                    'C6H6(GT)': float(predictions[0][2]),
                    'NO2(GT)': float(predictions[0][3])
                },
                'feature_store_metadata': {
                    'feature_count': feature_response['feature_count'],
                    'version': feature_response['feature_store_version'],
                    'retrieved_at': feature_response['retrieved_at']
                },
                'features_used': len(self.feature_columns),
                'feature_store_stats': self.feature_store.get_feature_metadata()
            }
            
            logger.info(f"✅ Real-time prediction completed: {result['predictions']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in real-time prediction: {e}")
            return {
                'error': str(e),
                'sensor_id': sensor_id,
                'location': location,
                'timestamp': timestamp.isoformat()
            }
    
    def run_analysis(self):
        """Run complete linear regression analysis"""
        logger.info("🚀 Starting Linear Regression Analysis")
        logger.info("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Prepare existing features
        if not self.prepare_features():
            return False
        
        # Prepare model data
        X, y = self.prepare_model_data()
        if X is None:
            return False
        
        # Train model
        if not self.train_model(X, y):
            return False
        
        # Create visualizations
        if not self.create_visualizations():
            return False
        
        logger.info("✅ Linear Regression analysis completed successfully!")
        return True

def main():
    """Main function to run linear regression analysis"""
    model = AirQualityLinearRegression()
    model.run_analysis()

if __name__ == "__main__":
    main()

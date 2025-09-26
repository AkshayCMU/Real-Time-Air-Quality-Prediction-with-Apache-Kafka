"""
Data Preprocessing Utilities for Air Quality Data
Phase 1: Streaming Infrastructure and Data Pipeline Architecture

This module provides comprehensive data preprocessing utilities for
the UCI Air Quality dataset, including missing value handling,
data validation, and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AirQualityDataPreprocessor:
    """
    Comprehensive data preprocessor for air quality sensor data.
    
    Features:
    - Missing value handling for sensor malfunctions
    - Data validation and quality assessment
    - Feature engineering and temporal encoding
    - Outlier detection and treatment
    - Data quality scoring
    """
    
    def __init__(self, missing_value_threshold: float = -200.0):
        """
        Initialize the data preprocessor.
        
        Args:
            missing_value_threshold: Threshold for identifying missing values
        """
        self.missing_value_threshold = missing_value_threshold
        
        # Define sensor specifications and quality thresholds
        self.sensor_specs = {
            'CO(GT)': {
                'unit': 'mg/m³',
                'normal_range': (0.0, 50.0),
                'missing_threshold': missing_value_threshold,
                'description': 'Carbon Monoxide'
            },
            'PT08.S1(CO)': {
                'unit': 'mg/m³',
                'normal_range': (0.0, 1000.0),
                'missing_threshold': missing_value_threshold,
                'description': 'CO Sensor Response'
            },
            'NMHC(GT)': {
                'unit': 'mg/m³',
                'normal_range': (0.0, 1000.0),
                'missing_threshold': missing_value_threshold,
                'description': 'Non-methane Hydrocarbons'
            },
            'C6H6(GT)': {
                'unit': 'µg/m³',
                'normal_range': (0.0, 100.0),
                'missing_threshold': missing_value_threshold,
                'description': 'Benzene'
            },
            'PT08.S2(NMHC)': {
                'unit': 'mg/m³',
                'normal_range': (0.0, 1000.0),
                'missing_threshold': missing_value_threshold,
                'description': 'NMHC Sensor Response'
            },
            'NOx(GT)': {
                'unit': 'ppb',
                'normal_range': (0.0, 1000.0),
                'missing_threshold': missing_value_threshold,
                'description': 'Nitrogen Oxides'
            },
            'PT08.S3(NOx)': {
                'unit': 'mg/m³',
                'normal_range': (0.0, 1000.0),
                'missing_threshold': missing_value_threshold,
                'description': 'NOx Sensor Response'
            },
            'NO2(GT)': {
                'unit': 'µg/m³',
                'normal_range': (0.0, 500.0),
                'missing_threshold': missing_value_threshold,
                'description': 'Nitrogen Dioxide'
            },
            'PT08.S4(NO2)': {
                'unit': 'mg/m³',
                'normal_range': (0.0, 1000.0),
                'missing_threshold': missing_value_threshold,
                'description': 'NO2 Sensor Response'
            },
            'PT08.S5(O3)': {
                'unit': 'mg/m³',
                'normal_range': (0.0, 1000.0),
                'missing_threshold': missing_value_threshold,
                'description': 'Ozone Sensor Response'
            },
            'T': {
                'unit': '°C',
                'normal_range': (-20.0, 50.0),
                'missing_threshold': missing_value_threshold,
                'description': 'Temperature'
            },
            'RH': {
                'unit': '%',
                'normal_range': (0.0, 100.0),
                'missing_threshold': missing_value_threshold,
                'description': 'Relative Humidity'
            },
            'AH': {
                'unit': 'g/m³',
                'normal_range': (0.0, 50.0),
                'missing_threshold': missing_value_threshold,
                'description': 'Absolute Humidity'
            }
        }
        
        logger.info("AirQualityDataPreprocessor initialized")
    
    def load_air_quality_data(self, file_path: str) -> pd.DataFrame:
        """
        Load air quality data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading data from: {file_path}")
            
            # Load CSV with proper separator (UCI dataset uses semicolon)
            df = pd.read_csv(file_path, sep=';')
            
            # Remove any unnamed columns that might be created
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Handle European number format (comma as decimal separator)
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                if col not in ['Date', 'Time']:
                    try:
                        # Try to convert string numbers with comma decimal separator
                        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                        logger.debug(f"Converted {col} from European format to float")
                    except (ValueError, TypeError):
                        # If conversion fails, keep as string
                        logger.debug(f"Could not convert {col} to float, keeping as string")
            
            logger.info(f"Data loaded successfully: {len(df)} records, {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create datetime index from Date and Time columns.
        
        Args:
            df: DataFrame with Date and Time columns
            
        Returns:
            DataFrame with datetime index
        """
        try:
            df_processed = df.copy()
            
            # Check if Date and Time columns exist
            if 'Date' in df_processed.columns and 'Time' in df_processed.columns:
                # Combine Date and Time columns
                df_processed['datetime'] = pd.to_datetime(
                    df_processed['Date'] + ' ' + df_processed['Time'],
                    format='%d/%m/%Y %H.%M.%S',
                    errors='coerce'
                )
                
                # Set datetime as index
                df_processed.set_index('datetime', inplace=True)
                
                # Drop original Date and Time columns
                df_processed.drop(['Date', 'Time'], axis=1, inplace=True)
                
                logger.info("Datetime index created successfully")
                
            else:
                # Create synthetic datetime if columns don't exist
                logger.warning("Date/Time columns not found, creating synthetic datetime")
                start_date = datetime(2004, 3, 1)
                df_processed['datetime'] = pd.date_range(
                    start=start_date,
                    periods=len(df_processed),
                    freq='H'
                )
                df_processed.set_index('datetime', inplace=True)
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error creating datetime index: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            method: str = 'hybrid') -> pd.DataFrame:
        """
        Handle missing values using specified method.
        
        Args:
            df: DataFrame to process
            method: Method for handling missing values ('forward', 'backward', 'interpolate', 'hybrid')
            
        Returns:
            DataFrame with missing values handled
        """
        try:
            df_processed = df.copy()
            missing_stats = {}
            
            logger.info(f"Handling missing values using method: {method}")
            
            for column in df_processed.columns:
                if column in self.sensor_specs:
                    # Count missing values before processing
                    missing_before = df_processed[column].isna().sum()
                    
                    # Replace threshold values with NaN
                    threshold = self.sensor_specs[column]['missing_threshold']
                    df_processed[column] = df_processed[column].replace(threshold, np.nan)
                    
                    # Count missing values after threshold replacement
                    missing_after = df_processed[column].isna().sum()
                    
                    # Apply missing value handling method
                    if method == 'forward':
                        df_processed[column] = df_processed[column].fillna(method='ffill')
                    elif method == 'backward':
                        df_processed[column] = df_processed[column].fillna(method='bfill')
                    elif method == 'interpolate':
                        df_processed[column] = df_processed[column].interpolate(method='linear')
                    elif method == 'hybrid':
                        # Forward fill with limit
                        df_processed[column] = df_processed[column].fillna(method='ffill', limit=3)
                        # Backward fill with limit
                        df_processed[column] = df_processed[column].fillna(method='bfill', limit=3)
                        # Fill remaining with median
                        if df_processed[column].isna().any():
                            median_val = df_processed[column].median()
                            df_processed[column] = df_processed[column].fillna(median_val)
                            logger.warning(f"Filled remaining missing values in {column} with median: {median_val:.2f}")
                    
                    # Count final missing values
                    missing_final = df_processed[column].isna().sum()
                    
                    missing_stats[column] = {
                        'before': missing_before,
                        'after_threshold': missing_after,
                        'final': missing_final,
                        'replaced_threshold': missing_after - missing_before
                    }
            
            # Log missing value statistics
            logger.info("Missing value handling completed:")
            for column, stats in missing_stats.items():
                logger.info(f"  {column}: {stats['before']} -> {stats['final']} missing values")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and generate quality report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            quality_report = {
                'overall_quality_score': 0.0,
                'sensor_quality': {},
                'data_completeness': 0.0,
                'range_violations': {},
                'outlier_percentage': 0.0,
                'temporal_consistency': 0.0
            }
            
            sensor_scores = []
            total_records = len(df)
            
            logger.info("Validating data quality...")
            
            for column in df.columns:
                if column in self.sensor_specs:
                    sensor_data = df[column].dropna()
                    spec = self.sensor_specs[column]
                    
                    # Calculate completeness
                    completeness = len(sensor_data) / total_records
                    
                    # Check range violations
                    normal_min, normal_max = spec['normal_range']
                    range_violations = ((sensor_data < normal_min) | (sensor_data > normal_max)).sum()
                    range_violation_rate = range_violations / len(sensor_data) if len(sensor_data) > 0 else 0
                    
                    # Calculate outlier percentage (using IQR method)
                    if len(sensor_data) > 4:
                        Q1 = sensor_data.quantile(0.25)
                        Q3 = sensor_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = ((sensor_data < lower_bound) | (sensor_data > upper_bound)).sum()
                        outlier_percentage = outliers / len(sensor_data) * 100
                    else:
                        outlier_percentage = 0
                    
                    # Calculate sensor quality score
                    quality_score = (
                        completeness * 0.4 +  # 40% weight on completeness
                        (1 - range_violation_rate) * 0.4 +  # 40% weight on range compliance
                        (1 - outlier_percentage / 100) * 0.2  # 20% weight on outlier rate
                    )
                    
                    sensor_scores.append(quality_score)
                    
                    quality_report['sensor_quality'][column] = {
                        'completeness': completeness,
                        'range_violation_rate': range_violation_rate,
                        'outlier_percentage': outlier_percentage,
                        'quality_score': quality_score,
                        'valid_records': len(sensor_data),
                        'total_records': total_records
                    }
                    
                    quality_report['range_violations'][column] = {
                        'count': int(range_violations),
                        'rate': range_violation_rate,
                        'normal_range': spec['normal_range']
                    }
            
            # Calculate overall quality score
            if sensor_scores:
                quality_report['overall_quality_score'] = np.mean(sensor_scores)
                quality_report['data_completeness'] = np.mean([q['completeness'] for q in quality_report['sensor_quality'].values()])
                quality_report['outlier_percentage'] = np.mean([q['outlier_percentage'] for q in quality_report['sensor_quality'].values()])
            
            # Check temporal consistency (no large gaps in datetime index)
            if hasattr(df.index, 'to_series'):
                time_diffs = df.index.to_series().diff().dropna()
                expected_freq = pd.Timedelta(hours=1)  # Expected hourly data
                temporal_consistency = (time_diffs == expected_freq).mean()
                quality_report['temporal_consistency'] = temporal_consistency
            
            logger.info(f"Data quality validation completed:")
            logger.info(f"  Overall quality score: {quality_report['overall_quality_score']:.3f}")
            logger.info(f"  Data completeness: {quality_report['data_completeness']:.3f}")
            logger.info(f"  Average outlier percentage: {quality_report['outlier_percentage']:.2f}%")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer temporal and statistical features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df_engineered = df.copy()
            
            logger.info("Engineering features...")
            
            # Temporal features
            if hasattr(df_engineered.index, 'hour'):
                df_engineered['hour'] = df_engineered.index.hour
                df_engineered['day_of_week'] = df_engineered.index.dayofweek
                df_engineered['month'] = df_engineered.index.month
                df_engineered['day_of_year'] = df_engineered.index.dayofyear
                df_engineered['week_of_year'] = df_engineered.index.isocalendar().week
                
                # Cyclical encoding for temporal features
                df_engineered['hour_sin'] = np.sin(2 * np.pi * df_engineered['hour'] / 24)
                df_engineered['hour_cos'] = np.cos(2 * np.pi * df_engineered['hour'] / 24)
                df_engineered['day_sin'] = np.sin(2 * np.pi * df_engineered['day_of_week'] / 7)
                df_engineered['day_cos'] = np.cos(2 * np.pi * df_engineered['day_of_week'] / 7)
                df_engineered['month_sin'] = np.sin(2 * np.pi * df_engineered['month'] / 12)
                df_engineered['month_cos'] = np.cos(2 * np.pi * df_engineered['month'] / 12)
                
                # Season encoding
                df_engineered['season'] = df_engineered['month'].map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'
                })
                
                # Weekend indicator
                df_engineered['is_weekend'] = (df_engineered['day_of_week'] >= 5).astype(int)
                
                # Rush hour indicators
                df_engineered['is_rush_hour_morning'] = ((df_engineered['hour'] >= 7) & (df_engineered['hour'] <= 9)).astype(int)
                df_engineered['is_rush_hour_evening'] = ((df_engineered['hour'] >= 17) & (df_engineered['hour'] <= 19)).astype(int)
            
            # Statistical features for each sensor (simplified - only most useful)
            for column in df_engineered.columns:
                if column in self.sensor_specs and column not in ['hour', 'day_of_week', 'month', 'day_of_year', 'week_of_year']:
                    # Rolling window statistics (simplified - only mean and std)
                    for window in [3, 12, 24]:  # 3h, 12h, 24h windows (reduced from 4)
                        df_engineered[f'{column}_ma_{window}h'] = df_engineered[column].rolling(window=window, min_periods=1).mean()
                        df_engineered[f'{column}_std_{window}h'] = df_engineered[column].rolling(window=window, min_periods=1).std()
                    
                    # Lagged features (simplified - only most important lags)
                    for lag in [1, 3, 12, 24]:  # 1h, 3h, 12h, 24h lags (reduced from 6)
                        df_engineered[f'{column}_lag_{lag}h'] = df_engineered[column].shift(lag)
                    
                    # Difference features (keep these - they're useful)
                    df_engineered[f'{column}_diff_1h'] = df_engineered[column].diff(1)
                    df_engineered[f'{column}_diff_24h'] = df_engineered[column].diff(24)
                    
                    # Percentage change (keep these - they're useful)
                    df_engineered[f'{column}_pct_change_1h'] = df_engineered[column].pct_change(1)
                    df_engineered[f'{column}_pct_change_24h'] = df_engineered[column].pct_change(24)
            
            logger.info(f"Feature engineering completed: {len(df_engineered.columns)} total features")
            
            return df_engineered
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    def preprocess_pipeline(self, file_path: str, 
                          missing_value_method: str = 'hybrid',
                          engineer_features: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path: Path to the CSV file
            missing_value_method: Method for handling missing values
            engineer_features: Whether to engineer additional features
            
        Returns:
            Tuple of (processed DataFrame, quality report)
        """
        try:
            logger.info("Starting complete preprocessing pipeline...")
            
            # Load data
            df = self.load_air_quality_data(file_path)
            
            # Create datetime index
            df = self.create_datetime_index(df)
            
            # Handle missing values
            df = self.handle_missing_values(df, method=missing_value_method)
            
            # Validate data quality
            quality_report = self.validate_data_quality(df)
            
            # Engineer features
            if engineer_features:
                df = self.engineer_features(df)
            
            logger.info("Preprocessing pipeline completed successfully")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            raise


def main():
    """Main function for testing the preprocessor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Air Quality Data Preprocessor')
    parser.add_argument('--input-file', required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output-file', default='processed_air_quality_data.csv',
                       help='Path to output CSV file')
    parser.add_argument('--missing-method', default='hybrid',
                       choices=['forward', 'backward', 'interpolate', 'hybrid'],
                       help='Method for handling missing values')
    parser.add_argument('--no-features', action='store_true',
                       help='Skip feature engineering')
    
    args = parser.parse_args()
    
    try:
        # Create preprocessor
        preprocessor = AirQualityDataPreprocessor()
        
        # Run preprocessing pipeline
        df_processed, quality_report = preprocessor.preprocess_pipeline(
            file_path=args.input_file,
            missing_value_method=args.missing_method,
            engineer_features=not args.no_features
        )
        
        # Save processed data
        df_processed.to_csv(args.output_file)
        logger.info(f"Processed data saved to: {args.output_file}")
        
        # Print quality report
        print("\n" + "="*50)
        print("DATA QUALITY REPORT")
        print("="*50)
        print(f"Overall Quality Score: {quality_report['overall_quality_score']:.3f}")
        print(f"Data Completeness: {quality_report['data_completeness']:.3f}")
        print(f"Average Outlier Percentage: {quality_report['outlier_percentage']:.2f}%")
        print(f"Temporal Consistency: {quality_report['temporal_consistency']:.3f}")
        
        print("\nSensor Quality Details:")
        for sensor, metrics in quality_report['sensor_quality'].items():
            print(f"  {sensor}: {metrics['quality_score']:.3f} "
                  f"(completeness: {metrics['completeness']:.3f}, "
                  f"outliers: {metrics['outlier_percentage']:.1f}%)")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

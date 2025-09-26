"""
Professional Feature Store Implementation for Air Quality Prediction
Showcases enterprise-grade feature management with entities, feature views, and metadata
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature types for validation and metadata"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    SENSOR = "sensor"
    ENVIRONMENTAL = "environmental"

@dataclass
class FeatureDefinition:
    """Feature definition with metadata"""
    name: str
    feature_type: FeatureType
    description: str
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_required: bool = True

@dataclass
class Entity:
    """Entity definition for feature store"""
    name: str
    description: str
    data_type: str

class AirQualityFeatureStore:
    """
    Professional Feature Store Implementation
    - Entity-based feature management
    - Feature metadata and validation
    - Real-time feature serving
    - Feature lineage and versioning
    """
    
    def __init__(self, data_path: str = "../../phase_1_streaming_infrastructure/processed_data/"):
        self.data_path = data_path
        self.feature_cache = {}
        self.feature_definitions = self._define_features()
        self.entities = self._define_entities()
        self.feature_views = self._define_feature_views()
        self.historical_data = None
        self.feature_store_metadata = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_features': len(self.feature_definitions),
            'total_entities': len(self.entities)
        }
        
        # Load historical data
        self._load_historical_data()
        
    def _define_entities(self) -> Dict[str, Entity]:
        """Define entities for the feature store"""
        return {
            'sensor_id': Entity(
                name='sensor_id',
                description='Unique sensor identifier',
                data_type='string'
            ),
            'location': Entity(
                name='location',
                description='Geographic location of sensor',
                data_type='string'
            ),
            'timestamp': Entity(
                name='timestamp',
                description='Event timestamp',
                data_type='datetime'
            )
        }
    
    def _define_features(self) -> Dict[str, FeatureDefinition]:
        """Define the 20 selected features with metadata"""
        return {
            'PT08.S1(CO)_ma_3h': FeatureDefinition(
                name='PT08.S1(CO)_ma_3h',
                feature_type=FeatureType.SENSOR,
                description='3-hour moving average of CO sensor readings',
                unit='mg/m³',
                min_value=0.0,
                max_value=50.0,
                is_required=True
            ),
            'PT08.S4(NO2)_ma_3h': FeatureDefinition(
                name='PT08.S4(NO2)_ma_3h',
                feature_type=FeatureType.SENSOR,
                description='3-hour moving average of NO2 sensor readings',
                unit='μg/m³',
                min_value=0.0,
                max_value=500.0,
                is_required=True
            ),
            'PT08.S2(NMHC)_pct_change_24h': FeatureDefinition(
                name='PT08.S2(NMHC)_pct_change_24h',
                feature_type=FeatureType.SENSOR,
                description='24-hour percentage change in NMHC readings',
                unit='%',
                min_value=-100.0,
                max_value=1000.0,
                is_required=True
            ),
            'PT08.S2(NMHC)_diff_24h': FeatureDefinition(
                name='PT08.S2(NMHC)_diff_24h',
                feature_type=FeatureType.SENSOR,
                description='24-hour difference in NMHC readings',
                unit='mg/m³',
                min_value=-50.0,
                max_value=50.0,
                is_required=True
            ),
            'PT08.S1(CO)_lag_1h': FeatureDefinition(
                name='PT08.S1(CO)_lag_1h',
                feature_type=FeatureType.SENSOR,
                description='1-hour lagged CO sensor reading',
                unit='mg/m³',
                min_value=0.0,
                max_value=50.0,
                is_required=True
            ),
            'T_ma_3h': FeatureDefinition(
                name='T_ma_3h',
                feature_type=FeatureType.ENVIRONMENTAL,
                description='3-hour moving average temperature',
                unit='°C',
                min_value=-20.0,
                max_value=50.0,
                is_required=True
            ),
            'PT08.S5(O3)_ma_3h': FeatureDefinition(
                name='PT08.S5(O3)_ma_3h',
                feature_type=FeatureType.SENSOR,
                description='3-hour moving average of Ozone readings',
                unit='μg/m³',
                min_value=0.0,
                max_value=300.0,
                is_required=True
            ),
            'AH_ma_24h': FeatureDefinition(
                name='AH_ma_24h',
                feature_type=FeatureType.ENVIRONMENTAL,
                description='24-hour moving average absolute humidity',
                unit='g/m³',
                min_value=0.0,
                max_value=50.0,
                is_required=True
            ),
            'RH_ma_3h': FeatureDefinition(
                name='RH_ma_3h',
                feature_type=FeatureType.ENVIRONMENTAL,
                description='3-hour moving average relative humidity',
                unit='%',
                min_value=0.0,
                max_value=100.0,
                is_required=True
            ),
            'PT08.S4(NO2)_ma_24h': FeatureDefinition(
                name='PT08.S4(NO2)_ma_24h',
                feature_type=FeatureType.SENSOR,
                description='24-hour moving average NO2 readings',
                unit='μg/m³',
                min_value=0.0,
                max_value=500.0,
                is_required=True
            ),
            'T_ma_12h': FeatureDefinition(
                name='T_ma_12h',
                feature_type=FeatureType.ENVIRONMENTAL,
                description='12-hour moving average temperature',
                unit='°C',
                min_value=-20.0,
                max_value=50.0,
                is_required=True
            ),
            'PT08.S1(CO)_pct_change_1h': FeatureDefinition(
                name='PT08.S1(CO)_pct_change_1h',
                feature_type=FeatureType.SENSOR,
                description='1-hour percentage change in CO readings',
                unit='%',
                min_value=-100.0,
                max_value=1000.0,
                is_required=True
            ),
            'PT08.S2(NMHC)_ma_3h': FeatureDefinition(
                name='PT08.S2(NMHC)_ma_3h',
                feature_type=FeatureType.SENSOR,
                description='3-hour moving average NMHC readings',
                unit='mg/m³',
                min_value=0.0,
                max_value=50.0,
                is_required=True
            ),
            'PT08.S3(NOx)_ma_3h': FeatureDefinition(
                name='PT08.S3(NOx)_ma_3h',
                feature_type=FeatureType.SENSOR,
                description='3-hour moving average NOx readings',
                unit='μg/m³',
                min_value=0.0,
                max_value=500.0,
                is_required=True
            ),
            'T_lag_1h': FeatureDefinition(
                name='T_lag_1h',
                feature_type=FeatureType.ENVIRONMENTAL,
                description='1-hour lagged temperature',
                unit='°C',
                min_value=-20.0,
                max_value=50.0,
                is_required=True
            ),
            'RH_lag_1h': FeatureDefinition(
                name='RH_lag_1h',
                feature_type=FeatureType.ENVIRONMENTAL,
                description='1-hour lagged relative humidity',
                unit='%',
                min_value=0.0,
                max_value=100.0,
                is_required=True
            ),
            'AH_lag_1h': FeatureDefinition(
                name='AH_lag_1h',
                feature_type=FeatureType.ENVIRONMENTAL,
                description='1-hour lagged absolute humidity',
                unit='g/m³',
                min_value=0.0,
                max_value=50.0,
                is_required=True
            ),
            'PT08.S4(NO2)_lag_1h': FeatureDefinition(
                name='PT08.S4(NO2)_lag_1h',
                feature_type=FeatureType.SENSOR,
                description='1-hour lagged NO2 reading',
                unit='μg/m³',
                min_value=0.0,
                max_value=500.0,
                is_required=True
            ),
            'PT08.S5(O3)_lag_1h': FeatureDefinition(
                name='PT08.S5(O3)_lag_1h',
                feature_type=FeatureType.SENSOR,
                description='1-hour lagged Ozone reading',
                unit='μg/m³',
                min_value=0.0,
                max_value=300.0,
                is_required=True
            ),
            'data_quality_score': FeatureDefinition(
                name='data_quality_score',
                feature_type=FeatureType.NUMERICAL,
                description='Data quality score (0-1)',
                unit='score',
                min_value=0.0,
                max_value=1.0,
                is_required=True
            )
        }
    
    def _define_feature_views(self) -> Dict[str, Dict]:
        """Define feature views for different use cases"""
        return {
            'core_sensor_features': {
                'description': 'Core sensor readings and their transformations',
                'features': [
                    'PT08.S1(CO)_ma_3h', 'PT08.S4(NO2)_ma_3h', 'PT08.S5(O3)_ma_3h',
                    'PT08.S2(NMHC)_ma_3h', 'PT08.S3(NOx)_ma_3h'
                ]
            },
            'environmental_features': {
                'description': 'Environmental conditions and their trends',
                'features': [
                    'T_ma_3h', 'T_ma_12h', 'T_lag_1h',
                    'RH_ma_3h', 'RH_lag_1h', 'AH_ma_24h', 'AH_lag_1h'
                ]
            },
            'temporal_features': {
                'description': 'Time-based feature transformations',
                'features': [
                    'PT08.S1(CO)_pct_change_1h', 'PT08.S2(NMHC)_pct_change_24h',
                    'PT08.S2(NMHC)_diff_24h', 'PT08.S1(CO)_lag_1h',
                    'PT08.S4(NO2)_lag_1h', 'PT08.S5(O3)_lag_1h'
                ]
            },
            'quality_indicators': {
                'description': 'Data quality and reliability indicators',
                'features': ['data_quality_score']
            }
        }
    
    def _load_historical_data(self):
        """Load historical data for feature computation"""
        logger.info("🔄 Loading historical data for feature store...")
        
        import glob
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        if not csv_files:
            logger.error(f"❌ No data files found in {self.data_path}")
            self.historical_data = pd.DataFrame()
            return
        
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file, parse_dates=['timestamp'], index_col='timestamp')
                dataframes.append(df)
            except Exception as e:
                logger.error(f"❌ Error loading {file}: {e}")
        
        if not dataframes:
            logger.error("❌ No data loaded successfully")
            self.historical_data = pd.DataFrame()
            return
        
        self.historical_data = pd.concat(dataframes).sort_index()
        self.historical_data = self.historical_data.apply(pd.to_numeric, errors='coerce')
        self.historical_data = self.historical_data.replace([np.inf, -np.inf], np.nan)
        self.historical_data = self.historical_data.dropna(how='all')
        
        logger.info(f"✅ Loaded {len(self.historical_data)} historical records")
        logger.info(f"📊 Feature store initialized with {len(self.feature_definitions)} features")
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get feature store metadata"""
        return {
            'feature_store_metadata': self.feature_store_metadata,
            'entities': {name: {'description': entity.description, 'data_type': entity.data_type} 
                        for name, entity in self.entities.items()},
            'features': {name: {
                'type': feature.feature_type.value,
                'description': feature.description,
                'unit': feature.unit,
                'range': f"{feature.min_value}-{feature.max_value}" if feature.min_value and feature.max_value else None,
                'required': feature.is_required
            } for name, feature in self.feature_definitions.items()},
            'feature_views': self.feature_views,
            'data_statistics': {
                'total_records': len(self.historical_data),
                'feature_count': len(self.feature_definitions),
                'time_range': {
                    'start': self.historical_data.index.min().isoformat() if not self.historical_data.empty else None,
                    'end': self.historical_data.index.max().isoformat() if not self.historical_data.empty else None
                }
            }
        }
    
    def get_features_for_prediction(self, sensor_id: str, location: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Get features for real-time prediction with entity structure
        """
        logger.info(f"🔍 Getting features for entity: {sensor_id}@{location} at {timestamp}")
        
        # Filter data for the specific entity
        entity_data = self.historical_data[
            (self.historical_data['sensor_id'] == sensor_id) & 
            (self.historical_data['location'] == location)
        ].sort_index()
        
        if entity_data.empty:
            logger.warning(f"⚠️ No data for entity: {sensor_id}@{location}")
            return self._get_default_features(sensor_id, location, timestamp)
        
        # Get most recent data before timestamp
        recent_data = entity_data[entity_data.index <= timestamp].tail(1)
        
        if recent_data.empty:
            logger.warning(f"⚠️ No recent data before {timestamp}")
            return self._get_default_features(sensor_id, location, timestamp)
        
        # Build feature vector with entity structure
        feature_vector = {
            # Entity keys
            'sensor_id': sensor_id,
            'location': location,
            'timestamp': timestamp.isoformat(),
            'entity_key': f"{sensor_id}@{location}",
            
            # Feature metadata
            'feature_count': len(self.feature_definitions),
            'feature_store_version': self.feature_store_metadata['version'],
            'retrieved_at': datetime.now().isoformat(),
            
            # Actual features
            'features': {}
        }
        
        # Extract features with validation
        for feature_name, feature_def in self.feature_definitions.items():
            if feature_name in recent_data.columns:
                value = recent_data[feature_name].iloc[0]
                
                # Validate feature value
                if self._validate_feature_value(feature_name, value):
                    feature_vector['features'][feature_name] = {
                        'value': float(value),
                        'type': feature_def.feature_type.value,
                        'unit': feature_def.unit,
                        'description': feature_def.description
                    }
                else:
                    logger.warning(f"⚠️ Invalid value for {feature_name}: {value}")
                    feature_vector['features'][feature_name] = {
                        'value': 0.0,
                        'type': feature_def.feature_type.value,
                        'unit': feature_def.unit,
                        'description': feature_def.description,
                        'status': 'default_value'
                    }
            else:
                logger.warning(f"⚠️ Feature {feature_name} not found in data")
                feature_vector['features'][feature_name] = {
                    'value': 0.0,
                    'type': feature_def.feature_type.value,
                    'unit': feature_def.unit,
                    'description': feature_def.description,
                    'status': 'missing'
                }
        
        logger.info(f"✅ Retrieved {len(feature_vector['features'])} features for {sensor_id}@{location}")
        return feature_vector
    
    def _validate_feature_value(self, feature_name: str, value: Any) -> bool:
        """Validate feature value against definition"""
        if pd.isna(value) or np.isinf(value):
            return False
        
        feature_def = self.feature_definitions[feature_name]
        
        # Check range if defined
        if feature_def.min_value is not None and value < feature_def.min_value:
            return False
        if feature_def.max_value is not None and value > feature_def.max_value:
            return False
        
        return True
    
    def _get_default_features(self, sensor_id: str, location: str, timestamp: datetime) -> Dict[str, Any]:
        """Get default features when no data is available"""
        logger.info(f"🔄 Using default features for {sensor_id}@{location}")
        
        feature_vector = {
            'sensor_id': sensor_id,
            'location': location,
            'timestamp': timestamp.isoformat(),
            'entity_key': f"{sensor_id}@{location}",
            'feature_count': len(self.feature_definitions),
            'feature_store_version': self.feature_store_metadata['version'],
            'retrieved_at': datetime.now().isoformat(),
            'features': {}
        }
        
        for feature_name, feature_def in self.feature_definitions.items():
            feature_vector['features'][feature_name] = {
                'value': 0.0,
                'type': feature_def.feature_type.value,
                'unit': feature_def.unit,
                'description': feature_def.description,
                'status': 'default_value'
            }
        
        return feature_vector
    
    def get_feature_view(self, view_name: str) -> Dict[str, Any]:
        """Get a specific feature view"""
        if view_name not in self.feature_views:
            raise ValueError(f"Feature view '{view_name}' not found")
        
        view = self.feature_views[view_name]
        return {
            'view_name': view_name,
            'description': view['description'],
            'features': view['features'],
            'feature_count': len(view['features'])
        }
    
    def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get feature lineage and dependencies"""
        if feature_name not in self.feature_definitions:
            raise ValueError(f"Feature '{feature_name}' not found")
        
        feature_def = self.feature_definitions[feature_name]
        return {
            'feature_name': feature_name,
            'type': feature_def.feature_type.value,
            'description': feature_def.description,
            'unit': feature_def.unit,
            'range': f"{feature_def.min_value}-{feature_def.max_value}" if feature_def.min_value and feature_def.max_value else None,
            'required': feature_def.is_required,
            'created_at': self.feature_store_metadata['created_at'],
            'last_updated': self.feature_store_metadata['last_updated']
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize feature store
    feature_store = AirQualityFeatureStore()
    
    # Get metadata
    metadata = feature_store.get_feature_metadata()
    print("🏪 Feature Store Metadata:")
    print(json.dumps(metadata, indent=2))
    
    # Test feature retrieval
    test_timestamp = datetime(2004, 3, 10, 18, 0, 0)
    test_sensor_id = 'air_quality_sensor_001'
    test_location = 'urban_monitoring_station'
    
    features = feature_store.get_features_for_prediction(
        test_sensor_id, test_location, test_timestamp
    )
    
    print(f"\n🔍 Features for {test_sensor_id}@{test_location}:")
    print(json.dumps(features, indent=2))

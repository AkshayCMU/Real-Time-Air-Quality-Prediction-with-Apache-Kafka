"""
Initialize Feast feature store for air quality prediction
"""

import os
import pandas as pd
from feast import FeatureStore
from feast.repo_config import RepoConfig
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
    PostgreSQLOfflineStoreConfig,
)
from feast.infra.online_stores.redis import RedisOnlineStoreConfig
import logging

logger = logging.getLogger(__name__)

def initialize_feast_store():
    """
    Initialize Feast feature store with proper configuration
    """
    logger.info("🚀 Initializing Feast feature store...")
    
    # Create data directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/offline_store", exist_ok=True)
    os.makedirs("data/online_store", exist_ok=True)
    
    # Create feature store configuration
    config = RepoConfig(
        project="air_quality_feature_store",
        registry="data/registry.db",
        provider="local",
        offline_store=PostgreSQLOfflineStoreConfig(
            type="postgres",
            host="localhost",
            port=5432,
            database="air_quality",
            user="postgres",
            password="password"
        ),
        online_store=RedisOnlineStoreConfig(
            type="redis",
            connection_string="redis://localhost:6379"
        )
    )
    
    # Save configuration
    with open("feature_store.yaml", "w") as f:
        f.write(f"""
project: {config.project}
registry: {config.registry}
provider: {config.provider}
offline_store:
  type: {config.offline_store.type}
  host: {config.offline_store.host}
  port: {config.offline_store.port}
  database: {config.offline_store.database}
  user: {config.offline_store.user}
  password: {config.offline_store.password}
online_store:
  type: {config.online_store.type}
  connection_string: {config.online_store.connection_string}
""")
    
    logger.info("✅ Feast feature store initialized")
    return config

def create_sample_data():
    """
    Create sample data for testing the feature store
    """
    logger.info("📊 Creating sample data...")
    
    # Create sample entity data
    entity_data = pd.DataFrame({
        'sensor_id': ['sensor_1', 'sensor_2', 'sensor_3'] * 100,
        'location': ['downtown', 'suburb', 'industrial'] * 100,
        'event_timestamp': pd.date_range('2024-01-01', periods=300, freq='H'),
        'created_timestamp': pd.Timestamp.now()
    })
    
    # Create sample feature data
    feature_data = pd.DataFrame({
        'sensor_id': ['sensor_1', 'sensor_2', 'sensor_3'] * 100,
        'location': ['downtown', 'suburb', 'industrial'] * 100,
        'event_timestamp': pd.date_range('2024-01-01', periods=300, freq='H'),
        'CO_GT': [2.5, 3.1, 4.2] * 100,
        'NOx_GT': [150, 200, 300] * 100,
        'C6H6_GT': [5.2, 6.8, 8.1] * 100,
        'NO2_GT': [45, 55, 65] * 100,
        'T': [20, 22, 25] * 100,
        'RH': [60, 65, 70] * 100,
        'AH': [12, 14, 16] * 100,
        'hour': [0, 1, 2] * 100,
        'day_of_week': [0, 1, 2] * 100,
        'month': [1, 2, 3] * 100,
        'season': [0, 0, 1] * 100,
    })
    
    # Save sample data
    entity_data.to_csv("data/sample_entities.csv", index=False)
    feature_data.to_csv("data/sample_features.csv", index=False)
    
    logger.info("✅ Sample data created")
    return entity_data, feature_data

def main():
    """
    Main function to initialize Feast feature store
    """
    logging.basicConfig(level=logging.INFO)
    
    # Initialize feature store
    config = initialize_feast_store()
    
    # Create sample data
    entity_data, feature_data = create_sample_data()
    
    logger.info("🎉 Feast feature store setup completed!")
    logger.info(f"📁 Data directory: {os.path.abspath('data')}")
    logger.info(f"📊 Sample entities: {len(entity_data)}")
    logger.info(f"📊 Sample features: {len(feature_data)}")

if __name__ == "__main__":
    main()

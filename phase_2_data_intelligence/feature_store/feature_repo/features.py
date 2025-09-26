"""
Feast Feature Definitions for Air Quality Data Intelligence
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field
from feast.types import Float64, Int64, String
from feast.data_source import FileSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
    PostgreSQLOfflineStoreConfig,
)
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig

# Define the entity (primary key)
air_quality_entity = Entity(
    name="sensor_id",
    description="Unique identifier for air quality sensor",
    join_keys=["sensor_id"],
)

# Define the timestamp entity
timestamp_entity = Entity(
    name="timestamp",
    description="Timestamp for the measurement",
    join_keys=["timestamp"],
)

# Define the location entity
location_entity = Entity(
    name="location",
    description="Geographic location of the sensor",
    join_keys=["location"],
)

# Define the data source
air_quality_source = FileSource(
    name="air_quality_data",
    path="../../data/processed_data/",  # Path to processed data from Phase 1
    timestamp_field="timestamp",
    created_timestamp_column="created_at",
)

# Define feature views for different aspects of air quality data

# Core sensor measurements
core_measurements = FeatureView(
    name="core_measurements",
    description="Core air quality sensor measurements",
    entities=[air_quality_entity, timestamp_entity, location_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="CO_GT", dtype=Float64),
        Field(name="PT08_S1_CO", dtype=Float64),
        Field(name="NMHC_GT", dtype=Float64),
        Field(name="C6H6_GT", dtype=Float64),
        Field(name="PT08_S2_NMHC", dtype=Float64),
        Field(name="NOx_GT", dtype=Float64),
        Field(name="PT08_S3_NOx", dtype=Float64),
        Field(name="NO2_GT", dtype=Float64),
        Field(name="PT08_S4_NO2", dtype=Float64),
        Field(name="PT08_S5_O3", dtype=Float64),
        Field(name="T", dtype=Float64),
        Field(name="RH", dtype=Float64),
        Field(name="AH", dtype=Float64),
    ],
    source=air_quality_source,
)

# Engineered features - temporal
temporal_features = FeatureView(
    name="temporal_features",
    description="Temporal features engineered from air quality data",
    entities=[air_quality_entity, timestamp_entity, location_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="hour", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="season", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="hour_sin", dtype=Float64),
        Field(name="hour_cos", dtype=Float64),
        Field(name="day_sin", dtype=Float64),
        Field(name="day_cos", dtype=Float64),
        Field(name="month_sin", dtype=Float64),
        Field(name="month_cos", dtype=Float64),
    ],
    source=air_quality_source,
)

# Engineered features - lagged
lagged_features = FeatureView(
    name="lagged_features",
    description="Lagged features for temporal dependencies",
    entities=[air_quality_entity, timestamp_entity, location_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="CO_GT_lag_1h", dtype=Float64),
        Field(name="CO_GT_lag_3h", dtype=Float64),
        Field(name="CO_GT_lag_12h", dtype=Float64),
        Field(name="CO_GT_lag_24h", dtype=Float64),
        Field(name="NOx_GT_lag_1h", dtype=Float64),
        Field(name="NOx_GT_lag_3h", dtype=Float64),
        Field(name="NOx_GT_lag_12h", dtype=Float64),
        Field(name="NOx_GT_lag_24h", dtype=Float64),
        Field(name="C6H6_GT_lag_1h", dtype=Float64),
        Field(name="C6H6_GT_lag_3h", dtype=Float64),
        Field(name="C6H6_GT_lag_12h", dtype=Float64),
        Field(name="C6H6_GT_lag_24h", dtype=Float64),
    ],
    source=air_quality_source,
)

# Engineered features - rolling statistics
rolling_features = FeatureView(
    name="rolling_features",
    description="Rolling window statistics for temporal patterns",
    entities=[air_quality_entity, timestamp_entity, location_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="CO_GT_rolling_mean_3h", dtype=Float64),
        Field(name="CO_GT_rolling_std_3h", dtype=Float64),
        Field(name="CO_GT_rolling_mean_12h", dtype=Float64),
        Field(name="CO_GT_rolling_std_12h", dtype=Float64),
        Field(name="CO_GT_rolling_mean_24h", dtype=Float64),
        Field(name="CO_GT_rolling_std_24h", dtype=Float64),
        Field(name="NOx_GT_rolling_mean_3h", dtype=Float64),
        Field(name="NOx_GT_rolling_std_3h", dtype=Float64),
        Field(name="NOx_GT_rolling_mean_12h", dtype=Float64),
        Field(name="NOx_GT_rolling_std_12h", dtype=Float64),
        Field(name="NOx_GT_rolling_mean_24h", dtype=Float64),
        Field(name="NOx_GT_rolling_std_24h", dtype=Float64),
        Field(name="C6H6_GT_rolling_mean_3h", dtype=Float64),
        Field(name="C6H6_GT_rolling_std_3h", dtype=Float64),
        Field(name="C6H6_GT_rolling_mean_12h", dtype=Float64),
        Field(name="C6H6_GT_rolling_std_12h", dtype=Float64),
        Field(name="C6H6_GT_rolling_mean_24h", dtype=Float64),
        Field(name="C6H6_GT_rolling_std_24h", dtype=Float64),
    ],
    source=air_quality_source,
)

# Engineered features - environmental
environmental_features = FeatureView(
    name="environmental_features",
    description="Environmental and weather-related features",
    entities=[air_quality_entity, timestamp_entity, location_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="temperature", dtype=Float64),
        Field(name="relative_humidity", dtype=Float64),
        Field(name="absolute_humidity", dtype=Float64),
        Field(name="temp_humidity_ratio", dtype=Float64),
        Field(name="humidity_trend", dtype=Float64),
        Field(name="temperature_trend", dtype=Float64),
    ],
    source=air_quality_source,
)

# Engineered features - pollutant ratios
pollutant_ratios = FeatureView(
    name="pollutant_ratios",
    description="Ratios and interactions between different pollutants",
    entities=[air_quality_entity, timestamp_entity, location_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="CO_NOx_ratio", dtype=Float64),
        Field(name="NOx_NO2_ratio", dtype=Float64),
        Field(name="CO_C6H6_ratio", dtype=Float64),
        Field(name="total_pollutants", dtype=Float64),
        Field(name="pollutant_diversity", dtype=Float64),
    ],
    source=air_quality_source,
)

# Engineered features - quality indicators
quality_indicators = FeatureView(
    name="quality_indicators",
    description="Data quality and reliability indicators",
    entities=[air_quality_entity, timestamp_entity, location_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="missing_data_count", dtype=Int64),
        Field(name="data_completeness", dtype=Float64),
        Field(name="sensor_reliability", dtype=Float64),
        Field(name="measurement_consistency", dtype=Float64),
    ],
    source=air_quality_source,
)

"""
Feast Feature Store for Air Quality Prediction
Defines entities, feature views, and data sources
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field
from feast.types import Float64, Int64, String
import pandas as pd

# Define entities
sensor_entity = Entity(
    name="sensor_id",
    description="Unique identifier for air quality sensor",
    join_keys=["sensor_id"]
)

location_entity = Entity(
    name="location", 
    description="Geographic location of the sensor",
    join_keys=["location"]
)

timestamp_entity = Entity(
    name="event_timestamp",
    description="Timestamp of the measurement",
    join_keys=["event_timestamp"]
)

# Define feature views for different feature categories

# Core measurements feature view
core_measurements_fv = FeatureView(
    name="core_measurements_fv",
    description="Core air quality measurements",
    entities=[sensor_entity, location_entity, timestamp_entity],
    ttl=timedelta(days=1),
    features=[
        Field(name="CO_GT", dtype=Float64),
        Field(name="PT08_S1_CO", dtype=Float64),
        Field(name="NMHC_GT", dtype=Float64),
        Field(name="C6H6_GT", dtype=Float64),
        Field(name="PT08_S2_NMHC", dtype=Float64),
        Field(name="NOx_GT", dtype=Float64),
        Field(name="NO2_GT", dtype=Float64),
        Field(name="PT08_S3_NOx", dtype=Float64),
        Field(name="PT08_S4_NO2", dtype=Float64),
        Field(name="PT08_S5_O3", dtype=Float64),
    ]
)

# Temporal features feature view
temporal_features_fv = FeatureView(
    name="temporal_features_fv",
    description="Temporal and cyclical features",
    entities=[sensor_entity, location_entity, timestamp_entity],
    ttl=timedelta(days=1),
    features=[
        Field(name="hour", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="season", dtype=Int64),
        Field(name="hour_sin", dtype=Float64),
        Field(name="hour_cos", dtype=Float64),
        Field(name="day_sin", dtype=Float64),
        Field(name="day_cos", dtype=Float64),
        Field(name="month_sin", dtype=Float64),
        Field(name="month_cos", dtype=Float64),
    ]
)

# Lagged features feature view
lagged_features_fv = FeatureView(
    name="lagged_features_fv",
    description="Historical values (lagged features)",
    entities=[sensor_entity, location_entity, timestamp_entity],
    ttl=timedelta(days=1),
    features=[
        Field(name="CO_GT_lag_1h", dtype=Float64),
        Field(name="CO_GT_lag_3h", dtype=Float64),
        Field(name="CO_GT_lag_6h", dtype=Float64),
        Field(name="CO_GT_lag_12h", dtype=Float64),
        Field(name="NOx_GT_lag_1h", dtype=Float64),
        Field(name="NOx_GT_lag_3h", dtype=Float64),
        Field(name="C6H6_GT_lag_1h", dtype=Float64),
        Field(name="C6H6_GT_lag_3h", dtype=Float64),
        Field(name="NO2_GT_lag_1h", dtype=Float64),
        Field(name="NO2_GT_lag_3h", dtype=Float64),
    ]
)

# Rolling window features feature view
rolling_features_fv = FeatureView(
    name="rolling_features_fv",
    description="Rolling window statistics",
    entities=[sensor_entity, location_entity, timestamp_entity],
    ttl=timedelta(days=1),
    features=[
        Field(name="CO_GT_ma_3h", dtype=Float64),
        Field(name="CO_GT_ma_6h", dtype=Float64),
        Field(name="CO_GT_ma_12h", dtype=Float64),
        Field(name="CO_GT_ma_24h", dtype=Float64),
        Field(name="NOx_GT_ma_3h", dtype=Float64),
        Field(name="NOx_GT_ma_6h", dtype=Float64),
        Field(name="C6H6_GT_ma_3h", dtype=Float64),
        Field(name="C6H6_GT_ma_6h", dtype=Float64),
        Field(name="NO2_GT_ma_3h", dtype=Float64),
        Field(name="NO2_GT_ma_6h", dtype=Float64),
    ]
)

# Environmental features feature view
environmental_features_fv = FeatureView(
    name="environmental_features_fv",
    description="Environmental conditions",
    entities=[sensor_entity, location_entity, timestamp_entity],
    ttl=timedelta(days=1),
    features=[
        Field(name="T", dtype=Float64),
        Field(name="RH", dtype=Float64),
        Field(name="AH", dtype=Float64),
        Field(name="T_ma_3h", dtype=Float64),
        Field(name="T_ma_6h", dtype=Float64),
        Field(name="T_ma_12h", dtype=Float64),
        Field(name="T_ma_24h", dtype=Float64),
        Field(name="RH_ma_3h", dtype=Float64),
        Field(name="RH_ma_6h", dtype=Float64),
        Field(name="AH_ma_3h", dtype=Float64),
        Field(name="AH_ma_6h", dtype=Float64),
        Field(name="AH_ma_12h", dtype=Float64),
        Field(name="AH_ma_24h", dtype=Float64),
    ]
)

# Pollutant ratios feature view
pollutant_ratios_fv = FeatureView(
    name="pollutant_ratios_fv",
    description="Pollutant concentration ratios",
    entities=[sensor_entity, location_entity, timestamp_entity],
    ttl=timedelta(days=1),
    features=[
        Field(name="CO_NOx_ratio", dtype=Float64),
        Field(name="NO2_NOx_ratio", dtype=Float64),
        Field(name="C6H6_CO_ratio", dtype=Float64),
        Field(name="NMHC_CO_ratio", dtype=Float64),
        Field(name="O3_NO2_ratio", dtype=Float64),
    ]
)

# Quality indicators feature view
quality_indicators_fv = FeatureView(
    name="quality_indicators_fv",
    description="Data quality and reliability indicators",
    entities=[sensor_entity, location_entity, timestamp_entity],
    ttl=timedelta(days=1),
    features=[
        Field(name="data_quality_score", dtype=Float64),
        Field(name="missing_data_count", dtype=Int64),
        Field(name="sensor_reliability", dtype=Float64),
        Field(name="measurement_confidence", dtype=Float64),
    ]
)

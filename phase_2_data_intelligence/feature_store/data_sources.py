"""
Data sources for Feast feature store
"""

from feast import FileSource
from feast.data_source import DataSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
    PostgreSQLOfflineStoreConfig,
)
from feast.infra.online_stores.redis import RedisOnlineStoreConfig
from feast.repo_config import RepoConfig
from feast.types import Float64, Int64, String
import pandas as pd

# Define data source for processed air quality data
air_quality_source = FileSource(
    name="air_quality_source",
    path="../../processed_data/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define data source for raw sensor data
sensor_data_source = FileSource(
    name="sensor_data_source", 
    path="../../processed_data/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define data source for environmental data
environmental_data_source = FileSource(
    name="environmental_data_source",
    path="../../processed_data/",
    timestamp_field="event_timestamp", 
    created_timestamp_column="created_timestamp",
)

# Define data source for temporal features
temporal_data_source = FileSource(
    name="temporal_data_source",
    path="../../processed_data/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define data source for lagged features
lagged_data_source = FileSource(
    name="lagged_data_source",
    path="../../processed_data/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define data source for rolling features
rolling_data_source = FileSource(
    name="rolling_data_source",
    path="../../processed_data/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define data source for pollutant ratios
ratios_data_source = FileSource(
    name="ratios_data_source",
    path="../../processed_data/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define data source for quality indicators
quality_data_source = FileSource(
    name="quality_data_source",
    path="../../processed_data/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

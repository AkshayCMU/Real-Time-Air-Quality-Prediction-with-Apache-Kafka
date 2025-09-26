"""
Real-Time Air Quality Data Producer for Apache Kafka
Phase 1: Streaming Infrastructure and Data Pipeline Architecture

This module implements a production-grade Kafka producer for ingesting
UCI Air Quality dataset with realistic temporal simulation and comprehensive
error handling.
"""

import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os
import sys
from pathlib import Path

# Confluent Kafka imports
from confluent_kafka import Producer
from confluent_kafka import KafkaError, KafkaException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kafka_producer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AirQualityDataProducer:
    """
    Production-grade Kafka producer for air quality sensor data.
    
    Features:
    - Realistic temporal simulation of sensor data
    - Comprehensive error handling and retry logic
    - Data quality validation and preprocessing
    - Structured logging and monitoring
    """
    
    def __init__(self, 
                 bootstrap_servers: str = 'localhost:9092',
                 topic_name: str = 'air-quality-data',
                 data_file_path: str = None):
        """
        Initialize the Air Quality Data Producer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic_name: Target Kafka topic for air quality data
            data_file_path: Path to UCI Air Quality dataset CSV file
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.data_file_path = data_file_path or self._get_default_data_path()
        
        # Initialize Kafka producer with production settings
        self.producer = self._create_producer()
        
        # Load and preprocess data
        self.air_quality_data = self._load_and_preprocess_data()
        self.current_index = 0
        
        logger.info(f"AirQualityDataProducer initialized for topic: {topic_name}")
    
    def _get_default_data_path(self) -> str:
        """Get default path to air quality dataset."""
        project_root = Path(__file__).parent.parent
        
        # Check for the actual UCI dataset file first
        uci_file = project_root / "data" / "AirQualityUCI.csv"
        if uci_file.exists():
            logger.info(f"Found UCI dataset: {uci_file}")
            return str(uci_file)
        
        # Fallback to the expected name
        expected_file = project_root / "data" / "air_quality_data.csv"
        if expected_file.exists():
            logger.info(f"Found expected dataset: {expected_file}")
            return str(expected_file)
        
        # If neither exists, return the expected path (will trigger sample data creation)
        logger.warning("No dataset found, will create sample data")
        return str(expected_file)
    
    def _create_producer(self) -> Producer:
        """Create Confluent Kafka producer with production-grade configuration."""
        try:
            producer_config = {
                'bootstrap.servers': self.bootstrap_servers,
                'acks': 'all',  # Wait for all replicas to acknowledge
                'retries': 3,   # Retry failed sends
                'retry.backoff.ms': 100,
                'request.timeout.ms': 30000,
                'compression.type': 'gzip',  # Compress messages
                'batch.size': 16384,  # Batch messages for efficiency
                'linger.ms': 10,  # Wait up to 10ms to batch messages
                'enable.idempotence': True,  # Ensure exactly-once delivery
                'max.in.flight.requests.per.connection': 5,
                'delivery.timeout.ms': 120000,
            }
            
            producer = Producer(producer_config)
            logger.info("Confluent Kafka producer created successfully")
            return producer
            
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            raise
    
    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the UCI Air Quality dataset with full feature engineering.
        
        Returns:
            Preprocessed DataFrame with 177 engineered features
        """
        try:
            logger.info(f"Loading data from: {self.data_file_path}")
            
            # Check if data file exists
            if not os.path.exists(self.data_file_path):
                logger.warning(f"Data file not found: {self.data_file_path}")
                logger.info("Creating sample data for demonstration...")
                return self._create_sample_data()
            
            # Use the full preprocessing pipeline with feature engineering
            from data_preprocessing import AirQualityDataPreprocessor
            preprocessor = AirQualityDataPreprocessor()
            
            # Load and preprocess with full feature engineering
            df = preprocessor.preprocess_pipeline(self.data_file_path)
            
            logger.info(f"Data loaded successfully: {len(df)} records with {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            logger.info("Creating sample data for demonstration...")
            return self._create_sample_data()
    
    def _preprocess_air_quality_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess air quality data with comprehensive cleaning.
        
        Args:
            df: Raw air quality DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Convert date and time columns
        if 'Date' in df_clean.columns and 'Time' in df_clean.columns:
            df_clean['datetime'] = pd.to_datetime(
                df_clean['Date'] + ' ' + df_clean['Time'], 
                format='%d/%m/%Y %H.%M.%S',
                errors='coerce'
            )
        else:
            # Create synthetic datetime if not available
            start_date = datetime(2004, 3, 1)
            df_clean['datetime'] = pd.date_range(
                start=start_date, 
                periods=len(df_clean), 
                freq='H'
            )
        
        # Handle European number format (comma as decimal separator)
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        string_columns = df_clean.select_dtypes(include=['object']).columns
        
        # Convert string columns that contain numbers with comma decimal separator
        for col in string_columns:
            if col not in ['Date', 'Time', 'datetime']:
                try:
                    # Try to convert string numbers with comma decimal separator
                    df_clean[col] = df_clean[col].astype(str).str.replace(',', '.').astype(float)
                    logger.debug(f"Converted {col} from European format to float")
                except (ValueError, TypeError):
                    # If conversion fails, keep as string
                    logger.debug(f"Could not convert {col} to float, keeping as string")
        
        # Update numeric columns list after conversion
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        # Handle missing values (-200 indicates sensor malfunction)
        for col in numeric_columns:
            if col != 'datetime':
                # Replace -200 with NaN
                df_clean[col] = df_clean[col].replace(-200, np.nan)
                
                # Forward fill missing values (sensor recovery)
                df_clean[col] = df_clean[col].ffill(limit=3)
                
                # Backward fill remaining missing values
                df_clean[col] = df_clean[col].bfill(limit=3)
                
                # If still missing, use median
                if df_clean[col].isna().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    logger.warning(f"Filled remaining missing values in {col} with median: {median_val}")
        
        # Add derived features
        df_clean['hour'] = df_clean['datetime'].dt.hour
        df_clean['day_of_week'] = df_clean['datetime'].dt.dayofweek
        df_clean['month'] = df_clean['datetime'].dt.month
        df_clean['season'] = df_clean['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Ensure all numeric columns are float
        for col in numeric_columns:
            if col != 'datetime':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        logger.info("Data preprocessing completed successfully")
        return df_clean
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample air quality data for demonstration purposes.
        
        Returns:
            Sample DataFrame with realistic air quality patterns
        """
        logger.info("Creating sample air quality data...")
        
        # Generate 1000 hours of sample data
        n_samples = 1000
        start_date = datetime(2004, 3, 1)
        timestamps = pd.date_range(start=start_date, periods=n_samples, freq='H')
        
        # Create realistic air quality patterns
        np.random.seed(42)  # For reproducibility
        
        # Base patterns for different pollutants
        base_co = 2.0 + 1.5 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)  # Daily cycle
        base_nox = 50 + 30 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)  # Daily cycle
        base_benzene = 5 + 3 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)  # Daily cycle
        
        # Add noise and trends
        co_values = base_co + np.random.normal(0, 0.5, n_samples)
        nox_values = base_nox + np.random.normal(0, 10, n_samples)
        benzene_values = base_benzene + np.random.normal(0, 1, n_samples)
        
        # Ensure positive values
        co_values = np.maximum(co_values, 0.1)
        nox_values = np.maximum(nox_values, 1)
        benzene_values = np.maximum(benzene_values, 0.1)
        
        sample_data = pd.DataFrame({
            'datetime': timestamps,
            'CO(GT)': co_values,
            'PT08.S1(CO)': co_values * 100 + np.random.normal(0, 10, n_samples),
            'NMHC(GT)': np.random.normal(200, 50, n_samples),
            'C6H6(GT)': benzene_values,
            'PT08.S2(NMHC)': benzene_values * 50 + np.random.normal(0, 5, n_samples),
            'NOx(GT)': nox_values,
            'PT08.S3(NOx)': nox_values * 2 + np.random.normal(0, 5, n_samples),
            'NO2(GT)': nox_values * 0.6 + np.random.normal(0, 3, n_samples),
            'PT08.S4(NO2)': nox_values * 0.6 * 1.5 + np.random.normal(0, 2, n_samples),
            'PT08.S5(O3)': np.random.normal(100, 20, n_samples),
            'T': np.random.normal(20, 5, n_samples),  # Temperature
            'RH': np.random.normal(60, 15, n_samples),  # Relative humidity
            'AH': np.random.normal(1.0, 0.2, n_samples),  # Absolute humidity
            'hour': timestamps.hour,
            'day_of_week': timestamps.dayofweek,
            'month': timestamps.month,
            'season': timestamps.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        })
        
        logger.info(f"Sample data created: {len(sample_data)} records")
        return sample_data
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate message data quality for flat message format with 177 features.
        
        Args:
            message: Message dictionary to validate
            
        Returns:
            True if message is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ['timestamp', 'sensor_id', 'location']
            for field in required_fields:
                if field not in message:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check for key sensor data fields
            required_sensor_fields = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)']
            for field in required_sensor_fields:
                if field not in message:
                    logger.warning(f"Missing required sensor field: {field}")
                    return False
            
            # Check for reasonable value ranges
            if message['CO(GT)'] is not None and (message['CO(GT)'] < 0 or message['CO(GT)'] > 50):
                logger.warning(f"CO value out of range: {message['CO(GT)']}")
                return False
            
            if message['NOx(GT)'] is not None and (message['NOx(GT)'] < 0 or message['NOx(GT)'] > 1000):
                logger.warning(f"NOx value out of range: {message['NOx(GT)']}")
                return False
            
            if message['C6H6(GT)'] is not None and (message['C6H6(GT)'] < 0 or message['C6H6(GT)'] > 100):
                logger.warning(f"Benzene value out of range: {message['C6H6(GT)']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Message validation error: {e}")
            return False
    
    def _create_message(self, row: pd.Series) -> Dict[str, Any]:
        """
        Create a properly formatted message from a data row with all 177 features.
        
        Args:
            row: Pandas Series representing one data record with 177 features
            
        Returns:
            Formatted message dictionary with all engineered features
        """
        try:
            # Get datetime from index (since the DataFrame has datetime index after preprocessing)
            if hasattr(row, 'name') and row.name is not None:
                # Row has a datetime index
                datetime_str = row.name.isoformat() if hasattr(row.name, 'isoformat') else str(row.name)
            else:
                # Fallback to current time
                datetime_str = datetime.now().isoformat()
            
            # Create message with all features as a flat structure
            message = {
                'timestamp': datetime_str,
                'datetime': datetime_str,
                'sensor_id': 'air_quality_sensor_001',
                'location': 'urban_monitoring_station',
                'quality_score': 1.0,
                'processing_time': datetime.now().isoformat()
            }
            
            # Add all features from the row (210 features)
            for column in row.index:
                # Skip any datetime-related columns that might exist
                if column not in ['datetime', 'timestamp']:
                    try:
                        # Convert to appropriate type
                        value = row[column]
                        if pd.isna(value):
                            message[column] = None
                        elif isinstance(value, (int, float)):
                            message[column] = float(value)
                        else:
                            message[column] = str(value)
                    except Exception as e:
                        logger.warning(f"Error processing column {column}: {e}")
                        message[column] = None
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating message from row: {e}")
            return None
    
    def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a single message to Kafka topic.
        
        Args:
            message: Message dictionary to send
            
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            if not self._validate_message(message):
                logger.warning("Message validation failed, skipping send")
                return False
            
            # Create message key for partitioning
            message_key = f"{message['sensor_id']}_{message['timestamp']}"
            
            # Serialize message to JSON
            message_value = json.dumps(message).encode('utf-8')
            message_key_bytes = message_key.encode('utf-8')
            
            # Send message asynchronously
            self.producer.produce(
                topic=self.topic_name,
                value=message_value,
                key=message_key_bytes,
                callback=self._on_send_success
            )
            
            # Trigger delivery of queued messages
            self.producer.poll(0)
            
            return True
            
        except KafkaException as e:
            logger.error(f"Kafka error while sending message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while sending message: {e}")
            return False
    
    def _on_send_success(self, err, msg):
        """Callback for successful message send."""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message sent successfully to topic: {msg.topic()}, "
                        f"partition: {msg.partition()}, "
                        f"offset: {msg.offset()}")
    
    def _on_send_error(self, err, msg):
        """Callback for failed message send."""
        if err is not None:
            logger.error(f"Failed to send message: {err}")
    
    def start_streaming(self, 
                       simulation_speed: float = 1.0,
                       max_messages: Optional[int] = None,
                       batch_size: int = 10) -> None:
        """
        Start streaming air quality data to Kafka.
        
        Args:
            simulation_speed: Speed multiplier for time simulation (1.0 = real-time)
            max_messages: Maximum number of messages to send (None for all)
            batch_size: Number of messages to send in each batch
        """
        logger.info(f"Starting air quality data streaming...")
        logger.info(f"Simulation speed: {simulation_speed}x")
        logger.info(f"Max messages: {max_messages or 'All'}")
        logger.info(f"Batch size: {batch_size}")
        
        try:
            messages_sent = 0
            total_messages = min(len(self.air_quality_data), max_messages or len(self.air_quality_data))
            
            while self.current_index < total_messages:
                batch_messages = []
                
                # Prepare batch of messages
                for _ in range(min(batch_size, total_messages - self.current_index)):
                    if self.current_index >= len(self.air_quality_data):
                        break
                    
                    row = self.air_quality_data.iloc[self.current_index]
                    message = self._create_message(row)
                    
                    if message:
                        batch_messages.append(message)
                    
                    self.current_index += 1
                
                # Send batch
                for message in batch_messages:
                    if self.send_message(message):
                        messages_sent += 1
                
                # Log progress
                if messages_sent % 100 == 0:
                    logger.info(f"Messages sent: {messages_sent}/{total_messages}")
                
                # Simulate real-time delay
                if simulation_speed > 0:
                    time.sleep(1.0 / simulation_speed)
            
            # Flush remaining messages
            self.producer.flush()
            logger.info(f"Streaming completed. Total messages sent: {messages_sent}")
            
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
        finally:
            self.close()
    
    def close(self):
        """Close the Kafka producer."""
        try:
            self.producer.flush()  # Flush any remaining messages
            logger.info("Kafka producer closed successfully")
        except Exception as e:
            logger.error(f"Error closing producer: {e}")


def main():
    """Main function to run the air quality data producer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Air Quality Data Producer')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='air-quality-data',
                       help='Kafka topic name')
    parser.add_argument('--data-file', default=None,
                       help='Path to air quality data CSV file')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Simulation speed multiplier')
    parser.add_argument('--max-messages', type=int, default=None,
                       help='Maximum number of messages to send')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for sending messages')
    
    args = parser.parse_args()
    
    try:
        # Create producer
        producer = AirQualityDataProducer(
            bootstrap_servers=args.bootstrap_servers,
            topic_name=args.topic,
            data_file_path=args.data_file
        )
        
        # Start streaming
        producer.start_streaming(
            simulation_speed=args.speed,
            max_messages=args.max_messages,
            batch_size=args.batch_size
        )
        
    except Exception as e:
        logger.error(f"Producer failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

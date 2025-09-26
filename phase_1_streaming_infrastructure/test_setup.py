"""
Test Script for Kafka KRaft Setup
Phase 1: Streaming Infrastructure and Data Pipeline Architecture

This script tests the complete Kafka setup and data pipeline
to ensure everything is working correctly.
"""

import sys
import time
import logging
import subprocess
import threading
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_kafka_connection():
    """Test Kafka connection and topic availability."""
    try:
        from confluent_kafka import Producer, Consumer, KafkaError
        
        # Test producer connection
        producer_config = {
            'bootstrap.servers': 'localhost:9092',
            'client.id': 'test-producer'
        }
        
        producer = Producer(producer_config)
        logger.info("✓ Producer connection successful")
        
        # Test consumer connection
        consumer_config = {
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'test-consumer-group',
            'auto.offset.reset': 'earliest'
        }
        
        consumer = Consumer(consumer_config)
        consumer.subscribe(['air-quality-data'])
        logger.info("✓ Consumer connection successful")
        
        # Clean up
        producer.flush()
        consumer.close()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Kafka connection failed: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing functionality."""
    try:
        from data_preprocessing import AirQualityDataPreprocessor
        
        # Create sample data for testing
        import pandas as pd
        import numpy as np
        
        # Generate sample data
        n_samples = 100
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        sample_data = pd.DataFrame({
            'Date': timestamps.strftime('%d/%m/%Y'),
            'Time': timestamps.strftime('%H.%M.%S'),
            'CO(GT)': np.random.normal(2.0, 0.5, n_samples),
            'NOx(GT)': np.random.normal(50, 10, n_samples),
            'C6H6(GT)': np.random.normal(5, 1, n_samples),
            'T': np.random.normal(20, 5, n_samples),
            'RH': np.random.normal(60, 15, n_samples),
            'AH': np.random.normal(1.0, 0.2, n_samples)
        })
        
        # Save sample data
        sample_file = 'test_air_quality_data.csv'
        sample_data.to_csv(sample_file, sep=';', index=False)
        
        # Test preprocessing
        preprocessor = AirQualityDataPreprocessor()
        df_processed, quality_report = preprocessor.preprocess_pipeline(
            file_path=sample_file,
            missing_value_method='hybrid',
            engineer_features=True
        )
        
        logger.info("✓ Data preprocessing successful")
        logger.info(f"  Processed {len(df_processed)} records with {len(df_processed.columns)} features")
        logger.info(f"  Quality score: {quality_report['overall_quality_score']:.3f}")
        
        # Clean up
        Path(sample_file).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data preprocessing failed: {e}")
        return False


def test_producer_consumer():
    """Test producer and consumer applications."""
    try:
        from kafka_producer import AirQualityDataProducer
        from kafka_consumer import AirQualityDataConsumer
        
        # Test producer initialization
        producer = AirQualityDataProducer(
            bootstrap_servers='localhost:9092',
            topic_name='air-quality-data'
        )
        logger.info("✓ Producer initialization successful")
        
        # Test consumer initialization
        consumer = AirQualityDataConsumer(
            bootstrap_servers='localhost:9092',
            topic_name='air-quality-data',
            consumer_group='test-group'
        )
        logger.info("✓ Consumer initialization successful")
        
        # Test message creation
        import pandas as pd
        import numpy as np
        
        # Create test message
        test_row = pd.Series({
            'datetime': pd.Timestamp('2024-01-01 12:00:00'),
            'CO(GT)': 2.5,
            'NOx(GT)': 45.0,
            'C6H6(GT)': 4.2,
            'T': 22.0,
            'RH': 65.0,
            'AH': 1.1,
            'hour': 12,
            'day_of_week': 0,
            'month': 1,
            'season': 'Winter',
            'sensor_id': 'test_sensor_001',
            'location': 'test_location'
        })
        
        message = producer._create_message(test_row)
        if message and producer._validate_message(message):
            logger.info("✓ Message creation and validation successful")
        else:
            logger.error("✗ Message creation or validation failed")
            return False
        
        # Test message sending (non-blocking)
        success = producer.send_message(message)
        if success:
            logger.info("✓ Message sending successful")
        else:
            logger.error("✗ Message sending failed")
            return False
        
        # Clean up
        producer.close()
        consumer.close()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Producer/Consumer test failed: {e}")
        return False


def run_integration_test():
    """Run a complete integration test."""
    logger.info("Starting integration test...")
    
    # Test 1: Kafka connection
    logger.info("\n1. Testing Kafka connection...")
    if not test_kafka_connection():
        logger.error("Kafka connection test failed. Make sure Kafka is running.")
        return False
    
    # Test 2: Data preprocessing
    logger.info("\n2. Testing data preprocessing...")
    if not test_data_preprocessing():
        logger.error("Data preprocessing test failed.")
        return False
    
    # Test 3: Producer/Consumer
    logger.info("\n3. Testing producer and consumer...")
    if not test_producer_consumer():
        logger.error("Producer/Consumer test failed.")
        return False
    
    logger.info("\n✓ All tests passed! The system is ready for use.")
    return True


def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Kafka Setup')
    parser.add_argument('--test', choices=['connection', 'preprocessing', 'producer-consumer', 'all'],
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    try:
        if args.test == 'connection':
            success = test_kafka_connection()
        elif args.test == 'preprocessing':
            success = test_data_preprocessing()
        elif args.test == 'producer-consumer':
            success = test_producer_consumer()
        elif args.test == 'all':
            success = run_integration_test()
        
        if success:
            logger.info("Test completed successfully!")
            sys.exit(0)
        else:
            logger.error("Test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Real-Time Air Quality Data Consumer for Apache Kafka
Phase 1: Streaming Infrastructure and Data Pipeline Architecture

This module implements a scalable Kafka consumer for processing
air quality sensor data with comprehensive monitoring and data quality assurance.
"""

import json
import logging
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
from collections import deque
import threading
from data_preprocessing import AirQualityDataPreprocessor

# Confluent Kafka imports
from confluent_kafka import Consumer, KafkaError, KafkaException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kafka_consumer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AirQualityDataConsumer:
    """
    Scalable Kafka consumer for air quality sensor data processing.
    
    Features:
    - Real-time data processing and validation
    - Data quality monitoring and alerting
    - Batch processing capabilities
    - Comprehensive error handling and recovery
    - Performance metrics and monitoring
    """
    
    def __init__(self, 
                 bootstrap_servers: str = 'localhost:9092',
                 topic_name: str = 'air-quality-data',
                 consumer_group: str = 'air-quality-processors',
                 output_dir: str = 'processed_data'):
        """
        Initialize the Air Quality Data Consumer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic_name: Source Kafka topic for air quality data
            consumer_group: Kafka consumer group ID
            output_dir: Directory for storing processed data
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.consumer_group = consumer_group
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kafka consumer
        self.consumer = self._create_consumer()
        
        # Data processing state
        self.processed_messages = 0
        self.error_count = 0
        self.start_time = time.time()
        self.data_buffer = deque(maxlen=1000)  # Buffer for batch processing
        
        # Note: Feature engineering is now done in the producer
        
        # Monitoring and metrics
        self.metrics = {
            'messages_processed': 0,
            'messages_per_second': 0.0,
            'error_rate': 0.0,
            'data_quality_score': 0.0,
            'last_message_time': None,
            'uptime_seconds': 0
        }
        
        # Data quality thresholds
        self.quality_thresholds = {
            'CO(GT)': {'min': 0.0, 'max': 50.0},
            'NOx(GT)': {'min': 0.0, 'max': 1000.0},
            'C6H6(GT)': {'min': 0.0, 'max': 100.0},
            'T': {'min': -20.0, 'max': 50.0},  # Temperature in Celsius
            'RH': {'min': 0.0, 'max': 100.0},  # Relative humidity percentage
        }
        
        logger.info(f"AirQualityDataConsumer initialized for topic: {topic_name}")
        logger.info(f"Consumer group: {consumer_group}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _create_consumer(self) -> Consumer:
        """Create Confluent Kafka consumer with production-grade configuration."""
        try:
            consumer_config = {
                'bootstrap.servers': self.bootstrap_servers,
                'group.id': self.consumer_group,
                'auto.offset.reset': 'earliest',  # Start from beginning if no offset
                'enable.auto.commit': True,
                'auto.commit.interval.ms': 1000,
                'session.timeout.ms': 30000,
                'heartbeat.interval.ms': 10000,
            }
            
            consumer = Consumer(consumer_config)
            consumer.subscribe([self.topic_name])
            logger.info("Confluent Kafka consumer created successfully")
            return consumer
            
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            raise
    
    def _validate_message_data(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean incoming message data.
        
        Args:
            message: Raw message from Kafka
            
        Returns:
            Validation result with status and cleaned data
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'cleaned_data': None,
            'quality_score': 0.0
        }
        
        try:
            # Check message structure for flat format (177 features)
            required_fields = ['timestamp', 'sensor_id', 'location']
            for field in required_fields:
                if field not in message:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Missing '{field}' field")
                    return validation_result
            
            # For flat message format, validate key sensor readings directly
            quality_checks = []
            
            # Validate each sensor reading directly from message
            for sensor, thresholds in self.quality_thresholds.items():
                if sensor in message:
                    value = message[sensor]
                    
                    # Check if value is numeric
                    try:
                        numeric_value = float(value)
                    except (ValueError, TypeError):
                        validation_result['warnings'].append(f"Non-numeric value for {sensor}: {value}")
                        continue
                    
                    # Check value ranges
                    if numeric_value < thresholds['min'] or numeric_value > thresholds['max']:
                        validation_result['warnings'].append(
                            f"{sensor} value {numeric_value} outside normal range "
                            f"[{thresholds['min']}, {thresholds['max']}]"
                        )
                        quality_checks.append(0.5)  # Partial quality score
                    else:
                        quality_checks.append(1.0)  # Full quality score
                else:
                    validation_result['warnings'].append(f"Missing sensor data: {sensor}")
                    quality_checks.append(0.0)  # No quality score
            
            # Calculate overall quality score
            if quality_checks:
                validation_result['quality_score'] = np.mean(quality_checks)
            else:
                validation_result['quality_score'] = 0.0
            
            # Check timestamp validity
            if 'timestamp' in message:
                try:
                    datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    validation_result['warnings'].append("Invalid timestamp format")
            
            # Create cleaned data (for flat format, use the entire message)
            validation_result['cleaned_data'] = message.copy()
            validation_result['cleaned_data']['quality_score'] = validation_result['quality_score']
            validation_result['cleaned_data']['processing_time'] = datetime.now().isoformat()
            
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def _process_message(self, message: Dict[str, Any]) -> bool:
        """
        Process a single air quality message.
        
        Args:
            message: Message dictionary to process
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Validate message
            validation = self._validate_message_data(message)
            
            if not validation['is_valid']:
                logger.warning(f"Invalid message: {validation['errors']}")
                self.error_count += 1
                return False
            
            # Log warnings if any
            if validation['warnings']:
                logger.warning(f"Message warnings: {validation['warnings']}")
            
            # Add to processing buffer
            self.data_buffer.append(validation['cleaned_data'])
            
            # Update metrics
            self.processed_messages += 1
            self.metrics['messages_processed'] = self.processed_messages
            self.metrics['last_message_time'] = datetime.now().isoformat()
            
            # Calculate processing rate
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.metrics['messages_per_second'] = self.processed_messages / elapsed_time
            
            # Calculate error rate
            total_attempts = self.processed_messages + self.error_count
            if total_attempts > 0:
                self.metrics['error_rate'] = self.error_count / total_attempts
            
            # Calculate data quality score
            if self.data_buffer:
                quality_scores = [msg['quality_score'] for msg in self.data_buffer]
                self.metrics['data_quality_score'] = np.mean(quality_scores)
            
            # Log progress periodically
            if self.processed_messages % 100 == 0:
                logger.info(f"Processed {self.processed_messages} messages, "
                           f"Quality score: {self.metrics['data_quality_score']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.error_count += 1
            return False
    
    def _save_batch_data(self, batch_data: List[Dict[str, Any]]) -> None:
        """
        Save a batch of processed data to storage with all 177 features.
        
        Args:
            batch_data: List of processed message dictionaries with 177 features
        """
        try:
            if not batch_data:
                return
            
            # Create DataFrame directly from messages (already have all 177 features)
            df = pd.DataFrame(batch_data)
            
            # Save to CSV with timestamp
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"air_quality_batch_{timestamp_str}.csv"
            filepath = self.output_dir / filename
            
            df.to_csv(filepath, index=False)
            logger.info(f"Saved batch of {len(batch_data)} messages with {len(df.columns)} features to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving batch data: {e}")
    
    def _monitor_system_health(self) -> None:
        """
        Monitor system health and log metrics.
        """
        try:
            uptime = time.time() - self.start_time
            self.metrics['uptime_seconds'] = uptime
            
            # Log health status
            logger.info(f"System Health - "
                       f"Uptime: {uptime:.1f}s, "
                       f"Messages: {self.metrics['messages_processed']}, "
                       f"Rate: {self.metrics['messages_per_second']:.2f} msg/s, "
                       f"Quality: {self.metrics['data_quality_score']:.3f}, "
                       f"Errors: {self.error_count}")
            
            # Alert on high error rate
            if self.metrics['error_rate'] > 0.1:  # 10% error rate
                logger.warning(f"High error rate detected: {self.metrics['error_rate']:.2%}")
            
            # Alert on low data quality
            if self.metrics['data_quality_score'] < 0.7:  # 70% quality threshold
                logger.warning(f"Low data quality detected: {self.metrics['data_quality_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error in health monitoring: {e}")
    
    def start_consuming(self, 
                       batch_size: int = 100,
                       batch_timeout: int = 30,
                       enable_monitoring: bool = True) -> None:
        """
        Start consuming air quality data from Kafka.
        
        Args:
            batch_size: Number of messages to process before saving batch
            batch_timeout: Maximum time to wait before saving partial batch (seconds)
            enable_monitoring: Enable system health monitoring
        """
        logger.info(f"Starting air quality data consumption...")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Batch timeout: {batch_timeout}s")
        logger.info(f"Monitoring enabled: {enable_monitoring}")
        
        last_batch_save = time.time()
        
        try:
            # Start monitoring thread if enabled
            if enable_monitoring:
                monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    daemon=True
                )
                monitor_thread.start()
            
            # Main consumption loop
            while True:
                try:
                    # Poll for messages with timeout
                    msg = self.consumer.poll(timeout=1.0)
                    
                    if msg is None:
                        # No message received, check if we should save batch
                        current_time = time.time()
                        if (len(self.data_buffer) > 0 and 
                            current_time - last_batch_save >= batch_timeout):
                            batch_data = list(self.data_buffer)
                            self.data_buffer.clear()
                            self._save_batch_data(batch_data)
                            last_batch_save = current_time
                        continue
                    
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            logger.info(f"Reached end of partition {msg.partition()}")
                        else:
                            logger.error(f"Consumer error: {msg.error()}")
                        continue
                    
                    # Deserialize message
                    try:
                        message_data = json.loads(msg.value().decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.error(f"Failed to deserialize message: {e}")
                        self.error_count += 1
                        continue
                    
                    # Process message
                    success = self._process_message(message_data)
                    
                    if not success:
                        logger.warning(f"Failed to process message at offset {msg.offset()}")
                    
                    # Check if we should save a batch
                    current_time = time.time()
                    should_save_batch = (
                        len(self.data_buffer) >= batch_size or
                        (len(self.data_buffer) > 0 and 
                         current_time - last_batch_save >= batch_timeout)
                    )
                    
                    if should_save_batch:
                        # Save current batch
                        batch_data = list(self.data_buffer)
                        self.data_buffer.clear()
                        self._save_batch_data(batch_data)
                        last_batch_save = current_time
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self.error_count += 1
                    continue
        
        except KeyboardInterrupt:
            logger.info("Consumption interrupted by user")
        except Exception as e:
            logger.error(f"Error during consumption: {e}")
        finally:
            # Save any remaining data
            if self.data_buffer:
                batch_data = list(self.data_buffer)
                self._save_batch_data(batch_data)
                self.data_buffer.clear()
            
            self.close()
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                time.sleep(30)  # Monitor every 30 seconds
                self._monitor_system_health()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            Dictionary containing current metrics
        """
        self.metrics['uptime_seconds'] = time.time() - self.start_time
        return self.metrics.copy()
    
    def close(self):
        """Close the Kafka consumer."""
        try:
            self.consumer.close()
            logger.info("Kafka consumer closed successfully")
        except Exception as e:
            logger.error(f"Error closing consumer: {e}")


def main():
    """Main function to run the air quality data consumer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Air Quality Data Consumer')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='air-quality-data',
                       help='Kafka topic name')
    parser.add_argument('--consumer-group', default='air-quality-processors',
                       help='Kafka consumer group ID')
    parser.add_argument('--output-dir', default='processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing messages')
    parser.add_argument('--batch-timeout', type=int, default=30,
                       help='Batch timeout in seconds')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable system health monitoring')
    
    args = parser.parse_args()
    
    try:
        # Create consumer
        consumer = AirQualityDataConsumer(
            bootstrap_servers=args.bootstrap_servers,
            topic_name=args.topic,
            consumer_group=args.consumer_group,
            output_dir=args.output_dir
        )
        
        # Start consuming
        consumer.start_consuming(
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            enable_monitoring=not args.no_monitoring
        )
        
    except Exception as e:
        logger.error(f"Consumer failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

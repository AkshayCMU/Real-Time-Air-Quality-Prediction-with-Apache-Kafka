# Phase 1: Streaming Infrastructure and Data Pipeline Architecture

## Overview

This phase implements a robust Apache Kafka streaming infrastructure using KRaft mode (no Zookeeper required) for real-time air quality data processing. The system includes production-grade producer and consumer applications with comprehensive error handling, data quality management, and monitoring capabilities.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Kafka Producer │───▶│  Kafka Broker   │
│ (UCI Air Quality│    │                 │    │   (KRaft Mode)  │
│    Dataset)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Processed Data │◀───│ Kafka Consumer  │◀───│   Kafka Topic   │
│   (CSV Files)   │    │                 │    │ air-quality-data│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

- **KRaft Mode**: Modern Kafka deployment without Zookeeper dependency
- **Production-Grade Producer**: Realistic temporal simulation, comprehensive error handling
- **Scalable Consumer**: Batch processing, data quality validation, monitoring
- **Data Quality Management**: Missing value handling, range validation, quality scoring
- **Comprehensive Logging**: Structured logging with file and console output
- **Monitoring**: Real-time metrics, health checks, performance tracking

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB available space
- **CPU**: 4 cores minimum

### Software Requirements
- **Python**: 3.8 or newer
- **Apache Kafka**: 3.0.0 or newer (with KRaft support)
- **Java**: 11 or newer (required for Kafka)

## Installation and Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Install Apache Kafka

#### Windows
```bash
# Download Kafka from https://kafka.apache.org/downloads
# Extract to C:\kafka (or your preferred location)
# Ensure Java 11+ is installed and JAVA_HOME is set
```

#### macOS (using Homebrew)
```bash
brew install kafka
```

#### Linux (Ubuntu/Debian)
```bash
# Download and extract Kafka
wget https://downloads.apache.org/kafka/2.13-3.6.0/kafka_2.13-3.6.0.tgz
tar -xzf kafka_2.13-3.6.0.tgz
sudo mv kafka_2.13-3.6.0 /opt/kafka
```

### 3. Setup Kafka with KRaft

```bash
# Run the automated setup script
python setup_kafka_kraft.py

# Or manually configure:
# 1. Generate cluster ID
kafka-storage random-uuid

# 2. Format storage
kafka-storage format -t <cluster-id> -c server.properties

# 3. Start Kafka server
kafka-server-start server.properties

# 4. Create topic
kafka-topics --create --topic air-quality-data --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

## Usage

### 1. Start the Producer

```bash
# Basic usage
python kafka_producer.py

# With custom parameters
python kafka_producer.py \
    --bootstrap-servers localhost:9092 \
    --topic air-quality-data \
    --data-file data/air_quality_data.csv \
    --speed 2.0 \
    --max-messages 1000 \
    --batch-size 20
```

**Producer Parameters:**
- `--bootstrap-servers`: Kafka broker addresses (default: localhost:9092)
- `--topic`: Target topic name (default: air-quality-data)
- `--data-file`: Path to air quality CSV file (optional, creates sample data if not found)
- `--speed`: Simulation speed multiplier (default: 1.0 = real-time)
- `--max-messages`: Maximum messages to send (default: all)
- `--batch-size`: Messages per batch (default: 10)

### 2. Start the Consumer

```bash
# Basic usage
python kafka_consumer.py

# With custom parameters
python kafka_consumer.py \
    --bootstrap-servers localhost:9092 \
    --topic air-quality-data \
    --consumer-group air-quality-processors \
    --output-dir processed_data \
    --batch-size 100 \
    --batch-timeout 30
```

**Consumer Parameters:**
- `--bootstrap-servers`: Kafka broker addresses (default: localhost:9092)
- `--topic`: Source topic name (default: air-quality-data)
- `--consumer-group`: Consumer group ID (default: air-quality-processors)
- `--output-dir`: Output directory for processed data (default: processed_data)
- `--batch-size`: Batch size for processing (default: 100)
- `--batch-timeout`: Batch timeout in seconds (default: 30)
- `--no-monitoring`: Disable health monitoring

### 3. Run Both Applications

```bash
# Terminal 1: Start consumer
python kafka_consumer.py

# Terminal 2: Start producer
python kafka_producer.py --speed 5.0
```

## Data Flow

### Producer Data Flow
1. **Data Loading**: Load UCI Air Quality dataset or generate sample data
2. **Preprocessing**: Handle missing values (-200), clean data, add derived features
3. **Message Creation**: Format data into structured JSON messages
4. **Validation**: Validate message data quality and ranges
5. **Publishing**: Send messages to Kafka topic with error handling

### Consumer Data Flow
1. **Message Consumption**: Poll messages from Kafka topic
2. **Deserialization**: Parse JSON messages
3. **Validation**: Validate data quality and ranges
4. **Processing**: Add quality scores and metadata
5. **Batching**: Buffer messages for batch processing
6. **Storage**: Save processed data to CSV files

## Data Quality Management

### Missing Value Handling
- **Sensor Malfunctions**: -200 values replaced with NaN
- **Forward Fill**: Use previous valid values (limit: 3 consecutive)
- **Backward Fill**: Use next valid values (limit: 3 consecutive)
- **Median Imputation**: Fill remaining missing values with median

### Data Validation
- **Range Checks**: Validate sensor readings against expected ranges
- **Type Validation**: Ensure numeric values are properly formatted
- **Timestamp Validation**: Verify datetime format and validity
- **Quality Scoring**: Calculate data quality score (0.0-1.0)

### Quality Thresholds
```python
quality_thresholds = {
    'CO(GT)': {'min': 0.0, 'max': 50.0},      # mg/m³
    'NOx(GT)': {'min': 0.0, 'max': 1000.0},   # ppb
    'C6H6(GT)': {'min': 0.0, 'max': 100.0},   # µg/m³
    'T': {'min': -20.0, 'max': 50.0},         # °C
    'RH': {'min': 0.0, 'max': 100.0},         # %
}
```

## Monitoring and Metrics

### Producer Metrics
- Messages sent per second
- Error rate and retry attempts
- Data quality scores
- Processing latency

### Consumer Metrics
- Messages processed per second
- Data quality scores
- Error rate and validation failures
- Batch processing statistics
- System uptime

### Health Monitoring
- Real-time system health checks
- Alert on high error rates (>10%)
- Alert on low data quality (<70%)
- Performance degradation detection

## Configuration

### Producer Configuration
```python
producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'acks': 'all',                    # Wait for all replicas
    'retries': 3,                     # Retry failed sends
    'compression.type': 'gzip',       # Compress messages
    'enable.idempotence': True,       # Exactly-once delivery
    'batch.size': 16384,              # Batch messages
    'linger.ms': 10,                  # Wait to batch
}
```

### Consumer Configuration
```python
consumer_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'air-quality-processors',
    'auto.offset.reset': 'earliest',  # Start from beginning
    'enable.auto.commit': True,       # Auto-commit offsets
    'isolation.level': 'read_committed',  # Only committed messages
    'max.poll.records': 100,          # Max messages per poll
}
```

## Troubleshooting

### Common Issues

1. **Kafka Connection Failed**
   - Ensure Kafka server is running
   - Check bootstrap servers configuration
   - Verify network connectivity

2. **Topic Not Found**
   - Create topic manually: `kafka-topics --create --topic air-quality-data --bootstrap-server localhost:9092`
   - Check topic exists: `kafka-topics --list --bootstrap-server localhost:9092`

3. **High Memory Usage**
   - Reduce batch sizes
   - Increase consumer poll timeout
   - Monitor system resources

4. **Data Quality Issues**
   - Check input data format
   - Verify sensor value ranges
   - Review preprocessing logic

### Log Files
- **Producer**: `kafka_producer.log`
- **Consumer**: `kafka_consumer.log`
- **Kafka Server**: Check Kafka logs directory

### Performance Tuning
- Adjust batch sizes based on throughput needs
- Tune consumer poll intervals
- Optimize producer compression settings
- Monitor and adjust memory allocation

## File Structure

```
phase_1_streaming_infrastructure/
├── kafka_producer.py          # Main producer application
├── kafka_consumer.py          # Main consumer application
├── setup_kafka_kraft.py       # Kafka KRaft setup script
├── data_preprocessing.py      # Data preprocessing utilities
├── README.md                  # This documentation
└── processed_data/            # Output directory for processed data
    └── air_quality_batch_*.csv
```

## Next Steps

After completing Phase 1:
1. **Phase 2**: Advanced environmental data intelligence and pattern analysis
2. **Phase 3**: Predictive analytics model development and deployment
3. **Phase 4**: Final report and professional documentation

## Support

For technical issues or questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Verify Kafka server status
4. Consult Apache Kafka documentation

## References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Kafka Python Client](https://docs.confluent.io/kafka-clients/python/current/overview.html)
- [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

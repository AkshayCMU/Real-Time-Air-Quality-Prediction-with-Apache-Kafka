# Real-Time Air Quality Prediction Setup Instructions

## Quick Start Guide

This guide will help you set up the complete Real-Time Air Quality Prediction system with Apache Kafka using KRaft mode.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB available space
- **CPU**: 4 cores minimum

### Software Requirements
- **Python**: 3.8 or newer
- **Java**: 11 or newer (required for Kafka)
- **Git**: For cloning the repository

## Step 1: Install Dependencies

### Install Python Dependencies
```bash
# Navigate to the project directory
cd IndividualProject

# Install all required packages
pip install -r requirements.txt
```

### Verify Installation
```bash
# Test Python packages
python -c "import confluent_kafka, pandas, numpy, sklearn; print('All packages installed successfully!')"
```

## Step 2: Install Apache Kafka

### Option A: Download and Install Manually

#### Windows
1. Download Kafka from [Apache Kafka Downloads](https://kafka.apache.org/downloads)
2. Extract to `C:\kafka` (or your preferred location)
3. Ensure Java 11+ is installed and `JAVA_HOME` is set

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

### Option B: Use Docker (Alternative)
```bash
# Run Kafka with Docker Compose
docker-compose up -d
```

## Step 3: Setup Kafka with KRaft

### Automated Setup (Recommended)
```bash
# Run the automated setup script
python phase_1_streaming_infrastructure/setup_kafka_kraft.py

# Or with custom parameters
python phase_1_streaming_infrastructure/setup_kafka_kraft.py \
    --kafka-home /opt/kafka \
    --data-dir ./kafka_data \
    --topic air-quality-data \
    --partitions 3
```

### Manual Setup
```bash
# 1. Generate cluster ID
kafka-storage random-uuid

# 2. Format storage (replace <cluster-id> with generated ID)
kafka-storage format -t <cluster-id> -c server.properties

# 3. Start Kafka server
kafka-server-start server.properties

# 4. Create topic (in a new terminal)
kafka-topics --create \
    --topic air-quality-data \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1
```

## Step 4: Test the Setup

### Run Integration Tests
```bash
# Test all components
python phase_1_streaming_infrastructure/test_setup.py

# Test specific components
python phase_1_streaming_infrastructure/test_setup.py --test connection
python phase_1_streaming_infrastructure/test_setup.py --test preprocessing
python phase_1_streaming_infrastructure/test_setup.py --test producer-consumer
```

### Verify Kafka is Running
```bash
# List topics
kafka-topics --list --bootstrap-server localhost:9092

# Check topic details
kafka-topics --describe --topic air-quality-data --bootstrap-server localhost:9092
```

## Step 5: Run the Data Pipeline

### Terminal 1: Start Consumer
```bash
python phase_1_streaming_infrastructure/kafka_consumer.py
```

### Terminal 2: Start Producer
```bash
# Basic usage (creates sample data)
python phase_1_streaming_infrastructure/kafka_producer.py

# With custom parameters
python phase_1_streaming_infrastructure/kafka_producer.py \
    --speed 2.0 \
    --max-messages 1000 \
    --batch-size 20
```

### Terminal 3: Monitor (Optional)
```bash
# Monitor Kafka topics
kafka-console-consumer --topic air-quality-data --bootstrap-server localhost:9092 --from-beginning
```

## Step 6: Verify Data Processing

### Check Processed Data
```bash
# Check output directory
ls -la phase_1_streaming_infrastructure/processed_data/

# View processed data
head -5 phase_1_streaming_infrastructure/processed_data/air_quality_batch_*.csv
```

### Check Logs
```bash
# Producer logs
tail -f kafka_producer.log

# Consumer logs
tail -f kafka_consumer.log
```

## Troubleshooting

### Common Issues

#### 1. Kafka Connection Failed
```bash
# Check if Kafka is running
ps aux | grep kafka

# Check Kafka logs
tail -f kafka/logs/server.log

# Restart Kafka if needed
kafka-server-start server.properties
```

#### 2. Java Not Found
```bash
# Set JAVA_HOME (Windows)
set JAVA_HOME=C:\Program Files\Java\jdk-11

# Set JAVA_HOME (Linux/macOS)
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
```

#### 3. Port Already in Use
```bash
# Check what's using port 9092
netstat -tulpn | grep 9092

# Kill process if needed
kill -9 <PID>
```

#### 4. Permission Denied
```bash
# Make scripts executable (Linux/macOS)
chmod +x kafka/bin/*.sh

# Run with sudo if needed
sudo kafka-server-start server.properties
```

### Performance Issues

#### High Memory Usage
- Reduce batch sizes in producer/consumer
- Increase system memory
- Tune JVM heap size

#### Slow Processing
- Increase simulation speed
- Optimize batch processing
- Check system resources

## Next Steps

After successful setup:

1. **Phase 2**: Run advanced data analysis
   ```bash
   python phase_2_data_intelligence/exploratory_analysis.py
   ```

2. **Phase 3**: Train predictive models
   ```bash
   python phase_3_predictive_analytics/model_development.py
   ```

3. **Phase 4**: Generate final report
   ```bash
   python phase_4_final_report/generate_report.py
   ```

## Configuration Files

### Kafka Configuration
- `server.properties`: Kafka server configuration
- `kafka_data/`: Kafka data directory

### Application Configuration
- `requirements.txt`: Python dependencies
- `phase_1_streaming_infrastructure/`: Main application code

## Monitoring and Maintenance

### Health Checks
```bash
# Check Kafka status
kafka-topics --list --bootstrap-server localhost:9092

# Monitor system resources
top
htop  # if available
```

### Log Management
```bash
# Rotate logs
logrotate /etc/logrotate.d/kafka

# Clean old logs
find kafka/logs -name "*.log" -mtime +7 -delete
```

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review log files
3. Verify system requirements
4. Consult Apache Kafka documentation

## System Architecture

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

## Success Indicators

You'll know the setup is successful when:

1. ✅ All tests pass (`python test_setup.py`)
2. ✅ Kafka topics are created and accessible
3. ✅ Producer sends messages without errors
4. ✅ Consumer processes messages and saves data
5. ✅ Processed data files are created in `processed_data/`
6. ✅ Log files show successful operations

## Performance Benchmarks

Expected performance on a standard system:

- **Producer**: 100-1000 messages/second
- **Consumer**: 100-1000 messages/second
- **Memory Usage**: 2-4GB for Kafka + applications
- **Storage**: 1-2GB for data and logs

## Security Considerations

For production deployment:

1. Enable SSL/TLS encryption
2. Configure authentication (SASL)
3. Set up authorization (ACLs)
4. Use secure network configuration
5. Implement monitoring and alerting

---

**Note**: This setup is optimized for development and testing. For production deployment, additional security, monitoring, and performance tuning will be required.

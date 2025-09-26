# Phase 1: Kafka Streaming Infrastructure Setup Guide

## Overview
This guide provides comprehensive setup procedures for the Apache Kafka streaming infrastructure used in the Real-Time Air Quality Prediction system.

## Prerequisites

### System Requirements
- **Operating System:** Windows 10/11, macOS, or Linux
- **RAM:** Minimum 8GB (16GB recommended)
- **Storage:** 10GB available space
- **CPU:** 4 cores minimum
- **Python:** Version 3.8 or newer

### Software Dependencies
- **Apache Kafka:** Version 3.0.0 or newer
- **Python Runtime:** 3.8+ with virtual environment
- **Java:** Version 11 or newer (required for Kafka)

## Installation Steps

### 1. Java Installation
```bash
# Windows (using Chocolatey)
choco install openjdk11

# macOS (using Homebrew)
brew install openjdk@11

# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install openjdk-11-jdk
```

### 2. Apache Kafka Installation (KRaft Mode)

#### Download and Setup
```bash
# Download Kafka (replace with latest version)
wget https://downloads.apache.org/kafka/2.13-3.6.0/kafka_2.13-3.6.0.tgz
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0
```

#### KRaft Configuration
```bash
# Generate cluster ID
bin/kafka-storage.sh random-uuid > cluster.id

# Format storage directories
bin/kafka-storage.sh format -t $(cat cluster.id) -c config/kraft/server.properties
```

### 3. Python Environment Setup

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv kafka_env

# Activate environment
# Windows:
kafka_env\Scripts\activate
# macOS/Linux:
source kafka_env/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

## Configuration

### Kafka Server Configuration

#### server.properties (KRaft Mode)
```properties
# KRaft Configuration
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@localhost:9093
listeners=PLAINTEXT://localhost:9092,CONTROLLER://localhost:9093
advertised.listeners=PLAINTEXT://localhost:9092
controller.listener.names=CONTROLLER
inter.broker.listener.name=PLAINTEXT
listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
num.network.threads=3
num.io.threads=8
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600
log.dirs=/tmp/kraft-combined-logs
num.partitions=3
num.recovery.threads.per.data.dir=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
log.retention.hours=168
log.segment.bytes=1073741824
log.retention.check.interval.ms=300000
```

### Topic Configuration
```bash
# Create air-quality-data topic
bin/kafka-topics.sh --create --topic air-quality-data --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

## Architectural Decisions

### 1. KRaft Mode Selection
**Decision:** Use KRaft instead of Zookeeper
**Rationale:** 
- Eliminates Zookeeper dependency
- Simplified deployment and management
- Better performance and scalability
- Future-proof architecture

### 2. Confluent Kafka Client
**Decision:** Use confluent-kafka over kafka-python
**Rationale:**
- Production-grade reliability
- Advanced features (idempotent producers, compression)
- Better error handling and monitoring
- Industry standard for enterprise applications

### 3. Producer Configuration
**Key Settings:**
- `acks='all'`: Ensures data durability
- `retries=3`: Handles transient failures
- `compression.type='gzip'`: Reduces network overhead
- `enable.idempotence=True`: Prevents duplicate messages

### 4. Consumer Configuration
**Key Settings:**
- `auto.offset.reset='earliest'`: Processes all available data
- `enable.auto.commit=True`: Automatic offset management
- `isolation.level='read_committed'`: Ensures data consistency

### 5. Batching Strategy
**Decision:** 100-record batches
**Rationale:**
- Balances throughput and latency
- Optimizes storage efficiency
- Enables batch processing for analytics
- Reduces I/O overhead

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Kafka Server Won't Start
**Symptoms:** Server fails to start or crashes immediately
**Solutions:**
- Check Java version: `java -version`
- Verify port availability: `netstat -an | grep 9092`
- Check disk space: `df -h`
- Review server logs: `logs/server.log`

#### 2. Producer Connection Issues
**Symptoms:** Producer cannot connect to Kafka
**Solutions:**
- Verify Kafka is running: `bin/kafka-topics.sh --list --bootstrap-server localhost:9092`
- Check firewall settings
- Verify bootstrap server configuration
- Test with simple producer: `bin/kafka-console-producer.sh --topic test --bootstrap-server localhost:9092`

#### 3. Consumer Lag Issues
**Symptoms:** Consumer falls behind in processing
**Solutions:**
- Increase consumer group size
- Optimize batch processing
- Check consumer configuration
- Monitor consumer lag: `bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group your-group --describe`

#### 4. Memory Issues
**Symptoms:** OutOfMemoryError or system slowdown
**Solutions:**
- Increase JVM heap size: `export KAFKA_HEAP_OPTS="-Xmx2G -Xms2G"`
- Optimize batch sizes
- Monitor system resources
- Consider horizontal scaling

#### 5. Data Path Issues
**Symptoms:** Analysis modules cannot find processed data
**Solutions:**
- Verify data directory exists: `ls -la phase_1_streaming_infrastructure/processed_data/`
- Check file permissions
- Verify relative paths in configuration
- Test data loading manually

### Performance Optimization

#### 1. Producer Optimization
- Tune batch size and linger time
- Enable compression
- Use async processing
- Monitor producer metrics

#### 2. Consumer Optimization
- Adjust fetch size
- Optimize batch processing
- Use multiple consumer instances
- Monitor consumer lag

#### 3. System Optimization
- Allocate sufficient memory
- Use SSD storage for logs
- Optimize network settings
- Monitor system resources

## Monitoring and Maintenance

### Health Checks
```bash
# Check Kafka status
bin/kafka-topics.sh --list --bootstrap-server localhost:9092

# Monitor consumer groups
bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list

# Check topic details
bin/kafka-topics.sh --describe --topic air-quality-data --bootstrap-server localhost:9092
```

### Log Monitoring
- **Server logs:** `logs/server.log`
- **Producer logs:** `kafka_producer.log`
- **Consumer logs:** `kafka_consumer.log`

### Performance Metrics
- Message throughput
- Consumer lag
- Disk usage
- Memory consumption
- Network I/O

## Security Considerations

### 1. Network Security
- Use TLS/SSL for production
- Implement authentication
- Configure firewall rules
- Use VPN for remote access

### 2. Data Security
- Encrypt sensitive data
- Implement access controls
- Regular security audits
- Monitor for anomalies

## Production Deployment

### 1. Scaling Considerations
- Horizontal scaling with multiple brokers
- Partition strategy for load distribution
- Consumer group scaling
- Load balancing

### 2. High Availability
- Multi-broker setup
- Replication factor configuration
- Backup and recovery procedures
- Disaster recovery planning

### 3. Monitoring Setup
- Kafka metrics collection
- Alert configuration
- Dashboard setup
- Log aggregation

## Support and Resources

### Documentation
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Kafka Documentation](https://docs.confluent.io/)
- [KRaft Mode Guide](https://kafka.apache.org/documentation/#kraft)

### Community Support
- [Apache Kafka Mailing Lists](https://kafka.apache.org/contact)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/apache-kafka)
- [GitHub Issues](https://github.com/apache/kafka/issues)

---

*This setup guide ensures a robust, production-ready Kafka streaming infrastructure for real-time air quality prediction systems.*

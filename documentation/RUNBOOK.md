# Real-Time Air Quality Prediction System - Runbook

## 🚀 System Overview

This runbook provides operational procedures for the Real-Time Air Quality Prediction System built with Apache Kafka, featuring streaming data processing, advanced analytics, and machine learning models.

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Phase 1: Kafka Infrastructure](#phase-1-kafka-infrastructure)
4. [Phase 2: Data Intelligence](#phase-2-data-intelligence)
5. [Phase 3: Predictive Analytics](#phase-3-predictive-analytics)
6. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
7. [Production Deployment](#production-deployment)
8. [Maintenance Procedures](#maintenance-procedures)

---

## 🏗️ System Architecture

### Core Components
- **Apache Kafka**: Real-time data streaming platform (KRaft mode)
- **Python Producers/Consumers**: Data ingestion and processing using confluent-kafka
- **Feature Store**: Real-time feature serving for ML models
- **ML Models**: Linear Regression, ARIMA, ARIMAX
- **Analytics Engine**: Advanced statistical analysis and visualization

### Data Flow
```
AirQualityUCI.csv → Kafka Producer → air-quality-data Topic → Kafka Consumer → processed_data/ → Feature Store → ML Models → Predictions
```

### Directory Structure
```
Real-Time-Air-Quality-Prediction/
├── phase_1_streaming_infrastructure/
│   ├── kafka_producer.py
│   ├── kafka_consumer.py
│   ├── data_preprocessing.py
│   ├── processed_data/ (94 CSV files)
│   └── logs/
├── phase_2_data_intelligence/
│   ├── data_analysis/
│   └── Images/
├── phase_3_predictive_analytics/
│   ├── models/
│   └── proper_feature_store.py
└── phase_4_final_report/
    └── Images/
```

---

## 🔧 Prerequisites & Setup

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB available space
- **CPU**: 4 cores minimum
- **Java**: Version 11+ (for Kafka)

### Software Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Key dependencies:
# confluent-kafka==2.3.0
# pandas==2.1.4
# numpy==1.24.3
# scikit-learn==1.3.2
# statsmodels==0.14.0
# matplotlib==3.8.2
# seaborn==0.13.0
```

### Environment Setup
```bash
# Create virtual environment
python -m venv kafka_env

# Activate environment
# Windows:
kafka_env\Scripts\activate
# macOS/Linux:
source kafka_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Phase 1: Kafka Infrastructure

### 1.1 Kafka Setup (KRaft Mode)

```bash
# Navigate to project directory
cd Real-Time-Air-Quality-Prediction

# Start Kafka (KRaft mode) - requires separate Kafka installation
# Follow setup_instructions.md for detailed Kafka setup
```

### 1.2 Producer Deployment

```bash
# Navigate to Phase 1 directory
cd phase_1_streaming_infrastructure

# Start the producer
python kafka_producer.py

# Expected output:
# ✅ Kafka producer started successfully
# 🔄 Loading data from AirQualityUCI.csv
# 📊 Sending batch 1/100...
# 💾 Saved 100 records to processed_data/
```

**Key Metrics to Monitor:**
- Message throughput: ~100 messages/minute
- Batch size: 100 records per batch
- Processing time: <10 seconds per batch
- Logs: `logs/kafka_producer.log`

### 1.3 Consumer Deployment

```bash
# Start the consumer
python kafka_consumer.py

# Expected output:
# ✅ Kafka consumer started successfully
# 📥 Processing batch 1...
# 💾 Saved 100 records to processed_data/
```

**Key Metrics to Monitor:**
- Consumer lag: <5 seconds
- Processing rate: ~100 records/minute
- Data quality score: >0.8
- Logs: `logs/kafka_consumer.log`

### 1.4 Verification Commands

```bash
# Check processed data files
ls -la processed_data/
# Should show 94+ CSV files

# View producer logs
tail -f logs/kafka_producer.log

# View consumer logs
tail -f logs/kafka_consumer.log

# Test data preprocessing
python data_preprocessing.py
```

---

## 📊 Phase 2: Data Intelligence

### 2.1 Temporal Analysis

```bash
cd phase_2_data_intelligence/data_analysis
python temporal_analysis.py

# Expected outputs:
# 📁 Images/Temporal/hourly_patterns.png
# 📁 Images/Temporal/daily_patterns.png
# 📁 Images/Temporal/monthly_patterns.png
# 📁 Images/Temporal/seasonal_patterns.png
# 📁 Images/Temporal/trend_analysis.png
```

### 2.2 Correlation Analysis

```bash
python correlation_analysis.py

# Expected outputs:
# 📁 Images/Correlation/core_pollutant_correlations.png
# 📁 Images/Correlation/environmental_correlations.png
# 📁 Images/Correlation/pollutant_environmental_correlations.png
# 📁 Images/Correlation/strong_correlations.png
# 📁 Images/Correlation/correlation_distribution.png
```

### 2.3 Anomaly Detection

```bash
python anomaly_detection.py

# Expected outputs:
# 📁 Images/Anomaly/anomaly_detection.png
# 📄 anomaly_report.md
```

### 2.4 Advanced Statistical Analysis

```bash
python advanced_statistical_analysis.py

# Expected outputs:
# 📁 Images/Advanced/acf_pacf_CO.png
# 📁 Images/Advanced/acf_pacf_NOx.png
# 📁 Images/Advanced/acf_pacf_C6H6.png
# 📁 Images/Advanced/acf_pacf_NO2.png
# 📁 Images/Advanced/decomposition_CO.png
# 📁 Images/Advanced/decomposition_NOx.png
# 📁 Images/Advanced/decomposition_C6H6.png
# 📁 Images/Advanced/decomposition_NO2.png
# 📁 Images/Advanced/statistical_tests_summary.png
```

---

## 🤖 Phase 3: Predictive Analytics

### 3.1 Linear Regression Model

```bash
cd phase_3_predictive_analytics/models
python linear_regression_model.py

# Expected outputs:
# 📊 Model Performance: R² scores for all targets
# 📁 Images/Models/linear_regression_performance.png
# 📁 Images/Models/linear_regression_features.png
```

### 3.2 ARIMA Model

```bash
python arima_model.py

# Expected outputs:
# 📊 ARIMA Performance: MAE, RMSE, R²
# 📁 Images/Models/arima_performance.png
# 📁 Images/Models/arima_timeseries.png
```

### 3.3 ARIMAX Model

```bash
python arimax_model.py

# Expected outputs:
# 📊 ARIMAX Performance: R² > 0.95 for all targets
# 📁 Images/Models/arimax_performance.png
# 📁 Images/Models/arimax_features_CO.png
# 📁 Images/Models/arimax_features_NOx.png
# 📁 Images/Models/arimax_features_C6H6.png
# 📁 Images/Models/arimax_features_NO2.png
```

### 3.4 Feature Store Integration

```bash
# Test feature store
cd ..
python proper_feature_store.py

# Expected output:
# 🏪 Feature Store Metadata: 20 features defined
# 🔍 Features for sensor_001@urban_station: Retrieved successfully
```

### 3.5 Run All Models

```bash
# Run all models at once
python run_models.py

# Expected outputs:
# All model performance metrics
# All visualizations generated
# Feature store integration tested
```

---

## 📈 Monitoring & Troubleshooting

### 4.1 Health Checks

#### Kafka Health
```bash
# Check Kafka status
bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092

# Check topic health
bin/kafka-topics.sh --describe --topic air_quality_data --bootstrap-server localhost:9092
```

#### Producer Health
```bash
# Monitor producer logs
tail -f phase_1_streaming_infrastructure/logs/kafka_producer.log

# Check for errors
grep "ERROR" phase_1_streaming_infrastructure/logs/kafka_producer.log
```

#### Consumer Health
```bash
# Monitor consumer logs
tail -f phase_1_streaming_infrastructure/logs/kafka_consumer.log

# Check processing rate
grep "Processed batch" phase_1_streaming_infrastructure/logs/kafka_consumer.log | tail -10
```

### 4.2 Common Issues & Solutions

#### Issue: Producer Connection Failed
```bash
# Error: Failed to connect to Kafka
# Solution: Check Kafka is running
bin/kafka-server-start.sh config/kraft/server.properties
```

#### Issue: Consumer Lag High
```bash
# Error: Consumer lag > 30 seconds
# Solution: Increase consumer instances
python kafka_consumer.py --batch-size 50
```

#### Issue: Memory Issues
```bash
# Error: OutOfMemoryError
# Solution: Increase JVM heap size
export KAFKA_HEAP_OPTS="-Xmx2G -Xms2G"
```

#### Issue: Data Quality Issues
```bash
# Error: Data quality score < 0.5
# Solution: Check data preprocessing
python data_preprocessing.py --validate
```

### 4.3 Performance Monitoring

#### Key Metrics Dashboard
- **Throughput**: Messages per second
- **Latency**: End-to-end processing time
- **Error Rate**: Failed messages percentage
- **Data Quality**: Quality score trends
- **Model Performance**: R², MAE, RMSE trends

#### Alert Thresholds
- **Consumer Lag**: > 60 seconds
- **Error Rate**: > 5%
- **Data Quality**: < 0.7
- **Model Performance**: R² < 0.8

---

## 🚀 Production Deployment

### 5.1 Docker Deployment

```bash
# Create Docker Compose file
version: '3.8'
services:
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
    ports:
      - "9092:9092"
  
  producer:
    build: .
    command: python kafka_producer.py
    depends_on:
      - kafka
  
  consumer:
    build: .
    command: python kafka_consumer.py
    depends_on:
      - kafka
```

### 5.2 Kubernetes Deployment

```yaml
# kafka-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: air-quality-producer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: air-quality-producer
  template:
    metadata:
      labels:
        app: air-quality-producer
    spec:
      containers:
      - name: producer
        image: air-quality-producer:latest
        ports:
        - containerPort: 8080
```

### 5.3 Scaling Procedures

#### Horizontal Scaling
```bash
# Scale producers
kubectl scale deployment air-quality-producer --replicas=5

# Scale consumers
kubectl scale deployment air-quality-consumer --replicas=3
```

#### Vertical Scaling
```bash
# Increase memory limits
kubectl patch deployment air-quality-producer -p '{"spec":{"template":{"spec":{"containers":[{"name":"producer","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

---

## 🔧 Maintenance Procedures

### 6.1 Daily Operations

#### Morning Checklist
- [ ] Check Kafka cluster health
- [ ] Verify producer/consumer status
- [ ] Review error logs
- [ ] Check data quality metrics
- [ ] Validate model performance

#### Evening Checklist
- [ ] Backup processed data
- [ ] Archive old logs
- [ ] Update monitoring dashboards
- [ ] Review performance metrics

### 6.2 Weekly Maintenance

#### Data Quality Review
```bash
# Run data quality analysis
python phase_2_data_intelligence/data_analysis/data_quality_report.py

# Generate weekly report
python generate_weekly_report.py
```

#### Model Performance Review
```bash
# Retrain models if needed
python phase_3_predictive_analytics/models/retrain_models.py

# Update feature store
python phase_3_predictive_analytics/proper_feature_store.py --update
```

### 6.3 Monthly Maintenance

#### System Updates
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update Kafka
# Follow Kafka upgrade procedures
```

#### Performance Optimization
```bash
# Optimize Kafka configuration
# Tune JVM parameters
# Review and optimize queries
```

---

## 📞 Emergency Procedures

### 7.1 System Outage

#### Immediate Response
1. **Check Kafka Status**: `bin/kafka-server-start.sh config/kraft/server.properties`
2. **Restart Services**: Producer → Consumer → Analytics
3. **Verify Data Flow**: Check message processing
4. **Monitor Logs**: Look for error patterns

#### Recovery Steps
```bash
# Restart Kafka
sudo systemctl restart kafka

# Restart producers
python kafka_producer.py --restart

# Restart consumers
python kafka_consumer.py --restart
```

### 7.2 Data Loss Prevention

#### Backup Procedures
```bash
# Backup processed data
tar -czf processed_data_backup_$(date +%Y%m%d).tar.gz phase_1_streaming_infrastructure/processed_data/

# Backup Kafka topics
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic air_quality_data --from-beginning > kafka_backup.json
```

#### Recovery Procedures
```bash
# Restore from backup
tar -xzf processed_data_backup_20241225.tar.gz

# Replay Kafka messages
bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic air_quality_data < kafka_backup.json
```

---

## 📚 Documentation References

### Key Files
- **Setup Instructions**: `documentation/setup_instructions.md`
- **API Documentation**: `documentation/api_reference.md`
- **Troubleshooting Guide**: `documentation/troubleshooting.md`
- **Performance Tuning**: `documentation/performance_guide.md`

### Log Locations
- **Producer Logs**: `phase_1_streaming_infrastructure/logs/kafka_producer.log`
- **Consumer Logs**: `phase_1_streaming_infrastructure/logs/kafka_consumer.log`
- **Analytics Logs**: `phase_2_data_intelligence/logs/`
- **Model Logs**: `phase_3_predictive_analytics/logs/`

### Configuration Files
- **Kafka Config**: `phase_1_streaming_infrastructure/config/kraft/server.properties`
- **Producer Config**: `phase_1_streaming_infrastructure/kafka_producer.py`
- **Consumer Config**: `phase_1_streaming_infrastructure/kafka_consumer.py`
- **Feature Store Config**: `phase_3_predictive_analytics/proper_feature_store.py`

---

## 🎯 Success Metrics

### Operational KPIs
- **Uptime**: > 99.5%
- **Throughput**: > 100 messages/minute
- **Latency**: < 10 seconds end-to-end
- **Error Rate**: < 1%

### Business KPIs
- **Data Quality**: > 0.8
- **Model Accuracy**: R² > 0.9
- **Prediction Latency**: < 5 seconds
- **Feature Store Response**: < 100ms

---

## 📞 Support Contacts

### Technical Support
- **Primary**: System Administrator
- **Secondary**: Data Engineering Team
- **Emergency**: On-call Engineer

### Escalation Procedures
1. **Level 1**: Check logs and restart services
2. **Level 2**: Contact technical team
3. **Level 3**: Escalate to engineering lead
4. **Level 4**: Contact system architect

---

**Last Updated**: December 25, 2024  
**Version**: 1.0  
**Maintained By**: Air Quality Prediction Team

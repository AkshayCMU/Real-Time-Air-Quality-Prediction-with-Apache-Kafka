# Air Quality Anomaly Detection Report
Generated: 2025-09-23 22:39:07
Data Points: 9892

## Statistical Anomaly Detection

### ZSCORE Method
- Anomalies detected: 4294
- Percentage: 44.05%

### IQR Method
- Anomalies detected: 5528
- Percentage: 56.71%

### MODIFIED_ZSCORE Method
- Anomalies detected: 6844
- Percentage: 70.22%

## Machine Learning Anomaly Detection

### Isolation Forest
- Anomalies detected: 488
- Percentage: 5.01%

### Pca
- Anomalies detected: 488
- Percentage: 5.01%

## Temporal Anomaly Detection

### CO(GT)
- Spikes detected: 8
- Sudden changes: 488

### NOx(GT)
- Spikes detected: 10
- Sudden changes: 495

### C6H6(GT)
- Spikes detected: 21
- Sudden changes: 492

### NO2(GT)
- Spikes detected: 0
- Sudden changes: 477

## Recommendations

1. **High Priority**: Investigate extreme anomalies (|z-score| > 5)
2. **Medium Priority**: Review temporal spikes and sudden changes
3. **Data Quality**: Check sensor calibration for consistent anomalies
4. **Monitoring**: Set up automated alerts for anomaly detection

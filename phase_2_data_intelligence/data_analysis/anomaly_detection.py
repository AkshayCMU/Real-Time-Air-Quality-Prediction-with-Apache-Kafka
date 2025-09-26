#!/usr/bin/env python3
"""
Anomaly Detection Module for Air Quality Data
Identifies unusual patterns and outliers in air quality measurements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import zscore
import os
import glob
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AirQualityAnomalyDetector:
    """Detects anomalies in air quality data using multiple methods"""
    
    def __init__(self, data_path="../../phase_1_streaming_infrastructure/processed_data/"):
        self.data_path = data_path
        self.pollutants = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)']
        self.environmental = ['T', 'RH', 'AH']
        self.data = None
        self.anomaly_results = {}
        
    def load_data(self):
        """Load and combine all processed data files"""
        try:
            # Find all CSV files in processed_data directory
            csv_files = glob.glob(os.path.join(self.data_path, "air_quality_batch_*.csv"))
            
            if not csv_files:
                raise FileNotFoundError("No processed data files found")
            
            logger.info(f"Found {len(csv_files)} data files")
            
            # Load and combine all files
            dataframes = []
            for file in csv_files:
                df = pd.read_csv(file)
                dataframes.append(df)
            
            self.data = pd.concat(dataframes, ignore_index=True)
            
            # Convert timestamp to datetime
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                self.data = self.data.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"✅ Loaded data: {len(self.data)} records from {len(csv_files)} files")
            logger.info(f"📊 Data shape: {self.data.shape}")
            logger.info(f"📅 Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def detect_statistical_anomalies(self):
        """Detect anomalies using statistical methods (Z-score, IQR)"""
        try:
            logger.info("🔍 Detecting statistical anomalies...")
            
            # Select numeric columns and clean data
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            numeric_data = self.data[numeric_cols].copy()
            
            # Clean data: remove inf, -inf, and NaN values
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
            numeric_data = numeric_data.dropna()
            
            # Check if we have enough data after cleaning
            if len(numeric_data) < 10:
                logger.warning("⚠️ Insufficient data after cleaning for statistical analysis")
                return {}
            
            anomalies = {}
            
            # 1. Z-score method (|z| > 3) - with error handling
            try:
                z_scores = np.abs(zscore(numeric_data))
                # Replace any inf values in z_scores with 0
                z_scores = np.nan_to_num(z_scores, nan=0, posinf=0, neginf=0)
                z_anomalies = (z_scores > 3).any(axis=1)
                anomalies['zscore'] = {
                    'indices': np.where(z_anomalies)[0],
                    'count': z_anomalies.sum(),
                    'percentage': (z_anomalies.sum() / len(z_anomalies)) * 100
                }
            except Exception as e:
                logger.warning(f"⚠️ Z-score calculation failed: {e}")
                anomalies['zscore'] = {'indices': [], 'count': 0, 'percentage': 0}
            
            # 2. IQR method (using 3.0 * IQR for more realistic thresholds)
            iqr_anomalies = pd.Series(False, index=numeric_data.index)
            for col in numeric_cols:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3.0 * IQR  # Changed from 1.5 to 3.0
                upper_bound = Q3 + 3.0 * IQR  # Changed from 1.5 to 3.0
                col_anomalies = (numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)
                iqr_anomalies = iqr_anomalies | col_anomalies
            
            anomalies['iqr'] = {
                'indices': np.where(iqr_anomalies)[0],
                'count': iqr_anomalies.sum(),
                'percentage': (iqr_anomalies.sum() / len(iqr_anomalies)) * 100
            }
            
            # 3. Modified Z-score (using median) - more conservative threshold
            median_abs_dev = np.median(np.abs(numeric_data - np.median(numeric_data, axis=0)), axis=0)
            modified_z_scores = 0.6745 * (numeric_data - np.median(numeric_data, axis=0)) / median_abs_dev
            modified_z_anomalies = (np.abs(modified_z_scores) > 5.0).any(axis=1)  # Changed from 3.5 to 5.0
            anomalies['modified_zscore'] = {
                'indices': np.where(modified_z_anomalies)[0],
                'count': modified_z_anomalies.sum(),
                'percentage': (modified_z_anomalies.sum() / len(modified_z_anomalies)) * 100
            }
            
            self.anomaly_results['statistical'] = anomalies
            logger.info("✅ Statistical anomaly detection completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in statistical anomaly detection: {e}")
            return False
    
    def detect_ml_anomalies(self):
        """Detect anomalies using machine learning methods"""
        try:
            logger.info("🤖 Detecting ML-based anomalies...")
            
            # Select numeric columns and clean data
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            numeric_data = self.data[numeric_cols].copy()
            
            # Clean data: remove inf, -inf, and NaN values
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
            numeric_data = numeric_data.dropna()
            
            # Check if we have enough data after cleaning
            if len(numeric_data) < 10:
                logger.warning("⚠️ Insufficient data after cleaning for ML analysis")
                return {}
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Check for any remaining inf or nan values
            if np.any(np.isinf(scaled_data)) or np.any(np.isnan(scaled_data)):
                logger.warning("⚠️ Data still contains inf/nan values after scaling")
                scaled_data = np.nan_to_num(scaled_data, nan=0, posinf=0, neginf=0)
            
            ml_anomalies = {}
            
            # 1. Isolation Forest (more conservative contamination rate)
            iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Changed from 0.1 to 0.05
            iso_predictions = iso_forest.fit_predict(scaled_data)
            iso_anomaly_indices = np.where(iso_predictions == -1)[0]
            
            ml_anomalies['isolation_forest'] = {
                'indices': iso_anomaly_indices,
                'count': len(iso_anomaly_indices),
                'percentage': (len(iso_anomaly_indices) / len(scaled_data)) * 100,
                'scores': iso_forest.decision_function(scaled_data)
            }
            
            # 2. PCA-based anomaly detection
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            pca_data = pca.fit_transform(scaled_data)
            
            # Calculate reconstruction error
            reconstructed = pca.inverse_transform(pca_data)
            reconstruction_error = np.mean((scaled_data - reconstructed) ** 2, axis=1)
            
            # Anomalies are points with high reconstruction error
            pca_threshold = np.percentile(reconstruction_error, 95)
            pca_anomaly_indices = np.where(reconstruction_error > pca_threshold)[0]
            
            ml_anomalies['pca'] = {
                'indices': pca_anomaly_indices,
                'count': len(pca_anomaly_indices),
                'percentage': (len(pca_anomaly_indices) / len(scaled_data)) * 100,
                'reconstruction_errors': reconstruction_error
            }
            
            self.anomaly_results['ml'] = ml_anomalies
            logger.info("✅ ML-based anomaly detection completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in ML anomaly detection: {e}")
            return False
    
    def detect_temporal_anomalies(self):
        """Detect temporal anomalies (sudden spikes, unusual patterns)"""
        try:
            logger.info("⏰ Detecting temporal anomalies...")
            
            temporal_anomalies = {}
            
            # Focus on main pollutants
            for pollutant in self.pollutants:
                if pollutant in self.data.columns:
                    # Calculate rolling statistics
                    window = 24  # 24-hour window
                    rolling_mean = self.data[pollutant].rolling(window=window, center=True).mean()
                    rolling_std = self.data[pollutant].rolling(window=window, center=True).std()
                    
                    # Detect spikes (values > mean + 3*std)
                    spikes = self.data[pollutant] > (rolling_mean + 3 * rolling_std)
                    spike_indices = np.where(spikes)[0]
                    
                    # Detect sudden changes (large differences between consecutive values)
                    diff = self.data[pollutant].diff().abs()
                    diff_threshold = diff.quantile(0.95)  # Top 5% of differences
                    sudden_changes = diff > diff_threshold
                    change_indices = np.where(sudden_changes)[0]
                    
                    temporal_anomalies[pollutant] = {
                        'spikes': {
                            'indices': spike_indices,
                            'count': len(spike_indices)
                        },
                        'sudden_changes': {
                            'indices': change_indices,
                            'count': len(change_indices)
                        }
                    }
            
            self.anomaly_results['temporal'] = temporal_anomalies
            logger.info("✅ Temporal anomaly detection completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in temporal anomaly detection: {e}")
            return False
    
    def create_anomaly_visualizations(self):
        """Create comprehensive anomaly detection visualizations"""
        try:
            logger.info("📊 Creating anomaly detection visualizations...")
            
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle('Air Quality Anomaly Detection Analysis', fontsize=20, fontweight='bold')
            
            # 1. Statistical Anomalies Overview
            ax1 = plt.subplot(3, 3, 1)
            if 'statistical' in self.anomaly_results:
                methods = list(self.anomaly_results['statistical'].keys())
                counts = [self.anomaly_results['statistical'][method]['count'] for method in methods]
                percentages = [self.anomaly_results['statistical'][method]['percentage'] for method in methods]
                
                bars = ax1.bar(methods, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
                ax1.set_title('Statistical Anomaly Detection\n(Count by Method)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Number of Anomalies')
                ax1.tick_params(axis='x', rotation=45)
                
                # Add percentage labels on bars
                for bar, pct in zip(bars, percentages):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{pct:.1f}%', ha='center', va='bottom')
            
            # 2. ML Anomalies Overview
            ax2 = plt.subplot(3, 3, 2)
            if 'ml' in self.anomaly_results:
                methods = list(self.anomaly_results['ml'].keys())
                counts = [self.anomaly_results['ml'][method]['count'] for method in methods]
                percentages = [self.anomaly_results['ml'][method]['percentage'] for method in methods]
                
                bars = ax2.bar(methods, counts, color=['orange', 'purple'])
                ax2.set_title('ML Anomaly Detection\n(Count by Method)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Number of Anomalies')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add percentage labels on bars
                for bar, pct in zip(bars, percentages):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{pct:.1f}%', ha='center', va='bottom')
            
            # 3. Z-score Distribution
            ax3 = plt.subplot(3, 3, 3)
            if 'statistical' in self.anomaly_results and 'zscore' in self.anomaly_results['statistical']:
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                numeric_data = self.data[numeric_cols].dropna()
                
                # Clean data: replace inf, -inf, NaN with finite values
                numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(numeric_data) > 0:
                    z_scores = np.abs(zscore(numeric_data))
                    # Handle any remaining NaN values
                    z_scores = np.nan_to_num(z_scores, nan=0.0)
                    max_z_scores = z_scores.max(axis=1)
                    
                    # Filter out infinite values
                    max_z_scores = max_z_scores[np.isfinite(max_z_scores)]
                    
                    if len(max_z_scores) > 0:
                        ax3.hist(max_z_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                        ax3.axvline(3, color='red', linestyle='--', linewidth=2, label='Threshold (|z| = 3)')
                        ax3.set_title('Distribution of Maximum\nZ-scores', fontsize=12, fontweight='bold')
                        ax3.set_xlabel('Maximum |Z-score|')
                        ax3.set_ylabel('Frequency')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)
                    else:
                        ax3.text(0.5, 0.5, 'No valid data\nfor Z-score analysis', 
                                ha='center', va='center', transform=ax3.transAxes)
                        ax3.set_title('Z-score Distribution\n(No Data)', fontsize=12, fontweight='bold')
            
            # 4. Isolation Forest Scores
            ax4 = plt.subplot(3, 3, 4)
            if 'ml' in self.anomaly_results and 'isolation_forest' in self.anomaly_results['ml']:
                scores = self.anomaly_results['ml']['isolation_forest']['scores']
                ax4.hist(scores, bins=50, alpha=0.7, color='orange', edgecolor='black')
                ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Threshold (score = 0)')
                ax4.set_title('Isolation Forest\nAnomaly Scores', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Anomaly Score')
                ax4.set_ylabel('Frequency')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            # 5. PCA Reconstruction Error
            ax5 = plt.subplot(3, 3, 5)
            if 'ml' in self.anomaly_results and 'pca' in self.anomaly_results['ml']:
                errors = self.anomaly_results['ml']['pca']['reconstruction_errors']
                ax5.hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
                threshold = np.percentile(errors, 95)
                ax5.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                           label=f'Threshold (95th percentile)')
                ax5.set_title('PCA Reconstruction\nError Distribution', fontsize=12, fontweight='bold')
                ax5.set_xlabel('Reconstruction Error')
                ax5.set_ylabel('Frequency')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            
            # 6. Time Series with Anomalies (CO)
            ax6 = plt.subplot(3, 3, 6)
            if 'CO(GT)' in self.data.columns:
                co_data = self.data['CO(GT)'].dropna()
                
                # Clean data: remove inf, -inf, NaN values
                co_data = co_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(co_data) > 0:
                    timestamps = self.data.loc[co_data.index, 'timestamp'] if 'timestamp' in self.data.columns else range(len(co_data))
                    
                    # Ensure timestamps are finite
                    if 'timestamp' in self.data.columns:
                        timestamps = timestamps.dropna()
                        co_data = co_data.loc[timestamps.index]
                    
                    ax6.plot(timestamps, co_data, alpha=0.7, color='blue', label='CO(GT)')
                
                # Highlight statistical anomalies
                if 'statistical' in self.anomaly_results and 'zscore' in self.anomaly_results['statistical']:
                    z_anomaly_indices = self.anomaly_results['statistical']['zscore']['indices']
                    valid_anomalies = z_anomaly_indices[z_anomaly_indices < len(co_data)]
                    if len(valid_anomalies) > 0:
                        ax6.scatter(timestamps.iloc[valid_anomalies], co_data.iloc[valid_anomalies], 
                                  color='red', s=50, alpha=0.8, label='Z-score Anomalies')
                
                ax6.set_title('CO(GT) Time Series\nwith Anomalies', fontsize=12, fontweight='bold')
                ax6.set_xlabel('Time')
                ax6.set_ylabel('CO(GT) Concentration')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            
            # 7. Temporal Anomalies by Pollutant
            ax7 = plt.subplot(3, 3, 7)
            if 'temporal' in self.anomaly_results:
                pollutants = list(self.anomaly_results['temporal'].keys())
                spike_counts = [self.anomaly_results['temporal'][poll]['spikes']['count'] for poll in pollutants]
                change_counts = [self.anomaly_results['temporal'][poll]['sudden_changes']['count'] for poll in pollutants]
                
                x = np.arange(len(pollutants))
                width = 0.35
                
                ax7.bar(x - width/2, spike_counts, width, label='Spikes', color='lightcoral', alpha=0.8)
                ax7.bar(x + width/2, change_counts, width, label='Sudden Changes', color='lightblue', alpha=0.8)
                
                ax7.set_title('Temporal Anomalies\nby Pollutant', fontsize=12, fontweight='bold')
                ax7.set_xlabel('Pollutant')
                ax7.set_ylabel('Number of Anomalies')
                ax7.set_xticks(x)
                ax7.set_xticklabels([poll[:8] for poll in pollutants], rotation=45)
                ax7.legend()
                ax7.grid(True, alpha=0.3)
            
            # 8. Anomaly Overlap Analysis
            ax8 = plt.subplot(3, 3, 8)
            self._plot_anomaly_overlap(ax8)
            
            # 9. Anomaly Severity Distribution
            ax9 = plt.subplot(3, 3, 9)
            self._plot_anomaly_severity(ax9)
            
            plt.tight_layout()
            
            # Save the plot
            output_path = "Images/Anomaly/anomaly_detection.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Anomaly detection visualizations saved to {output_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating anomaly visualizations: {e}")
            return False
    
    def _plot_anomaly_overlap(self, ax):
        """Plot overlap between different anomaly detection methods"""
        try:
            # Create a simple overlap analysis
            all_anomaly_indices = set()
            method_names = []
            
            if 'statistical' in self.anomaly_results:
                for method, data in self.anomaly_results['statistical'].items():
                    all_anomaly_indices.update(data['indices'])
                    method_names.append(f"Stat_{method}")
            
            if 'ml' in self.anomaly_results:
                for method, data in self.anomaly_results['ml'].items():
                    all_anomaly_indices.update(data['indices'])
                    method_names.append(f"ML_{method}")
            
            # Count overlaps (simplified)
            overlap_counts = [len(all_anomaly_indices)]
            method_labels = ['Total Unique\nAnomalies']
            
            ax.bar(method_labels, overlap_counts, color='lightgreen', alpha=0.8)
            ax.set_title('Anomaly Detection\nSummary', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Anomalies')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error plotting anomaly overlap: {e}")
    
    def _plot_anomaly_severity(self, ax):
        """Plot anomaly severity distribution"""
        try:
            # Calculate severity based on z-scores with proper data cleaning
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            numeric_data = self.data[numeric_cols].dropna()
            
            # Clean data: replace inf, -inf, NaN with finite values
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(numeric_data) > 0:
                z_scores = np.abs(zscore(numeric_data))
                # Handle any remaining NaN values
                z_scores = np.nan_to_num(z_scores, nan=0.0)
                max_z_scores = z_scores.max(axis=1)
                
                # Filter out infinite values
                max_z_scores = max_z_scores[np.isfinite(max_z_scores)]
                
                if len(max_z_scores) > 0:
                    # Categorize severity
                    severity_categories = ['Low (1-2)', 'Medium (2-3)', 'High (3-5)', 'Extreme (>5)']
                    severity_counts = [
                        np.sum((max_z_scores >= 1) & (max_z_scores < 2)),
                        np.sum((max_z_scores >= 2) & (max_z_scores < 3)),
                        np.sum((max_z_scores >= 3) & (max_z_scores < 5)),
                        np.sum(max_z_scores >= 5)
                    ]
                    
                    colors = ['lightgreen', 'yellow', 'orange', 'red']
                    bars = ax.bar(severity_categories, severity_counts, color=colors, alpha=0.8)
                    ax.set_title('Anomaly Severity\nDistribution', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Number of Anomalies')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
                    
                    # Add count labels on bars
                    for bar, count in zip(bars, severity_counts):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{count}', ha='center', va='bottom')
                else:
                    ax.text(0.5, 0.5, 'No valid data\nfor severity analysis', 
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Anomaly Severity\nDistribution (No Data)', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No valid data\nfor severity analysis', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Anomaly Severity\nDistribution (No Data)', fontsize=12, fontweight='bold')
            
        except Exception as e:
            logger.error(f"Error plotting anomaly severity: {e}")
    
    def generate_anomaly_report(self):
        """Generate detailed anomaly detection report"""
        try:
            logger.info("📝 Generating anomaly detection report...")
            
            report = []
            report.append("# Air Quality Anomaly Detection Report")
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Data Points: {len(self.data)}")
            report.append("")
            
            # Statistical anomalies summary
            if 'statistical' in self.anomaly_results:
                report.append("## Statistical Anomaly Detection")
                report.append("")
                
                for method, data in self.anomaly_results['statistical'].items():
                    report.append(f"### {method.upper()} Method")
                    report.append(f"- Anomalies detected: {data['count']}")
                    report.append(f"- Percentage: {data['percentage']:.2f}%")
                    report.append("")
            
            # ML anomalies summary
            if 'ml' in self.anomaly_results:
                report.append("## Machine Learning Anomaly Detection")
                report.append("")
                
                for method, data in self.anomaly_results['ml'].items():
                    report.append(f"### {method.replace('_', ' ').title()}")
                    report.append(f"- Anomalies detected: {data['count']}")
                    report.append(f"- Percentage: {data['percentage']:.2f}%")
                    report.append("")
            
            # Temporal anomalies summary
            if 'temporal' in self.anomaly_results:
                report.append("## Temporal Anomaly Detection")
                report.append("")
                
                for pollutant, data in self.anomaly_results['temporal'].items():
                    report.append(f"### {pollutant}")
                    report.append(f"- Spikes detected: {data['spikes']['count']}")
                    report.append(f"- Sudden changes: {data['sudden_changes']['count']}")
                    report.append("")
            
            # Recommendations
            report.append("## Recommendations")
            report.append("")
            report.append("1. **High Priority**: Investigate extreme anomalies (|z-score| > 5)")
            report.append("2. **Medium Priority**: Review temporal spikes and sudden changes")
            report.append("3. **Data Quality**: Check sensor calibration for consistent anomalies")
            report.append("4. **Monitoring**: Set up automated alerts for anomaly detection")
            report.append("")
            
            # Save report
            report_text = "\n".join(report)
            output_path = "phase_2_data_intelligence/results/anomaly_report.md"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_text)
            
            logger.info(f"✅ Anomaly detection report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating anomaly report: {e}")
            return False
    
    def run_analysis(self):
        """Run complete anomaly detection analysis"""
        logger.info("🚀 Starting Anomaly Detection for Air Quality Data")
        logger.info("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Detect different types of anomalies
        if not self.detect_statistical_anomalies():
            return False
        
        if not self.detect_ml_anomalies():
            return False
        
        if not self.detect_temporal_anomalies():
            return False
        
        # Create visualizations
        if not self.create_anomaly_visualizations():
            return False
        
        # Generate report
        if not self.generate_anomaly_report():
            return False
        
        logger.info("✅ Anomaly detection analysis completed successfully!")
        return True

def main():
    """Main function to run anomaly detection"""
    detector = AirQualityAnomalyDetector()
    detector.run_analysis()

if __name__ == "__main__":
    main()

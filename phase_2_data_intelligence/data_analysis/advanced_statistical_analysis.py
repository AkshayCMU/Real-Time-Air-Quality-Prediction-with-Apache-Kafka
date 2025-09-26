"""
Advanced Statistical Analysis for Air Quality Data
Implements time series decomposition, ACF/PACF analysis, and advanced statistical methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis for air quality data"""
    
    def __init__(self, data_path="../../phase_1_streaming_infrastructure/processed_data/"):
        self.data_path = data_path
        self.data = None
        self.results = {}
        
    def load_data(self):
        """Load all processed data from CSV files"""
        logger.info(f"🔄 Loading data for advanced statistical analysis from {self.data_path}...")
        
        import glob
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        if not csv_files:
            logger.error(f"❌ No data files found in {self.data_path}")
            return False
            
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file, parse_dates=['timestamp'], index_col='timestamp')
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"⚠️ Error loading {file}: {e}")
                
        if not dataframes:
            logger.error("❌ No data loaded successfully for advanced statistical analysis")
            return False
            
        self.data = pd.concat(dataframes).sort_index()
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"✅ Loaded {len(self.data)} records for advanced statistical analysis")
        return True
    
    def time_series_decomposition(self):
        """Perform time series decomposition into trend, seasonal, and residual components"""
        logger.info("🔍 Performing time series decomposition...")
        
        pollutants = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)']
        decomposition_results = {}
        
        for pollutant in pollutants:
            if pollutant in self.data.columns:
                # Clean data for decomposition
                series = self.data[pollutant].dropna()
                if len(series) < 100:  # Need sufficient data for decomposition
                    logger.warning(f"⚠️ Insufficient data for {pollutant} decomposition")
                    continue
                
                try:
                    # Perform seasonal decomposition
                    decomposition = seasonal_decompose(
                        series, 
                        model='additive', 
                        period=24,  # Daily period (24 hours)
                        extrapolate_trend='freq'
                    )
                    
                    decomposition_results[pollutant] = {
                        'observed': decomposition.observed,
                        'trend': decomposition.trend,
                        'seasonal': decomposition.seasonal,
                        'residual': decomposition.resid
                    }
                    
                    logger.info(f"✅ Decomposition completed for {pollutant}")
                    
                except Exception as e:
                    logger.error(f"❌ Error in decomposition for {pollutant}: {e}")
                    continue
        
        self.results['decomposition'] = decomposition_results
        self._create_decomposition_visualizations(decomposition_results)
        
        logger.info("✅ Time series decomposition completed")
        return decomposition_results
    
    def _create_decomposition_visualizations(self, decomposition_results):
        """Create decomposition visualizations"""
        logger.info("📊 Creating decomposition visualizations...")
        
        # Create output directory
        os.makedirs("../../phase_4_final_report/Images/Advanced", exist_ok=True)
        
        for pollutant, decomp in decomposition_results.items():
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            fig.suptitle(f'Time Series Decomposition - {pollutant}', fontsize=16, fontweight='bold')
            
            # Observed
            axes[0].plot(decomp['observed'].index, decomp['observed'].values, color='blue', linewidth=1)
            axes[0].set_title('Observed')
            axes[0].set_ylabel('Concentration')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomp['trend'].index, decomp['trend'].values, color='red', linewidth=2)
            axes[1].set_title('Trend')
            axes[1].set_ylabel('Concentration')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomp['seasonal'].index, decomp['seasonal'].values, color='green', linewidth=1)
            axes[2].set_title('Seasonal')
            axes[2].set_ylabel('Concentration')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].plot(decomp['residual'].index, decomp['residual'].values, color='orange', linewidth=1)
            axes[3].set_title('Residual')
            axes[3].set_ylabel('Concentration')
            axes[3].set_xlabel('Date')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'../../phase_4_final_report/Images/Advanced/decomposition_{pollutant.replace("(GT)", "")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("✅ Decomposition visualizations saved")
    
    def autocorrelation_analysis(self):
        """Perform ACF and PACF analysis for temporal dependencies"""
        logger.info("🔍 Performing autocorrelation analysis...")
        
        pollutants = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)']
        acf_results = {}
        
        for pollutant in pollutants:
            if pollutant in self.data.columns:
                # Clean data for ACF/PACF
                series = self.data[pollutant].dropna()
                if len(series) < 50:  # Need sufficient data for ACF/PACF
                    logger.warning(f"⚠️ Insufficient data for {pollutant} ACF/PACF analysis")
                    continue
                
                try:
                    # Calculate ACF and PACF (limit lags to 50% of sample size)
                    max_lags = min(40, len(series) // 2)
                    acf_values = acf(series, nlags=max_lags, fft=True)
                    pacf_values = pacf(series, nlags=max_lags, method='ols')
                    
                    acf_results[pollutant] = {
                        'acf': acf_values,
                        'pacf': pacf_values
                    }
                    
                    logger.info(f"✅ ACF/PACF analysis completed for {pollutant}")
                    
                except Exception as e:
                    logger.error(f"❌ Error in ACF/PACF analysis for {pollutant}: {e}")
                    continue
        
        self.results['autocorrelation'] = acf_results
        self._create_acf_pacf_visualizations(acf_results)
        
        logger.info("✅ Autocorrelation analysis completed")
        return acf_results
    
    def _create_acf_pacf_visualizations(self, acf_results):
        """Create ACF and PACF visualizations"""
        logger.info("📊 Creating ACF/PACF visualizations...")
        
        for pollutant, acf_data in acf_results.items():
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'Autocorrelation Analysis - {pollutant}', fontsize=16, fontweight='bold')
            
            # Get the original data for plotting
            series = self.data[pollutant].dropna()
            max_lags = min(20, len(series) // 2)  # Conservative limit
            
            # ACF Plot
            plot_acf(series, ax=axes[0], lags=max_lags, title='Autocorrelation Function (ACF)')
            axes[0].grid(True, alpha=0.3)
            
            # PACF Plot
            plot_pacf(series, ax=axes[1], lags=max_lags, title='Partial Autocorrelation Function (PACF)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'../../phase_4_final_report/Images/Advanced/acf_pacf_{pollutant.replace("(GT)", "")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("✅ ACF/PACF visualizations saved")
    
    def advanced_statistical_tests(self):
        """Perform advanced statistical tests"""
        logger.info("🔍 Performing advanced statistical tests...")
        
        pollutants = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)']
        statistical_results = {}
        
        for pollutant in pollutants:
            if pollutant in self.data.columns:
                series = self.data[pollutant].dropna()
                if len(series) < 30:
                    logger.warning(f"⚠️ Insufficient data for {pollutant} statistical tests")
                    continue
                
                try:
                    # Normality tests
                    shapiro_stat, shapiro_p = stats.shapiro(series.sample(min(5000, len(series))))
                    ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
                    
                    # Stationarity tests (Augmented Dickey-Fuller)
                    from statsmodels.tsa.stattools import adfuller
                    adf_result = adfuller(series)
                    
                    # Ljung-Box test for autocorrelation
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    ljung_box = acorr_ljungbox(series, lags=10, return_df=True)
                    
                    statistical_results[pollutant] = {
                        'normality': {
                            'shapiro_stat': shapiro_stat,
                            'shapiro_p': shapiro_p,
                            'ks_stat': ks_stat,
                            'ks_p': ks_p
                        },
                        'stationarity': {
                            'adf_stat': adf_result[0],
                            'adf_p': adf_result[1],
                            'critical_values': adf_result[4]
                        },
                        'autocorrelation': {
                            'ljung_box_stat': ljung_box['lb_stat'].iloc[-1],
                            'ljung_box_p': ljung_box['lb_pvalue'].iloc[-1]
                        }
                    }
                    
                    logger.info(f"✅ Statistical tests completed for {pollutant}")
                    
                except Exception as e:
                    logger.error(f"❌ Error in statistical tests for {pollutant}: {e}")
                    continue
        
        self.results['statistical_tests'] = statistical_results
        self._create_statistical_summary(statistical_results)
        
        logger.info("✅ Advanced statistical tests completed")
        return statistical_results
    
    def _create_statistical_summary(self, statistical_results):
        """Create statistical summary visualization"""
        logger.info("📊 Creating statistical summary...")
        
        # Create summary table
        summary_data = []
        for pollutant, tests in statistical_results.items():
            summary_data.append({
                'Pollutant': pollutant,
                'Shapiro_P': tests['normality']['shapiro_p'],
                'KS_P': tests['normality']['ks_p'],
                'ADF_P': tests['stationarity']['adf_p'],
                'Ljung_Box_P': tests['autocorrelation']['ljung_box_p']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Advanced Statistical Test Results', fontsize=16, fontweight='bold')
        
        # Normality tests
        axes[0, 0].bar(summary_df['Pollutant'], summary_df['Shapiro_P'])
        axes[0, 0].set_title('Shapiro-Wilk Test (Normality)')
        axes[0, 0].set_ylabel('P-value')
        axes[0, 0].axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
        axes[0, 0].legend()
        
        axes[0, 1].bar(summary_df['Pollutant'], summary_df['KS_P'])
        axes[0, 1].set_title('Kolmogorov-Smirnov Test (Normality)')
        axes[0, 1].set_ylabel('P-value')
        axes[0, 1].axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
        axes[0, 1].legend()
        
        # Stationarity test
        axes[1, 0].bar(summary_df['Pollutant'], summary_df['ADF_P'])
        axes[1, 0].set_title('Augmented Dickey-Fuller Test (Stationarity)')
        axes[1, 0].set_ylabel('P-value')
        axes[1, 0].axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
        axes[1, 0].legend()
        
        # Autocorrelation test
        axes[1, 1].bar(summary_df['Pollutant'], summary_df['Ljung_Box_P'])
        axes[1, 1].set_title('Ljung-Box Test (Autocorrelation)')
        axes[1, 1].set_ylabel('P-value')
        axes[1, 1].axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('../../phase_4_final_report/Images/Advanced/statistical_tests_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Statistical summary saved")
    
    def run_advanced_analysis(self):
        """Run complete advanced statistical analysis"""
        logger.info("🚀 Starting Advanced Statistical Analysis")
        logger.info("=" * 60)
        
        if not self.load_data():
            return False
        
        # Run all analyses
        self.time_series_decomposition()
        self.autocorrelation_analysis()
        self.advanced_statistical_tests()
        
        logger.info("✅ Advanced statistical analysis completed successfully!")
        return True

if __name__ == "__main__":
    analyzer = AdvancedStatisticalAnalyzer()
    analyzer.run_advanced_analysis()

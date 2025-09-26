#!/usr/bin/env python3
"""
Air Quality Correlation Analysis
Creates clean, separate correlation plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AirQualityCorrelationAnalyzer:
    def __init__(self, data_path="../../phase_1_streaming_infrastructure/processed_data/"):
        self.data_path = data_path
        self.data = None
        self.pollutants = ['CO(GT)', 'NO2(GT)', 'C6H6(GT)', 'NOx(GT)']
        self.environmental = ['T', 'RH', 'AH']
        
    def load_data(self):
        """Load all processed CSV files"""
        try:
            logger.info("🔄 Loading processed data...")
            
            # Find all CSV files in the processed_data directory
            csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            
            if not csv_files:
                logger.error(f"❌ No processed data files found in {self.data_path}")
                return False
                
            logger.info(f"📁 Found {len(csv_files)} CSV files")
            
            # Load and concatenate all CSV files
            dataframes = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    dataframes.append(df)
                    logger.info(f"✅ Loaded {len(df)} records from {os.path.basename(file)}")
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {file}: {e}")
                    
            if not dataframes:
                logger.error("❌ No data loaded successfully")
                return False
                
            # Concatenate all dataframes
            self.data = pd.concat(dataframes, ignore_index=True)
            logger.info(f"✅ Total data loaded: {len(self.data)} records")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def create_correlation_visualizations(self):
        """Create clean, separate correlation visualizations"""
        try:
            logger.info("📊 Creating correlation visualizations...")
            
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create separate, clean plots
            self._plot_core_pollutant_correlations()
            self._plot_environmental_correlations()
            self._plot_pollutant_environmental_correlations()
            self._plot_strong_correlations()
            self._plot_correlation_distribution()
            
            logger.info("✅ All correlation visualizations completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating correlation visualizations: {e}")
            return False
    
    def _plot_core_pollutant_correlations(self):
        """Plot correlations between core pollutants only"""
        core_pollutants = ['CO(GT)', 'NO2(GT)', 'C6H6(GT)', 'NOx(GT)']
        
        # Filter data to only core pollutants
        core_data = self.data[core_pollutants].dropna()
        
        if len(core_data) < 10:
            logger.warning("⚠️ Insufficient data for core pollutant analysis")
            return
            
        # Calculate correlations
        corr_matrix = core_data.corr()
        
        # Create plot
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Core Pollutant Correlations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = "Images/Correlation/core_pollutant_correlations.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Core pollutant correlations saved")

    def _plot_environmental_correlations(self):
        """Plot correlations between environmental factors"""
        env_factors = ['T', 'RH', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
        
        # Filter data to only environmental factors
        env_data = self.data[env_factors].dropna()
        
        if len(env_data) < 10:
            logger.warning("⚠️ Insufficient data for environmental factor analysis")
            return
            
        # Calculate correlations
        corr_matrix = env_data.corr()
        
        # Create plot
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Environmental Factor Correlations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = "Images/Correlation/environmental_correlations.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Environmental factor correlations saved")

    def _plot_pollutant_environmental_correlations(self):
        """Plot correlations between pollutants and environmental factors"""
        pollutants = ['CO(GT)', 'NO2(GT)', 'C6H6(GT)', 'NOx(GT)']
        env_factors = ['T', 'RH', 'AH']
        
        # Filter data
        combined_data = self.data[pollutants + env_factors].dropna()
        
        if len(combined_data) < 10:
            logger.warning("⚠️ Insufficient data for pollutant-environmental analysis")
            return
            
        # Calculate correlations
        corr_matrix = combined_data.corr()
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Pollutants vs Environmental Factors', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = "Images/Correlation/pollutant_environmental_correlations.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Pollutant-environmental correlations saved")

    def _plot_strong_correlations(self):
        """Plot only strong correlations (|r| > 0.7)"""
        # Calculate all correlations
        numeric_data = self.data.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_data) < 10:
            logger.warning("⚠️ Insufficient data for strong correlation analysis")
            return
            
        corr_matrix = numeric_data.corr()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append({
                        'pair': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                        'correlation': corr_val
                    })
        
        if not strong_corr:
            logger.warning("⚠️ No strong correlations found (|r| > 0.7)")
            return
            
        # Sort by absolute correlation
        strong_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        pairs = [item['pair'] for item in strong_corr[:10]]  # Top 10
        correlations = [item['correlation'] for item in strong_corr[:10]]
        
        colors = ['red' if corr > 0 else 'blue' for corr in correlations]
        bars = plt.barh(range(len(pairs)), correlations, color=colors, alpha=0.7)
        
        plt.yticks(range(len(pairs)), [pair[:30] + '...' if len(pair) > 30 else pair for pair in pairs])
        plt.xlabel('Correlation Coefficient')
        plt.title('Strong Correlations (|r| > 0.7)', fontsize=16, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', alpha=0.3)
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            plt.text(corr + (0.02 if corr > 0 else -0.02), i, f'{corr:.3f}', 
                    va='center', ha='left' if corr > 0 else 'right')
        
        plt.tight_layout()
        
        output_path = "Images/Correlation/strong_correlations.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Strong correlations saved")

    def _plot_correlation_distribution(self):
        """Plot distribution of correlation coefficients"""
        # Calculate all correlations
        numeric_data = self.data.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_data) < 10:
            logger.warning("⚠️ Insufficient data for correlation distribution analysis")
            return
            
        corr_matrix = numeric_data.corr()
        
        # Get all correlation values (excluding diagonal)
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_values.append(corr_matrix.iloc[i, j])
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.hist(corr_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Correlation')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.title('Distribution of Correlation Coefficients', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = "Images/Correlation/correlation_distribution.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Correlation distribution saved")

def main():
    print("🚀 Starting Correlation Analysis for Air Quality Data")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = AirQualityCorrelationAnalyzer()
    
    # Load data
    if not analyzer.load_data():
        print("❌ Failed to load data")
        return
    
    # Create visualizations
    analyzer.create_correlation_visualizations()
    
    print("✅ Correlation analysis completed!")

if __name__ == "__main__":
    main()
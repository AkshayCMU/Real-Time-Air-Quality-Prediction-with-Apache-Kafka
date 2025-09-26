"""
Temporal Analysis Module for Air Quality Data Intelligence
Comprehensive time series analysis and pattern recognition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import glob
import warnings
warnings.filterwarnings('ignore')

class TemporalAnalyzer:
    """Advanced temporal analysis for air quality data"""
    
    def __init__(self, data_path="../../phase_1_streaming_infrastructure/processed_data/"):
        self.data_path = data_path
        self.data = None
        self.results = {}
        
    def load_data(self):
        """Load processed air quality data"""
        try:
            # Load all processed data files
            import glob
            files = glob.glob(f"{self.data_path}*.csv")
            if not files:
                raise FileNotFoundError("No processed data files found")
            
            # Load and combine all files
            dataframes = []
            for file in files:
                df = pd.read_csv(file)
                dataframes.append(df)
            
            self.data = pd.concat(dataframes, ignore_index=True)
            
            # Convert timestamp to datetime
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.set_index('timestamp', inplace=True)
            
            print(f"✅ Loaded data: {len(self.data)} records from {len(files)} files")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in air quality data"""
        if self.data is None:
            print("❌ No data loaded. Call load_data() first.")
            return
        
        print("🔍 Analyzing temporal patterns...")
        
        # Key pollutants for analysis
        pollutants = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)']
        
        # 1. Hourly patterns
        hourly_patterns = {}
        for pollutant in pollutants:
            if pollutant in self.data.columns:
                hourly_patterns[pollutant] = self.data.groupby(self.data.index.hour)[pollutant].mean()
        
        # 2. Daily patterns
        daily_patterns = {}
        for pollutant in pollutants:
            if pollutant in self.data.columns:
                daily_patterns[pollutant] = self.data.groupby(self.data.index.dayofweek)[pollutant].mean()
        
        # 3. Monthly patterns
        monthly_patterns = {}
        for pollutant in pollutants:
            if pollutant in self.data.columns:
                monthly_patterns[pollutant] = self.data.groupby(self.data.index.month)[pollutant].mean()
        
        # 4. Seasonal patterns
        seasonal_patterns = {}
        for pollutant in pollutants:
            if pollutant in self.data.columns:
                seasonal_patterns[pollutant] = self.data.groupby(self.data.index.quarter)[pollutant].mean()
        
        self.results['temporal_patterns'] = {
            'hourly': hourly_patterns,
            'daily': daily_patterns,
            'monthly': monthly_patterns,
            'seasonal': seasonal_patterns
        }
        
        print("✅ Temporal pattern analysis completed")
        return self.results['temporal_patterns']
    
    def create_temporal_visualizations(self):
        """Create comprehensive temporal visualizations"""
        if 'temporal_patterns' not in self.results:
            print("❌ No temporal patterns found. Run analyze_temporal_patterns() first.")
            return
        
        print("📊 Creating temporal visualizations...")
        
        # Create reports directory if it doesn't exist
        os.makedirs('Images/Temporal', exist_ok=True)
        
        patterns = self.results['temporal_patterns']
        pollutants = list(patterns['hourly'].keys())
        
        # 1. Hourly patterns - separate plot
        plt.figure(figsize=(12, 8))
        hours = list(range(24))
        for pollutant in pollutants:
            if len(patterns['hourly'][pollutant]) > 0:
                plt.plot(hours[:len(patterns['hourly'][pollutant])], patterns['hourly'][pollutant], 
                        label=pollutant, marker='o', linewidth=3, markersize=6)
        plt.title('Hourly Air Quality Patterns', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Hour of Day', fontsize=14)
        plt.ylabel('Average Concentration', fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2), fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('Images/Temporal/hourly_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Daily patterns - separate plot
        plt.figure(figsize=(10, 6))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for pollutant in pollutants:
            if len(patterns['daily'][pollutant]) > 0:
                available_days = days[:len(patterns['daily'][pollutant])]
                plt.plot(available_days, patterns['daily'][pollutant], 
                        label=pollutant, marker='s', linewidth=3, markersize=8)
        plt.title('Daily Air Quality Patterns', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Average Concentration', fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('Images/Temporal/daily_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Monthly patterns - separate plot
        plt.figure(figsize=(12, 6))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for pollutant in pollutants:
            if len(patterns['monthly'][pollutant]) > 0:
                available_months = months[:len(patterns['monthly'][pollutant])]
                plt.plot(available_months, patterns['monthly'][pollutant], 
                        label=pollutant, marker='^', linewidth=3, markersize=8)
        plt.title('Monthly Air Quality Patterns', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Average Concentration', fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('Images/Temporal/monthly_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Seasonal patterns - separate plot
        plt.figure(figsize=(8, 6))
        seasons = ['Q1', 'Q2', 'Q3', 'Q4']
        for pollutant in pollutants:
            if len(patterns['seasonal'][pollutant]) > 0:
                available_seasons = seasons[:len(patterns['seasonal'][pollutant])]
                plt.plot(available_seasons, patterns['seasonal'][pollutant], 
                        label=pollutant, marker='d', linewidth=3, markersize=8)
        plt.title('Seasonal Air Quality Patterns', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Quarter', fontsize=14)
        plt.ylabel('Average Concentration', fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('Images/Temporal/seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ All temporal visualizations saved to phase_4_final_report/Images/Temporal/")
    
    def analyze_trends(self):
        """Analyze trends in air quality data"""
        if self.data is None:
            print("❌ No data loaded. Call load_data() first.")
            return
        
        print("📈 Analyzing trends...")
        
        pollutants = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)']
        trends = {}
        
        for pollutant in pollutants:
            if pollutant in self.data.columns:
                # Calculate rolling averages
                data_clean = self.data[pollutant].dropna()
                if len(data_clean) > 0:
                    # 7-day rolling average
                    rolling_7d = data_clean.rolling(window=7*24, min_periods=1).mean()
                    # 30-day rolling average
                    rolling_30d = data_clean.rolling(window=30*24, min_periods=1).mean()
                    
                    trends[pollutant] = {
                        'raw': data_clean,
                        'rolling_7d': rolling_7d,
                        'rolling_30d': rolling_30d
                    }
        
        self.results['trends'] = trends
        
        # Create trend visualization
        self._create_trend_visualization(trends)
        
        print("✅ Trend analysis completed")
        return trends
    
    def _create_trend_visualization(self, trends):
        """Create trend visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        pollutants = list(trends.keys())
        
        for i, pollutant in enumerate(pollutants):
            if i < 4:  # Only plot first 4 pollutants
                ax = axes[i]
                
                # Plot raw data (sampled for performance)
                raw_data = trends[pollutant]['raw']
                if len(raw_data) > 1000:
                    sample_data = raw_data.iloc[::len(raw_data)//1000]
                else:
                    sample_data = raw_data
                
                ax.plot(sample_data.index, sample_data.values, alpha=0.3, color='lightblue', label='Raw Data')
                ax.plot(trends[pollutant]['rolling_7d'].index, trends[pollutant]['rolling_7d'].values, 
                       color='orange', linewidth=2, label='7-Day Average')
                ax.plot(trends[pollutant]['rolling_30d'].index, trends[pollutant]['rolling_30d'].values, 
                       color='red', linewidth=2, label='30-Day Average')
                
                ax.set_title(f'{pollutant} Trends', fontsize=12, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Concentration')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('Images/Temporal', exist_ok=True)
        plt.savefig('Images/Temporal/trend_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Trend visualizations saved to reports/trend_analysis.png")
    
    def generate_temporal_insights(self):
        """Generate insights from temporal analysis"""
        if 'temporal_patterns' not in self.results:
            print("❌ No temporal patterns found. Run analyze_temporal_patterns() first.")
            return
        
        print("💡 Generating temporal insights...")
        
        insights = []
        patterns = self.results['temporal_patterns']
        
        # Analyze hourly patterns
        for pollutant, hourly_data in patterns['hourly'].items():
            peak_hour = hourly_data.idxmax()
            min_hour = hourly_data.idxmin()
            peak_value = hourly_data.max()
            min_value = hourly_data.min()
            
            insights.append(f"🕐 {pollutant}: Peak at {peak_hour}:00 ({peak_value:.2f}), Lowest at {min_hour}:00 ({min_value:.2f})")
        
        # Analyze daily patterns
        for pollutant, daily_data in patterns['daily'].items():
            weekday_avg = daily_data.iloc[:5].mean()  # Monday to Friday
            weekend_avg = daily_data.iloc[5:].mean()  # Saturday and Sunday
            
            if weekday_avg > weekend_avg:
                insights.append(f"📅 {pollutant}: Higher on weekdays ({weekday_avg:.2f}) vs weekends ({weekend_avg:.2f})")
            else:
                insights.append(f"📅 {pollutant}: Higher on weekends ({weekend_avg:.2f}) vs weekdays ({weekday_avg:.2f})")
        
        # Analyze seasonal patterns
        for pollutant, seasonal_data in patterns['seasonal'].items():
            peak_season = seasonal_data.idxmax()
            min_season = seasonal_data.idxmin()
            
            insights.append(f"🌍 {pollutant}: Peak in Q{peak_season}, Lowest in Q{min_season}")
        
        self.results['insights'] = insights
        
        # Print insights
        print("\n" + "="*60)
        print("🎯 TEMPORAL ANALYSIS INSIGHTS")
        print("="*60)
        for insight in insights:
            print(insight)
        print("="*60)
        
        return insights

def main():
    """Main function to run temporal analysis"""
    print("🚀 Starting Temporal Analysis for Air Quality Data")
    print("="*60)
    
    # Initialize analyzer
    analyzer = TemporalAnalyzer()
    
    # Load data
    if not analyzer.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Run analysis
    analyzer.analyze_temporal_patterns()
    analyzer.create_temporal_visualizations()
    analyzer.analyze_trends()
    analyzer.generate_temporal_insights()
    
    print("\n✅ Temporal analysis completed successfully!")
    print("📊 Check the reports/ folder for visualizations")

if __name__ == "__main__":
    main()
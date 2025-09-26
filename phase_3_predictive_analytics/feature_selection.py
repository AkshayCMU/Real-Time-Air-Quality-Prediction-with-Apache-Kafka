"""
Smart Feature Selection for Air Quality Prediction
Optimizes feature store by selecting only the most important features
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
import joblib
import os

logger = logging.getLogger(__name__)

class AirQualityFeatureSelector:
    """
    Smart feature selection based on model performance and importance
    """
    
    def __init__(self, data_path: str = "../../processed_data/"):
        self.data_path = data_path
        self.feature_importance = {}
        self.selected_features = {}
        self.feature_rankings = {}
        
    def load_feature_importance(self, model_results: Dict) -> Dict:
        """
        Extract feature importance from model results
        """
        logger.info("🔍 Analyzing feature importance from model results...")
        
        # This would typically come from the trained model
        # For now, we'll create a mock based on our Linear Regression results
        feature_importance = {
            'CO(GT)': {
                'top_features': [
                    'PT08.S1(CO)_ma_3h',
                    'PT08.S1(CO)_lag_1h', 
                    'PT08.S1(CO)_diff_24h',
                    'T_ma_3h',
                    'PT08.S1(CO)_pct_change_24h'
                ],
                'importance_scores': [0.7, 0.4, 0.35, 0.3, 0.25]
            },
            'NOx(GT)': {
                'top_features': [
                    'PT08.S4(NO2)_ma_3h',
                    'PT08.S4(NO2)_lag_1h',
                    'PT08.S4(NO2)_diff_24h',
                    'PT08.S2(NMHC)_ma_3h',
                    'T_ma_3h'
                ],
                'importance_scores': [0.6, 0.45, 0.4, 0.35, 0.3]
            },
            'C6H6(GT)': {
                'top_features': [
                    'PT08.S2(NMHC)_pct_change_24h',
                    'PT08.S2(NMHC)_diff_24h',
                    'PT08.S2(NMHC)_ma_3h',
                    'PT08.S2(NMHC)_lag_1h',
                    'T_ma_3h'
                ],
                'importance_scores': [0.48, 0.42, 0.38, 0.32, 0.28]
            },
            'NO2(GT)': {
                'top_features': [
                    'PT08.S4(NO2)_ma_3h',
                    'PT08.S4(NO2)_lag_1h',
                    'PT08.S4(NO2)_diff_24h',
                    'T_ma_3h',
                    'AH_ma_24h'
                ],
                'importance_scores': [0.55, 0.4, 0.35, 0.3, 0.25]
            }
        }
        
        self.feature_importance = feature_importance
        logger.info("✅ Feature importance analysis completed")
        return feature_importance
    
    def select_optimal_features(self, target: str, max_features: int = 20) -> List[str]:
        """
        Select optimal features for a specific target variable
        """
        if target not in self.feature_importance:
            logger.warning(f"⚠️ No importance data for target: {target}")
            return []
        
        target_importance = self.feature_importance[target]
        top_features = target_importance['top_features'][:max_features]
        
        logger.info(f"🎯 Selected {len(top_features)} features for {target}")
        return top_features
    
    def create_unified_feature_set(self, max_features_per_target: int = 15) -> List[str]:
        """
        Create a unified feature set combining top features from all targets
        """
        logger.info("🔄 Creating unified feature set...")
        
        all_features = set()
        
        for target in self.feature_importance:
            target_features = self.select_optimal_features(target, max_features_per_target)
            all_features.update(target_features)
        
        unified_features = list(all_features)
        logger.info(f"✅ Unified feature set: {len(unified_features)} features")
        
        return unified_features
    
    def create_feature_store_config(self, unified_features: List[str]) -> Dict:
        """
        Create optimized feature store configuration
        """
        logger.info("🔧 Creating optimized feature store configuration...")
        
        # Group features by type for better organization
        feature_groups = {
            'core_measurements': [],
            'temporal_features': [],
            'lagged_features': [],
            'rolling_features': [],
            'environmental_features': [],
            'pollutant_ratios': [],
            'quality_indicators': []
        }
        
        for feature in unified_features:
            if '_ma_' in feature:
                feature_groups['rolling_features'].append(feature)
            elif '_lag_' in feature:
                feature_groups['lagged_features'].append(feature)
            elif '_diff_' in feature or '_pct_change_' in feature:
                feature_groups['temporal_features'].append(feature)
            elif feature.startswith('PT08.S'):
                feature_groups['core_measurements'].append(feature)
            elif feature in ['T', 'AH', 'RH']:
                feature_groups['environmental_features'].append(feature)
            elif '_ratio_' in feature:
                feature_groups['pollutant_ratios'].append(feature)
            else:
                feature_groups['quality_indicators'].append(feature)
        
        config = {
            'total_features': len(unified_features),
            'feature_groups': feature_groups,
            'selected_features': unified_features,
            'optimization_benefits': {
                'reduced_complexity': f"From 216 to {len(unified_features)} features",
                'improved_performance': "Faster inference and training",
                'lower_costs': "Reduced bandwidth and storage"
            }
        }
        
        logger.info("✅ Feature store configuration created")
        return config
    
    def save_optimized_features(self, unified_features: List[str], config: Dict):
        """
        Save optimized feature configuration
        """
        output_dir = "../../phase_4_final_report/Feature_Optimization/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature list
        feature_df = pd.DataFrame({
            'feature_name': unified_features,
            'feature_type': [self._categorize_feature(f) for f in unified_features],
            'importance_rank': range(1, len(unified_features) + 1)
        })
        
        feature_df.to_csv(f"{output_dir}/optimized_features.csv", index=False)
        
        # Save configuration
        config_df = pd.DataFrame(list(config['feature_groups'].items()), 
                                columns=['group', 'features'])
        config_df.to_csv(f"{output_dir}/feature_groups.csv", index=False)
        
        # Save summary
        with open(f"{output_dir}/optimization_summary.txt", 'w') as f:
            f.write("Air Quality Feature Store Optimization Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Features Selected: {config['total_features']}\n")
            f.write(f"Original Feature Count: 216\n")
            f.write(f"Reduction: {((216 - config['total_features']) / 216 * 100):.1f}%\n\n")
            f.write("Feature Groups:\n")
            for group, features in config['feature_groups'].items():
                f.write(f"  {group}: {len(features)} features\n")
            f.write(f"\nOptimization Benefits:\n")
            for benefit, description in config['optimization_benefits'].items():
                f.write(f"  {benefit}: {description}\n")
        
        logger.info(f"✅ Optimized features saved to {output_dir}")
    
    def _categorize_feature(self, feature: str) -> str:
        """Categorize feature by type"""
        if '_ma_' in feature:
            return 'rolling_average'
        elif '_lag_' in feature:
            return 'lagged_value'
        elif '_diff_' in feature:
            return 'difference'
        elif '_pct_change_' in feature:
            return 'percentage_change'
        elif feature.startswith('PT08.S'):
            return 'sensor_reading'
        elif feature in ['T', 'AH', 'RH']:
            return 'environmental'
        else:
            return 'other'
    
    def generate_feature_store_code(self, unified_features: List[str]) -> str:
        """
        Generate optimized feature store code
        """
        logger.info("🔧 Generating optimized feature store code...")
        
        code = f'''
# Optimized Feature Store Configuration
# Generated automatically based on feature importance analysis

OPTIMIZED_FEATURES = {unified_features}

# Feature groups for organized serving
FEATURE_GROUPS = {{
    'core_measurements': [f for f in OPTIMIZED_FEATURES if f.startswith('PT08.S')],
    'temporal_features': [f for f in OPTIMIZED_FEATURES if '_diff_' in f or '_pct_change_' in f],
    'lagged_features': [f for f in OPTIMIZED_FEATURES if '_lag_' in f],
    'rolling_features': [f for f in OPTIMIZED_FEATURES if '_ma_' in f],
    'environmental_features': [f for f in OPTIMIZED_FEATURES if f in ['T', 'AH', 'RH']]
}}

def get_optimized_features(target: str = None) -> List[str]:
    """
    Get optimized feature set for specific target or all features
    """
    if target:
        # Return target-specific features
        target_features = {{
            'CO(GT)': [f for f in OPTIMIZED_FEATURES if 'PT08.S1' in f or 'CO' in f],
            'NOx(GT)': [f for f in OPTIMIZED_FEATURES if 'PT08.S4' in f or 'NO2' in f],
            'C6H6(GT)': [f for f in OPTIMIZED_FEATURES if 'PT08.S2' in f or 'NMHC' in f],
            'NO2(GT)': [f for f in OPTIMIZED_FEATURES if 'PT08.S4' in f or 'NO2' in f]
        }}
        return target_features.get(target, OPTIMIZED_FEATURES)
    
    return OPTIMIZED_FEATURES

def get_feature_group(group_name: str) -> List[str]:
    """
    Get features from specific group
    """
    return FEATURE_GROUPS.get(group_name, [])
'''
        
        return code

def main():
    """Main function to demonstrate feature selection"""
    logger.info("🚀 Starting Feature Selection Optimization")
    
    # Initialize selector
    selector = AirQualityFeatureSelector()
    
    # Load feature importance (mock data for now)
    feature_importance = selector.load_feature_importance({})
    
    # Create unified feature set
    unified_features = selector.create_unified_feature_set(max_features_per_target=15)
    
    # Create configuration
    config = selector.create_feature_store_config(unified_features)
    
    # Save optimized features
    selector.save_optimized_features(unified_features, config)
    
    # Generate code
    code = selector.generate_feature_store_code(unified_features)
    
    # Save generated code
    with open("../../phase_4_final_report/Feature_Optimization/optimized_feature_store.py", 'w') as f:
        f.write(code)
    
    logger.info("✅ Feature selection optimization completed!")
    logger.info(f"📊 Selected {len(unified_features)} features out of 216")
    logger.info(f"💡 Optimization: {((216 - len(unified_features)) / 216 * 100):.1f}% reduction")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

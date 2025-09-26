#!/usr/bin/env python3
"""
Run Phase 3 ML Models
Execute Linear Regression and ARIMA models for air quality prediction
"""

import sys
import os
import logging

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from linear_regression_model import AirQualityLinearRegression
from arima_model import AirQualityARIMA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_linear_regression():
    """Run Linear Regression model"""
    logger.info("🚀 Starting Linear Regression Model")
    logger.info("=" * 50)
    
    try:
        model = AirQualityLinearRegression()
        success = model.run_analysis()
        
        if success:
            logger.info("✅ Linear Regression completed successfully!")
            return True
        else:
            logger.error("❌ Linear Regression failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running Linear Regression: {e}")
        return False

def run_arima():
    """Run ARIMA model"""
    logger.info("🚀 Starting ARIMA Model")
    logger.info("=" * 50)
    
    try:
        model = AirQualityARIMA()
        success = model.run_analysis()
        
        if success:
            logger.info("✅ ARIMA completed successfully!")
            return True
        else:
            logger.error("❌ ARIMA failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running ARIMA: {e}")
        return False

def main():
    """Run all Phase 3 models"""
    logger.info("🎯 Phase 3: Predictive Analytics Model Development")
    logger.info("=" * 60)
    
    results = {}
    
    # Run Linear Regression
    results['linear_regression'] = run_linear_regression()
    
    # Run ARIMA
    results['arima'] = run_arima()
    
    # Summary
    logger.info("📊 Phase 3 Results Summary:")
    logger.info("=" * 40)
    
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{model.upper()}: {status}")
    
    # Overall status
    all_success = all(results.values())
    if all_success:
        logger.info("🎉 All Phase 3 models completed successfully!")
    else:
        logger.warning("⚠️ Some Phase 3 models failed")
    
    return all_success

if __name__ == "__main__":
    main()

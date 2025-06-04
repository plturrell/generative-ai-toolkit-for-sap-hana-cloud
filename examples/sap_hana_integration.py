#!/usr/bin/env python
"""
SAP HANA Integration Example for the Generative AI Toolkit for SAP HANA Cloud
This example demonstrates how to integrate the toolkit with SAP HANA Cloud
to perform advanced analytics, time series forecasting, and use specialized tools.
"""

import os
import logging
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

# Import toolkit components
from hana_ai.vectorstore import HANAMLinVectorEngine
from hana_ai.smart_dataframe import HanaSmartDataFrame
from hana_ai.tools.toolkit import HANAMLToolkit
from hana_ai.tools.hana_ml_tools.ts_visualizer_tools import TimeSeriesDatasetReport, ForecastLinePlot
from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import (
    AutomaticTimeSeriesFitAndSave,
    AutomaticTimeseriesLoadModelAndPredict
)
from hana_ai.tools.hana_ml_tools.ts_check_tools import (
    TimeSeriesCheck,
    StationarityTest,
    SeasonalityTest,
    TrendTest
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SAP HANA Connection parameters
HOST = os.environ.get("HANA_HOST", "localhost")
PORT = os.environ.get("HANA_PORT", "39015")
USER = os.environ.get("HANA_USER", "SYSTEM")
PASSWORD = os.environ.get("HANA_PASSWORD", "")
SCHEMA = os.environ.get("HANA_SCHEMA", "SYSTEM")

# LLM configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def create_sample_time_series_data(connection_string: str) -> str:
    """Create a sample time series table in SAP HANA"""
    from hdbcli import dbapi
    
    # Create connection
    conn = dbapi.connect(
        address=HOST,
        port=int(PORT),
        user=USER,
        password=PASSWORD,
        encrypt=True,
        sslValidateCertificate=False
    )
    
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    table_name = f"{SCHEMA}.SAMPLE_SALES_DATA"
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        DATE_ID DATE,
        PRODUCT_ID VARCHAR(20),
        SALES_AMOUNT DECIMAL(10,2),
        QUANTITY INTEGER
    )
    """)
    
    # Check if data exists
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    
    # Insert sample data if table is empty
    if count == 0:
        logger.info("Inserting sample time series data...")
        
        # Generate sample data - 3 years of daily sales for 3 products
        from datetime import datetime, timedelta
        import random
        import math
        
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2023, 12, 31)
        products = ["Product_A", "Product_B", "Product_C"]
        
        # Prepare batch insert
        insert_query = f"INSERT INTO {table_name} VALUES(?, ?, ?, ?)"
        batch_data = []
        
        current_date = start_date
        while current_date <= end_date:
            for product in products:
                # Create seasonal pattern with trend
                day_of_year = current_date.timetuple().tm_yday
                season_factor = 1.0 + 0.3 * math.sin(day_of_year / 365 * 2 * math.pi)
                
                # Add trend
                days_since_start = (current_date - start_date).days
                trend_factor = 1.0 + (days_since_start / 1095) * 0.5  # 50% growth over 3 years
                
                # Base values
                if product == "Product_A":
                    base_amount = 100
                    base_quantity = 10
                elif product == "Product_B":
                    base_amount = 250
                    base_quantity = 5
                else:
                    base_amount = 50
                    base_quantity = 20
                
                # Calculate with randomness
                quantity = max(1, int(base_quantity * season_factor * trend_factor * (0.8 + random.random() * 0.4)))
                amount = round(base_amount * season_factor * trend_factor * (0.9 + random.random() * 0.2), 2)
                
                batch_data.append((current_date, product, amount, quantity))
            
            current_date += timedelta(days=1)
        
        # Execute batch insert
        cursor.executemany(insert_query, batch_data)
        conn.commit()
        logger.info(f"Inserted {len(batch_data)} records into {table_name}")
    
    cursor.close()
    conn.close()
    
    return table_name


def run_time_series_analysis(connection_string: str, table_name: str, api_key: str):
    """Run time series analysis using the toolkit's time series tools"""
    # Initialize toolkit
    toolkit = HANAMLToolkit(connection_string=connection_string, api_key=api_key)
    
    # 1. Create dataset report
    logger.info("Generating time series dataset report...")
    dataset_report_tool = TimeSeriesDatasetReport(
        connection_context=connection_string,
        table_name=table_name,
        timestamp_col="DATE_ID",
        target_cols=["SALES_AMOUNT", "QUANTITY"],
        by_col=["PRODUCT_ID"],
        output_path="./dataset_report.png"
    )
    dataset_report = toolkit.run_tool(dataset_report_tool)
    logger.info("Dataset report generated")
    
    # 2. Run time series checks
    logger.info("Running time series checks...")
    
    # Initialize the check tools
    ts_check = TimeSeriesCheck(
        connection_context=connection_string,
        table_name=table_name,
        timestamp_col="DATE_ID",
        target_col="SALES_AMOUNT",
        by_col=["PRODUCT_ID"]
    )
    
    stationarity_test = StationarityTest(
        connection_context=connection_string,
        table_name=table_name,
        timestamp_col="DATE_ID",
        target_col="SALES_AMOUNT",
        by_col=["PRODUCT_ID"]
    )
    
    seasonality_test = SeasonalityTest(
        connection_context=connection_string,
        table_name=table_name,
        timestamp_col="DATE_ID",
        target_col="SALES_AMOUNT",
        by_col=["PRODUCT_ID"]
    )
    
    trend_test = TrendTest(
        connection_context=connection_string,
        table_name=table_name,
        timestamp_col="DATE_ID",
        target_col="SALES_AMOUNT",
        by_col=["PRODUCT_ID"]
    )
    
    # Run the tools
    ts_check_result = toolkit.run_tool(ts_check)
    stationarity_result = toolkit.run_tool(stationarity_test)
    seasonality_result = toolkit.run_tool(seasonality_test)
    trend_result = toolkit.run_tool(trend_test)
    
    logger.info(f"Time Series Check Results: {ts_check_result}")
    logger.info(f"Stationarity Test Results: {stationarity_result}")
    logger.info(f"Seasonality Test Results: {seasonality_result}")
    logger.info(f"Trend Test Results: {trend_result}")
    
    # 3. Fit automatic time series model
    logger.info("Fitting automatic time series model...")
    fit_tool = AutomaticTimeSeriesFitAndSave(
        connection_context=connection_string,
        table_name=table_name,
        timestamp_col="DATE_ID",
        target_col="SALES_AMOUNT",
        by_col=["PRODUCT_ID"],
        forecast_horizon=30,  # 30 days forecast
        model_name="SALES_FORECAST_MODEL"
    )
    fit_result = toolkit.run_tool(fit_tool)
    logger.info(f"Model fitting result: {fit_result}")
    
    # 4. Make predictions with the fitted model
    logger.info("Making forecasts with the fitted model...")
    predict_tool = AutomaticTimeseriesLoadModelAndPredict(
        connection_context=connection_string,
        model_name="SALES_FORECAST_MODEL",
        forecast_horizon=30
    )
    prediction_result = toolkit.run_tool(predict_tool)
    logger.info(f"Prediction result: {prediction_result}")
    
    # 5. Visualize the forecast
    logger.info("Generating forecast visualization...")
    forecast_plot_tool = ForecastLinePlot(
        connection_context=connection_string,
        table_name=table_name,
        timestamp_col="DATE_ID",
        target_col="SALES_AMOUNT",
        by_col=["PRODUCT_ID"],
        model_name="SALES_FORECAST_MODEL",
        output_path="./forecast_line_plot.png"
    )
    forecast_plot = toolkit.run_tool(forecast_plot_tool)
    logger.info("Forecast visualization generated")
    
    return {
        "dataset_report": dataset_report,
        "ts_checks": {
            "basic_check": ts_check_result,
            "stationarity": stationarity_result,
            "seasonality": seasonality_result,
            "trend": trend_result
        },
        "model_fit": fit_result,
        "predictions": prediction_result,
        "visualization": forecast_plot
    }


def create_smart_dataframe_analysis(connection_string: str, table_name: str, api_key: str):
    """Use the smart dataframe to analyze the time series data"""
    # Create a query to get the data
    query = f"""
    SELECT 
        DATE_ID,
        PRODUCT_ID,
        SALES_AMOUNT,
        QUANTITY,
        SALES_AMOUNT / QUANTITY AS UNIT_PRICE
    FROM {table_name}
    ORDER BY DATE_ID, PRODUCT_ID
    """
    
    # Create a smart dataframe
    logger.info("Creating smart dataframe for analysis...")
    smart_df = HanaSmartDataFrame(
        connection_string=connection_string,
        query=query,
        api_key=api_key
    )
    
    # Ask questions about the data
    questions = [
        "What's the overall trend in sales for each product?",
        "Which product has the highest average unit price?",
        "What is the total sales amount for each product?",
        "Is there seasonality in the sales data?",
        "Which month typically has the highest sales across all products?"
    ]
    
    results = {}
    for question in questions:
        logger.info(f"Asking: {question}")
        answer = smart_df.ask(question)
        logger.info(f"Answer: {answer}")
        results[question] = answer
    
    # Try a transformation
    logger.info("Performing smart transformation on the dataframe...")
    transformed_df = smart_df.transform(
        "Create a monthly sales summary with total sales amount and average unit price by product"
    )
    
    return {
        "question_answers": results,
        "transformation": "Monthly sales summary created successfully" if transformed_df is not None else "Transformation failed"
    }


def main():
    """Main function demonstrating SAP HANA integration"""
    logger.info("Starting SAP HANA integration example")
    
    # Check for required credentials
    if not PASSWORD or not OPENAI_API_KEY:
        logger.error("Missing required credentials. Please set HANA_PASSWORD and OPENAI_API_KEY environment variables.")
        return
    
    try:
        # Connect to SAP HANA
        connection_string = f"hana://{USER}:{PASSWORD}@{HOST}:{PORT}/?schema={SCHEMA}"
        logger.info(f"Connecting to SAP HANA at {HOST}:{PORT} with user {USER}")
        
        # Create sample time series data
        table_name = create_sample_time_series_data(connection_string)
        logger.info(f"Sample time series data table: {table_name}")
        
        # Run time series analysis
        logger.info("Running time series analysis...")
        ts_results = run_time_series_analysis(connection_string, table_name, OPENAI_API_KEY)
        
        # Run smart dataframe analysis
        logger.info("Running smart dataframe analysis...")
        df_results = create_smart_dataframe_analysis(connection_string, table_name, OPENAI_API_KEY)
        
        logger.info("SAP HANA integration example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in SAP HANA integration example: {str(e)}")
        raise

if __name__ == "__main__":
    main()
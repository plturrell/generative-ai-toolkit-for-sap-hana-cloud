"""
This module is used to generate interactive visualizations for time series data.

The following classes are available:

    * :class `TimeSeriesDatasetReport`
    * :class `ForecastLinePlot`
    * :class `InteractiveTimeSeriesPlot`
    * :class `CustomizableDashboard`
"""

import json
import logging
import os
import uuid
from pathlib import Path
import tempfile
from typing import Optional, Type, List, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.visualizers.visualizer_base import forecast_line_plot
from hana_ml.visualizers.unified_report import UnifiedReport

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TSDatasetInput(BaseModel):
    """
    The input schema for the TSDatasetTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    output_dir: Optional[str] = Field(description="the output directory to save the report, it is optional", default=None)
    interactive: Optional[bool] = Field(description="whether to generate an interactive report", default=True)

class ForecastLinePlotInput(BaseModel):
    """
    The input schema for the ForecastLinePlot tool.
    """
    predict_result: str = Field(description="the name of the predicted result table. If not provided, ask the user. Do not guess.")
    actual_table_name: Optional[str] = Field(description="the name of the actual data table, it is optional", default=None)
    confidence: Optional[tuple] = Field(description="the column names of confidence bounds, it is optional", default=None)
    output_dir: Optional[str] = Field(description="the output directory to save the line plot, it is optional", default=None)
    theme: Optional[str] = Field(description="the theme for the plot (light, dark, or custom)", default="light")
    include_controls: Optional[bool] = Field(description="whether to include interactive controls", default=True)

class InteractivePlotInput(BaseModel):
    """
    The input schema for the InteractiveTimeSeriesPlot tool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    time_column: str = Field(description="the column containing time/date information. If not provided, ask the user. Do not guess.")
    value_columns: List[str] = Field(description="list of columns to visualize. If not provided, ask the user. Do not guess.")
    plot_type: str = Field(description="type of plot (line, scatter, bar, area, candlestick)", default="line")
    title: Optional[str] = Field(description="title for the plot", default=None)
    output_dir: Optional[str] = Field(description="the output directory to save the plot", default=None)
    theme: Optional[str] = Field(description="the theme for the plot (light, dark, or custom)", default="light")
    height: Optional[int] = Field(description="height of the plot in pixels", default=600)
    width: Optional[int] = Field(description="width of the plot in pixels", default=900)
    annotations: Optional[List[Dict[str, Any]]] = Field(description="annotations to add to the plot", default=None)

class CustomizableDashboardInput(BaseModel):
    """
    The input schema for the CustomizableDashboard tool.
    """
    table_names: List[str] = Field(description="list of table names to include in the dashboard")
    layout: Optional[List[Dict[str, Any]]] = Field(description="dashboard layout configuration", default=None)
    title: Optional[str] = Field(description="dashboard title", default="Time Series Dashboard")
    theme: Optional[str] = Field(description="dashboard theme (light, dark, custom)", default="light")
    output_dir: Optional[str] = Field(description="the output directory to save the dashboard", default=None)
    include_filters: Optional[bool] = Field(description="whether to include interactive filters", default=True)
    height: Optional[int] = Field(description="height of the dashboard in pixels", default=900)
    width: Optional[int] = Field(description="width of the dashboard in pixels", default=1200)

class TimeSeriesDatasetReport(BaseTool):
    """
    This tool generates an interactive report for a time series dataset.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The path of the generated report.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - table_name
                  - the name of the table. If not provided, ask the user. Do not guess.
                * - key
                  - the key of the dataset. If not provided, ask the user. Do not guess.
                * - endog
                  - the endog of the dataset. If not provided, ask the user. Do not guess
                * - interactive
                  - whether to generate an interactive report with Plotly
    """
    name: str = "ts_dataset_report"
    """Name of the tool."""
    description: str = "Generate an interactive time series report for a HANA table with advanced visualization features."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = TSDatasetInput
    return_direct: bool = False
    bas: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def set_bas(self, bas: bool) -> None:
        """
        Set the bas flag to True or False.
        """
        self.bas = bas

    def _run(
        self, table_name: str, key: str, endog: str, output_dir: Optional[str]=None, 
        interactive: Optional[bool]=True, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # check hana has the table
        if not self.connection_context.has_table(table_name):
            return json.dumps({"error": f"Table {table_name} does not exist."})
        # check key in the table
        if key not in self.connection_context.table(table_name).columns:
            return json.dumps({"error": f"Key {key} does not exist in table {table_name}."})
        # check endog in the table
        if endog not in self.connection_context.table(table_name).columns:
            return json.dumps({"error": f"Endog {endog} does not exist in table {table_name}."})
            
        df = self.connection_context.table(table_name).select(key, endog)
        
        if output_dir is None:
            destination_dir = os.path.join(tempfile.gettempdir(), "hanaml_report")
        else:
            destination_dir = output_dir
            
        if not os.path.exists(destination_dir):
            try:
                os.makedirs(destination_dir, exist_ok=True)
            except Exception as e:
                logger.error("Error creating directory %s: %s", destination_dir, e)
                raise

        output_base = os.path.join(destination_dir, f"{table_name}_ts_report")
        
        # Generate standard report with UnifiedReport
        ur = UnifiedReport(df).build(key=key, endog=endog)
        ur.display(save_html=output_base)
        
        if not self.bas:
            ur.display()  # directly display in jupyter
            
        # If interactive is True, generate enhanced interactive Plotly report
        if interactive:
            df_pd = df.collect()
            
            # Create an interactive Plotly report with multiple visualizations
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=("Time Series Plot", "Distribution", 
                               "Seasonal Decomposition", "Autocorrelation",
                               "Rolling Statistics", "Histogram"),
                specs=[[{"colspan": 2}, None],
                       [{}, {}],
                       [{}, {}]],
                vertical_spacing=0.1
            )
            
            # Main time series plot (interactive)
            fig.add_trace(
                go.Scatter(
                    x=df_pd[key], 
                    y=df_pd[endog],
                    mode='lines+markers',
                    name=endog,
                    hovertemplate=f"{key}: %{{x}}<br>{endog}: %{{y:.2f}}<extra></extra>",
                    line=dict(width=1.5),
                ),
                row=1, col=1
            )
            
            # Add range slider and buttons for time navigation
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                row=1, col=1
            )
            
            # Distribution plot
            try:
                # Calculate kernel density estimate for the data
                kde = pd.Series(df_pd[endog]).plot.kde()
                x_kde = kde.get_lines()[0].get_xdata()
                y_kde = kde.get_lines()[0].get_ydata()
                
                fig.add_trace(
                    go.Scatter(
                        x=x_kde, 
                        y=y_kde,
                        mode='lines',
                        name='Density',
                        fill='tozeroy',
                        line=dict(color='rgba(31, 119, 180, 0.7)'),
                    ),
                    row=2, col=1
                )
            except Exception as e:
                logger.warning(f"Could not generate distribution plot: {e}")
                fig.add_annotation(
                    text="Distribution plot not available",
                    x=0.5, y=0.5,
                    showarrow=False,
                    row=2, col=1
                )
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df_pd[endog],
                    nbinsx=30,
                    name='Histogram',
                    marker_color='rgba(31, 119, 180, 0.7)',
                ),
                row=3, col=2
            )
            
            # Rolling statistics
            window = min(30, len(df_pd) // 10) if len(df_pd) > 30 else 5
            rolling_mean = df_pd[endog].rolling(window=window).mean()
            rolling_std = df_pd[endog].rolling(window=window).std()
            
            fig.add_trace(
                go.Scatter(
                    x=df_pd[key],
                    y=rolling_mean,
                    mode='lines',
                    name=f'Rolling Mean ({window})',
                    line=dict(color='rgba(255, 127, 14, 0.8)'),
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_pd[key],
                    y=rolling_std,
                    mode='lines',
                    name=f'Rolling Std ({window})',
                    line=dict(color='rgba(44, 160, 44, 0.8)'),
                ),
                row=3, col=1
            )
            
            # Add buttons for user interaction
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        buttons=[
                            dict(
                                args=[{"visible": [True, True, True, True, True]}],
                                label="Show All",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": [True, False, False, False, False]}],
                                label="Time Series Only",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": [False, True, False, False, False]}],
                                label="Distribution Only",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": [False, False, True, False, False]}],
                                label="ACF Only",
                                method="update"
                            ),
                        ],
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    ),
                ]
            )
            
            # Improve overall layout
            fig.update_layout(
                title=f"Interactive Time Series Analysis for {table_name}",
                height=900,
                width=1200,
                hovermode="closest",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                template="plotly_white",
            )
            
            # Add annotations and insights
            # Calculate basic statistics
            mean_val = df_pd[endog].mean()
            median_val = df_pd[endog].median()
            std_val = df_pd[endog].std()
            min_val = df_pd[endog].min()
            max_val = df_pd[endog].max()
            
            stats_text = (f"Statistics:<br>"
                         f"Mean: {mean_val:.2f}<br>"
                         f"Median: {median_val:.2f}<br>"
                         f"Std Dev: {std_val:.2f}<br>"
                         f"Min: {min_val:.2f}<br>"
                         f"Max: {max_val:.2f}")
            
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4
            )
            
            # Save the interactive report
            interactive_output = f"{output_base}_interactive.html"
            fig.write_html(
                interactive_output,
                include_plotlyjs="cdn",
                full_html=True,
                include_mathjax="cdn",
            )
            
            return json.dumps({
                "standard_html_file": str(Path(output_base + ".html").as_posix()),
                "interactive_html_file": str(Path(interactive_output).as_posix())
            }, ensure_ascii=False)
        
        return json.dumps({"html_file": str(Path(output_base + ".html").as_posix())}, ensure_ascii=False)

    async def _arun(
        self, table_name: str, key: str, endog: str, output_dir: Optional[str]=None, 
        interactive: Optional[bool]=True, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(table_name, key, endog, output_dir, interactive, run_manager=run_manager)

class ForecastLinePlot(BaseTool):
    """
    This tool generates an interactive line plot for the forecasted result.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The path of the generated line plot.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - predict_result
                  - the name of the predicted result table. If not provided, ask the user. Do not guess.
                * - actual_table_name
                  - the name of the actual data table, it is optional
                * - confidence
                  - the column names of confidence bounds, it is optional
                * - theme
                  - the theme for the plot (light, dark, or custom)
                * - include_controls
                  - whether to include interactive controls
    """
    name: str = "forecast_line_plot"
    """Name of the tool."""
    description: str = "Generate interactive forecast plots with confidence intervals, customizable themes, and rich annotations."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ForecastLinePlotInput
    """Input schema of the tool."""
    return_direct: bool = False
    bas: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def set_bas(self, bas: bool) -> None:
        """
        Set the bas flag to True or False.
        """
        self.bas = bas

    def _run(
        self, predict_result: str, actual_table_name: Optional[str]=None, 
        confidence: Optional[tuple]=None, output_dir: Optional[str]=None,
        theme: Optional[str]="light", include_controls: Optional[bool]=True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # check predict_result in the hana db
        if not self.connection_context.has_table(predict_result):
            return json.dumps({"error": f"Table {predict_result} does not exist."})
        # check actual_table_name in the hana db
        if actual_table_name is not None and not self.connection_context.has_table(actual_table_name):
            return json.dumps({"error": f"Table {actual_table_name} does not exist."})
            
        predict_df = self.connection_context.table(predict_result)
        
        # Determine confidence interval columns
        if confidence is None:
            if "YHAT_LOWER" in predict_df.columns and "YHAT_UPPER" in predict_df.columns:
                # check if "YHAT_LOWER" column has values
                if not predict_df["YHAT_LOWER"].collect()["YHAT_LOWER"].isnull().all():
                    confidence = ("YHAT_LOWER", "YHAT_UPPER")
            elif "LO80" in predict_df.columns and "HI80" in predict_df.columns:
                if not predict_df["LO80"].collect()["LO80"].isnull().all():
                    confidence = ("LO80", "HI80")
            elif "LO95" in predict_df.columns and "HI95" in predict_df.columns:
                if not predict_df["LO95"].collect()["LO95"].isnull().all():
                    if confidence is None:
                        confidence = ("LO95", "HI95")
                    else:
                        confidence = confidence + ("LO95", "HI95")
            elif "PI1_LOWER" in predict_df.columns and "PI1_UPPER" in predict_df.columns:
                if not predict_df["PI1_LOWER"].collect()["PI1_LOWER"].isnull().all():
                    confidence = ("PI1_LOWER", "PI1_UPPER")
            elif "PI2_LOWER" in predict_df.columns and "PI2_UPPER" in predict_df.columns:
                if not predict_df["PI2_LOWER"].collect()["PI2_LOWER"].isnull().all():
                    if confidence is None:
                        confidence = ("PI2_LOWER", "PI2_UPPER")
                    else:
                        confidence = confidence + ("PI2_LOWER", "PI2_UPPER")

        # Create output directory
        if output_dir is None:
            destination_dir = os.path.join(tempfile.gettempdir(), "hanaml_chart")
        else:
            destination_dir = output_dir
            
        if not os.path.exists(destination_dir):
            try:
                os.makedirs(destination_dir, exist_ok=True)
            except Exception as e:
                logger.error("Error creating directory %s: %s", destination_dir, e)
                raise

        # First generate the standard forecast plot with original function
        if actual_table_name is None:
            fig_standard = forecast_line_plot(predict_df, confidence=confidence)
        else:
            fig_standard = forecast_line_plot(predict_df, self.connection_context.table(actual_table_name), confidence)
            
        output_file_standard = os.path.join(
            destination_dir,
            f"{predict_result}_forecast_line_plot.html",
        )
        
        with Path(output_file_standard).open("w", encoding="utf-8") as f:
            f.write(fig_standard.to_html(full_html=True))
            
        if not self.bas:
            fig_standard.show()  # directly display in jupyter

        # Now create an enhanced interactive Plotly version
        predict_data = predict_df.collect()
        
        # Determine key columns for forecasting (typically DATE or TIMESTAMP and YHAT or FORECAST)
        date_cols = [col for col in predict_data.columns if any(term in col.upper() for term in ["DATE", "TIME", "TS", "TIMESTAMP"])]
        if not date_cols:
            date_cols = [predict_data.columns[0]]  # Default to first column if no date column found
            
        date_col = date_cols[0]
        
        # Find forecast column
        forecast_cols = [col for col in predict_data.columns if any(term in col.upper() for term in ["YHAT", "FORECAST", "PREDICTION", "PREDICTED"])]
        if not forecast_cols:
            # Take the second column or any numeric column that's not a date
            non_date_cols = [col for col in predict_data.columns if col != date_col]
            if non_date_cols:
                forecast_cols = [non_date_cols[0]]
            else:
                forecast_cols = [predict_data.columns[1] if len(predict_data.columns) > 1 else predict_data.columns[0]]
                
        forecast_col = forecast_cols[0]
        
        # Create interactive Plotly figure
        fig = go.Figure()
        
        # Set theme
        template = "plotly_white"
        if theme == "dark":
            template = "plotly_dark"
        elif theme == "custom":
            template = {
                "layout": {
                    "colorway": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
                    "paper_bgcolor": "rgba(240, 240, 240, 0.95)",
                    "plot_bgcolor": "rgba(240, 240, 240, 0.95)",
                    "font": {"color": "#333333"},
                    "title": {"font": {"size": 24, "color": "#333333"}},
                }
            }
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=predict_data[date_col],
            y=predict_data[forecast_col],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#1f77b4", width=2),
            hovertemplate=f"{date_col}: %{{x}}<br>{forecast_col}: %{{y:.2f}}<extra></extra>"
        ))
        
        # Add confidence intervals if available
        if confidence:
            lower_bound, upper_bound = confidence[0], confidence[1]
            
            if lower_bound in predict_data.columns and upper_bound in predict_data.columns:
                # Add confidence interval as a filled area
                fig.add_trace(go.Scatter(
                    x=predict_data[date_col].tolist() + predict_data[date_col].tolist()[::-1],
                    y=predict_data[upper_bound].tolist() + predict_data[lower_bound].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(31, 119, 180, 0.2)",
                    line=dict(color="rgba(31, 119, 180, 0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="95% Confidence Interval"
                ))
                
                # Add upper and lower bounds as separate lines
                fig.add_trace(go.Scatter(
                    x=predict_data[date_col],
                    y=predict_data[upper_bound],
                    mode="lines",
                    line=dict(color="#1f77b4", width=1, dash="dash"),
                    name="Upper Bound",
                    hovertemplate=f"{date_col}: %{{x}}<br>Upper Bound: %{{y:.2f}}<extra></extra>"
                ))
                
                fig.add_trace(go.Scatter(
                    x=predict_data[date_col],
                    y=predict_data[lower_bound],
                    mode="lines",
                    line=dict(color="#1f77b4", width=1, dash="dash"),
                    name="Lower Bound",
                    hovertemplate=f"{date_col}: %{{x}}<br>Lower Bound: %{{y:.2f}}<extra></extra>"
                ))
        
        # Add actual data if available
        if actual_table_name:
            actual_data = self.connection_context.table(actual_table_name).collect()
            
            # Find actual value column
            actual_cols = [col for col in actual_data.columns if col != date_col]
            if actual_cols:
                actual_col = actual_cols[0]
                
                fig.add_trace(go.Scatter(
                    x=actual_data[date_col],
                    y=actual_data[actual_col],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="#ff7f0e", width=2),
                    hovertemplate=f"{date_col}: %{{x}}<br>{actual_col}: %{{y:.2f}}<extra></extra>"
                ))
        
        # Add interactive controls if requested
        if include_controls:
            # Add range slider
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            # Add buttons for different visualizations
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        buttons=[
                            dict(
                                args=[{"visible": [True, True, True, True, True]}],
                                label="Show All",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": [True, False, False, False, True if actual_table_name else False]}],
                                label="Forecast Only",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": [False, True, True, True, False]}],
                                label="Confidence Interval",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": [False, False, False, False, True] if actual_table_name else [False, False, False, False, False]}],
                                label="Actual Only",
                                method="update"
                            ),
                        ],
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    ),
                ]
            )
        
        # Calculate forecast statistics
        forecast_mean = predict_data[forecast_col].mean()
        forecast_std = predict_data[forecast_col].std()
        forecast_min = predict_data[forecast_col].min()
        forecast_max = predict_data[forecast_col].max()
        
        # Add annotations for forecast statistics
        stats_text = (f"Forecast Statistics:<br>"
                      f"Mean: {forecast_mean:.2f}<br>"
                      f"Std Dev: {forecast_std:.2f}<br>"
                      f"Min: {forecast_min:.2f}<br>"
                      f"Max: {forecast_max:.2f}")
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
        
        # Improve overall layout
        fig.update_layout(
            title=f"Interactive Forecast for {predict_result}",
            xaxis_title=date_col,
            yaxis_title=forecast_col,
            hovermode="closest",
            height=700,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            template=template,
            margin=dict(l=50, r=50, t=80, b=80),
        )
        
        # Save the enhanced interactive plot
        output_file_interactive = os.path.join(
            destination_dir,
            f"{predict_result}_interactive_forecast.html",
        )
        
        fig.write_html(
            output_file_interactive,
            include_plotlyjs="cdn",
            full_html=True,
            include_mathjax="cdn",
        )
        
        return json.dumps({
            "standard_html_file": str(Path(output_file_standard).as_posix()),
            "interactive_html_file": str(Path(output_file_interactive).as_posix())
        }, ensure_ascii=False)

    async def _arun(
        self, predict_result: str, actual_table_name: Optional[str]=None, 
        confidence: Optional[tuple]=None, output_dir: Optional[str]=None,
        theme: Optional[str]="light", include_controls: Optional[bool]=True,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            predict_result, actual_table_name, confidence, output_dir,
            theme, include_controls, run_manager=run_manager
        )

class InteractiveTimeSeriesPlot(BaseTool):
    """
    This tool generates highly interactive time series plots with customization options.
    
    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.
        
    Returns
    -------
    str
        The path of the generated interactive plot.
    """
    name: str = "interactive_time_series_plot"
    """Name of the tool."""
    description: str = "Create fully customizable interactive time series visualizations with multiple plot types and annotations."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = InteractivePlotInput
    """Input schema of the tool."""
    return_direct: bool = False
    bas: bool = False
    
    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )
        
    def set_bas(self, bas: bool) -> None:
        """
        Set the bas flag to True or False.
        """
        self.bas = bas
        
    def _run(
        self, table_name: str, time_column: str, value_columns: List[str], 
        plot_type: str = "line", title: Optional[str] = None,
        output_dir: Optional[str] = None, theme: Optional[str] = "light",
        height: Optional[int] = 600, width: Optional[int] = 900,
        annotations: Optional[List[Dict[str, Any]]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # Check if table exists
        if not self.connection_context.has_table(table_name):
            return json.dumps({"error": f"Table {table_name} does not exist."})
            
        # Check if columns exist
        df = self.connection_context.table(table_name)
        all_columns = [time_column] + value_columns
        missing_columns = [col for col in all_columns if col not in df.columns]
        
        if missing_columns:
            return json.dumps({"error": f"Columns {', '.join(missing_columns)} do not exist in table {table_name}."})
            
        # Fetch data
        select_columns = ", ".join(f'"{col}"' for col in all_columns)
        df_pd = df.select(all_columns).collect()
        
        # Create output directory
        if output_dir is None:
            destination_dir = os.path.join(tempfile.gettempdir(), "hanaml_visualizations")
        else:
            destination_dir = output_dir
            
        if not os.path.exists(destination_dir):
            try:
                os.makedirs(destination_dir, exist_ok=True)
            except Exception as e:
                logger.error("Error creating directory %s: %s", destination_dir, e)
                raise
        
        # Set title
        if title is None:
            title = f"Time Series Analysis for {table_name}"
            
        # Set theme
        template = "plotly_white"
        if theme == "dark":
            template = "plotly_dark"
        elif theme == "custom":
            template = {
                "layout": {
                    "colorway": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
                    "paper_bgcolor": "rgba(240, 240, 240, 0.95)",
                    "plot_bgcolor": "rgba(240, 240, 240, 0.95)",
                    "font": {"color": "#333333"},
                    "title": {"font": {"size": 24, "color": "#333333"}},
                }
            }
            
        # Create interactive figure based on plot type
        if plot_type.lower() == "line":
            fig = go.Figure()
            
            for col in value_columns:
                fig.add_trace(go.Scatter(
                    x=df_pd[time_column],
                    y=df_pd[col],
                    mode="lines+markers",
                    name=col,
                    hovertemplate=f"{time_column}: %{{x}}<br>{col}: %{{y:.2f}}<extra></extra>"
                ))
                
        elif plot_type.lower() == "bar":
            fig = go.Figure()
            
            for col in value_columns:
                fig.add_trace(go.Bar(
                    x=df_pd[time_column],
                    y=df_pd[col],
                    name=col,
                    hovertemplate=f"{time_column}: %{{x}}<br>{col}: %{{y:.2f}}<extra></extra>"
                ))
                
        elif plot_type.lower() == "area":
            fig = go.Figure()
            
            for col in value_columns:
                fig.add_trace(go.Scatter(
                    x=df_pd[time_column],
                    y=df_pd[col],
                    mode="lines",
                    name=col,
                    fill="tozeroy",
                    hovertemplate=f"{time_column}: %{{x}}<br>{col}: %{{y:.2f}}<extra></extra>"
                ))
                
        elif plot_type.lower() == "scatter":
            fig = go.Figure()
            
            for col in value_columns:
                fig.add_trace(go.Scatter(
                    x=df_pd[time_column],
                    y=df_pd[col],
                    mode="markers",
                    name=col,
                    marker=dict(size=8),
                    hovertemplate=f"{time_column}: %{{x}}<br>{col}: %{{y:.2f}}<extra></extra>"
                ))
                
        elif plot_type.lower() == "candlestick":
            # For candlestick, we need specific columns (open, high, low, close)
            ohlc_columns = ["open", "high", "low", "close"]
            matching_columns = {}
            
            for required in ohlc_columns:
                matches = [col for col in value_columns if required.lower() in col.lower()]
                if matches:
                    matching_columns[required] = matches[0]
                    
            if len(matching_columns) < 4:
                return json.dumps({
                    "error": f"Candlestick plot requires columns for open, high, low, and close values. Found: {list(matching_columns.values())}"
                })
                
            fig = go.Figure(go.Candlestick(
                x=df_pd[time_column],
                open=df_pd[matching_columns["open"]],
                high=df_pd[matching_columns["high"]],
                low=df_pd[matching_columns["low"]],
                close=df_pd[matching_columns["close"]],
                name="OHLC"
            ))
            
        else:
            return json.dumps({"error": f"Unsupported plot type: {plot_type}. Supported types: line, bar, area, scatter, candlestick"})
            
        # Add range slider and selectors
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Add custom annotations if provided
        if annotations:
            for annotation in annotations:
                fig.add_annotation(annotation)
                
        # Add statistics for each value column
        stats_text = "Statistics:<br>"
        for i, col in enumerate(value_columns):
            if i > 2:  # Limit to first 3 columns to avoid cluttering
                stats_text += "...<br>"
                break
                
            mean_val = df_pd[col].mean()
            std_val = df_pd[col].std()
            min_val = df_pd[col].min()
            max_val = df_pd[col].max()
            
            stats_text += (
                f"<b>{col}</b>:<br>"
                f"Mean: {mean_val:.2f}<br>"
                f"Std: {std_val:.2f}<br>"
                f"Range: [{min_val:.2f}, {max_val:.2f}]<br>"
            )
            
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
        
        # Add zoom and pan buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=[
                        dict(
                            args=[{"visible": [True] * len(value_columns)}],
                            label="Show All",
                            method="update"
                        ),
                    ] + [
                        dict(
                            args=[{"visible": [i == j for j in range(len(value_columns))]}],
                            label=f"Only {col}",
                            method="update"
                        ) for i, col in enumerate(value_columns)
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
            ]
        )
        
        # Improve overall layout
        fig.update_layout(
            title=title,
            xaxis_title=time_column,
            yaxis_title="Value",
            hovermode="closest",
            height=height,
            width=width,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            template=template,
            margin=dict(l=50, r=50, t=80, b=80),
        )
        
        # Generate unique filename
        uid = str(uuid.uuid4())[:8]
        output_file = os.path.join(
            destination_dir,
            f"{table_name}_{plot_type}_{uid}.html",
        )
        
        # Save the interactive visualization
        fig.write_html(
            output_file,
            include_plotlyjs="cdn",
            full_html=True,
            include_mathjax="cdn",
        )
        
        if not self.bas:
            fig.show()  # Display in jupyter if applicable
            
        return json.dumps({"html_file": str(Path(output_file).as_posix())}, ensure_ascii=False)
        
    async def _arun(
        self, table_name: str, time_column: str, value_columns: List[str], 
        plot_type: str = "line", title: Optional[str] = None,
        output_dir: Optional[str] = None, theme: Optional[str] = "light",
        height: Optional[int] = 600, width: Optional[int] = 900,
        annotations: Optional[List[Dict[str, Any]]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            table_name, time_column, value_columns, plot_type, title,
            output_dir, theme, height, width, annotations,
            run_manager=run_manager
        )

class CustomizableDashboard(BaseTool):
    """
    This tool generates a customizable dashboard with multiple visualizations.
    
    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.
        
    Returns
    -------
    str
        The path of the generated dashboard.
    """
    name: str = "customizable_dashboard"
    """Name of the tool."""
    description: str = "Create a comprehensive dashboard with multiple visualizations, interactive filters, and customizable layout."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = CustomizableDashboardInput
    """Input schema of the tool."""
    return_direct: bool = False
    bas: bool = False
    
    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )
        
    def set_bas(self, bas: bool) -> None:
        """
        Set the bas flag to True or False.
        """
        self.bas = bas
        
    def _run(
        self, table_names: List[str], layout: Optional[List[Dict[str, Any]]] = None,
        title: Optional[str] = "Time Series Dashboard", theme: Optional[str] = "light",
        output_dir: Optional[str] = None, include_filters: Optional[bool] = True,
        height: Optional[int] = 900, width: Optional[int] = 1200,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # Check if tables exist
        missing_tables = []
        for table_name in table_names:
            if not self.connection_context.has_table(table_name):
                missing_tables.append(table_name)
                
        if missing_tables:
            return json.dumps({"error": f"Tables do not exist: {', '.join(missing_tables)}"})
            
        # Create output directory
        if output_dir is None:
            destination_dir = os.path.join(tempfile.gettempdir(), "hanaml_dashboard")
        else:
            destination_dir = output_dir
            
        if not os.path.exists(destination_dir):
            try:
                os.makedirs(destination_dir, exist_ok=True)
            except Exception as e:
                logger.error("Error creating directory %s: %s", destination_dir, e)
                raise
                
        # Set theme
        template = "plotly_white"
        if theme == "dark":
            template = "plotly_dark"
        elif theme == "custom":
            template = {
                "layout": {
                    "colorway": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
                    "paper_bgcolor": "rgba(240, 240, 240, 0.95)",
                    "plot_bgcolor": "rgba(240, 240, 240, 0.95)",
                    "font": {"color": "#333333"},
                    "title": {"font": {"size": 24, "color": "#333333"}},
                }
            }
            
        # Determine layout based on number of tables
        if layout is None:
            num_tables = len(table_names)
            if num_tables == 1:
                # Single table - create multiple visualizations
                rows, cols = 2, 2
            elif num_tables <= 2:
                rows, cols = 1, 2
            elif num_tables <= 4:
                rows, cols = 2, 2
            elif num_tables <= 6:
                rows, cols = 2, 3
            else:
                rows, cols = 3, 3
                
            # Create default layout configuration
            layout = []
            for i, table_name in enumerate(table_names[:rows*cols]):
                row = i // cols + 1
                col = i % cols + 1
                layout.append({
                    "table": table_name,
                    "row": row,
                    "col": col,
                    "rowspan": 1,
                    "colspan": 1,
                    "plot_type": "line"
                })
        
        # Create subplot figure with specified layout
        fig = make_subplots(
            rows=max(item["row"] + item.get("rowspan", 1) - 1 for item in layout),
            cols=max(item["col"] + item.get("colspan", 1) - 1 for item in layout),
            subplot_titles=[item["table"] for item in layout],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            specs=[[{"colspan": item.get("colspan", 1), "rowspan": item.get("rowspan", 1)} 
                   if i+1 == item["col"] and j+1 == item["row"] else None 
                   for i in range(max(item["col"] + item.get("colspan", 1) - 1 for item in layout))]
                  for j in range(max(item["row"] + item.get("rowspan", 1) - 1 for item in layout))]
        )
        
        # Store data for interactive filtering
        all_data = {}
        
        # Process each table according to layout
        for item in layout:
            table_name = item["table"]
            row = item["row"]
            col = item["col"]
            plot_type = item.get("plot_type", "line")
            
            # Get table data
            df = self.connection_context.table(table_name)
            columns = df.columns
            
            # Determine time column and value columns
            time_cols = [col for col in columns if any(term in col.upper() for term in ["DATE", "TIME", "TS", "TIMESTAMP"])]
            if not time_cols:
                time_cols = [columns[0]]  # Default to first column if no date column found
                
            time_col = time_cols[0]
            value_cols = [col for col in columns if col != time_col]
            
            # Get data
            df_pd = df.collect()
            all_data[table_name] = df_pd
            
            # Add plot based on type
            if plot_type.lower() == "line":
                for val_col in value_cols[:3]:  # Limit to first 3 value columns
                    fig.add_trace(
                        go.Scatter(
                            x=df_pd[time_col],
                            y=df_pd[val_col],
                            mode="lines",
                            name=f"{table_name}.{val_col}",
                            hovertemplate=f"{time_col}: %{{x}}<br>{val_col}: %{{y:.2f}}<extra></extra>"
                        ),
                        row=row, col=col
                    )
                    
            elif plot_type.lower() == "bar":
                for val_col in value_cols[:3]:  # Limit to first 3 value columns
                    fig.add_trace(
                        go.Bar(
                            x=df_pd[time_col],
                            y=df_pd[val_col],
                            name=f"{table_name}.{val_col}",
                            hovertemplate=f"{time_col}: %{{x}}<br>{val_col}: %{{y:.2f}}<extra></extra>"
                        ),
                        row=row, col=col
                    )
                    
            elif plot_type.lower() == "area":
                for val_col in value_cols[:3]:  # Limit to first 3 value columns
                    fig.add_trace(
                        go.Scatter(
                            x=df_pd[time_col],
                            y=df_pd[val_col],
                            mode="lines",
                            name=f"{table_name}.{val_col}",
                            fill="tozeroy",
                            hovertemplate=f"{time_col}: %{{x}}<br>{val_col}: %{{y:.2f}}<extra></extra>"
                        ),
                        row=row, col=col
                    )
                    
            # Add range slider for time series
            fig.update_xaxes(
                rangeslider_visible=True,
                row=row, col=col
            )
        
        # Add dashboard title
        fig.update_layout(
            title_text=title,
            height=height,
            width=width,
            template=template,
            showlegend=True
        )
        
        # Add interactive dashboard controls
        if include_filters:
            # Generate HTML with interactive filters
            # This requires a custom HTML template with JavaScript for filter interactions
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
                    .dashboard-container {{ padding: 20px; }}
                    .filters {{ margin-bottom: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
                    .filter-group {{ margin-bottom: 10px; }}
                    .filter-group label {{ display: inline-block; width: 120px; font-weight: bold; }}
                    .filter-group select {{ width: 200px; padding: 5px; }}
                    .dashboard-title {{ text-align: center; margin-bottom: 20px; }}
                    .plot-container {{ margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="dashboard-container">
                    <h1 class="dashboard-title">{title}</h1>
                    
                    <div class="filters">
                        <h3>Dashboard Filters</h3>
                        <div class="filter-group">
                            <label for="table-filter">Table:</label>
                            <select id="table-filter">
                                <option value="all">All Tables</option>
                                {' '.join(f'<option value="{table}">{table}</option>' for table in table_names)}
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="date-range">Date Range:</label>
                            <select id="date-range">
                                <option value="all">All Time</option>
                                <option value="1w">Last Week</option>
                                <option value="1m">Last Month</option>
                                <option value="3m">Last 3 Months</option>
                                <option value="6m">Last 6 Months</option>
                                <option value="1y">Last Year</option>
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="view-type">View Type:</label>
                            <select id="view-type">
                                <option value="normal">Normal</option>
                                <option value="stacked">Stacked</option>
                                <option value="normalized">Normalized</option>
                            </select>
                        </div>
                        <button id="apply-filters">Apply Filters</button>
                        <button id="reset-filters">Reset</button>
                    </div>
                    
                    <div id="dashboard-plot" class="plot-container"></div>
                </div>
                
                <script>
                    // Initial plot render
                    var dashboardFigure = {fig.to_json()};
                    Plotly.newPlot('dashboard-plot', dashboardFigure.data, dashboardFigure.layout);
                    
                    // Filter functionality
                    $('#apply-filters').click(function() {
                        var tableFilter = $('#table-filter').val();
                        var dateRange = $('#date-range').val();
                        var viewType = $('#view-type').val();
                        
                        // Apply filters
                        var filteredData = dashboardFigure.data;
                        
                        if (tableFilter !== 'all') {
                            filteredData = filteredData.filter(trace => 
                                trace.name.startsWith(tableFilter + '.'));
                        }
                        
                        // Update layout based on view type
                        var updatedLayout = JSON.parse(JSON.stringify(dashboardFigure.layout));
                        
                        if (viewType === 'stacked') {
                            updatedLayout.barmode = 'stack';
                        } else if (viewType === 'normalized') {
                            updatedLayout.barmode = 'stack';
                            // Additional logic for normalization would go here
                        } else {
                            updatedLayout.barmode = 'group';
                        }
                        
                        Plotly.react('dashboard-plot', filteredData, updatedLayout);
                    });
                    
                    $('#reset-filters').click(function() {
                        $('#table-filter').val('all');
                        $('#date-range').val('all');
                        $('#view-type').val('normal');
                        
                        Plotly.react('dashboard-plot', dashboardFigure.data, dashboardFigure.layout);
                    });
                </script>
            </body>
            </html>
            """
            
            # Save the dashboard HTML
            output_file = os.path.join(
                destination_dir,
                f"interactive_dashboard_{str(uuid.uuid4())[:8]}.html",
            )
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(dashboard_html)
                
        else:
            # Save standard Plotly figure
            output_file = os.path.join(
                destination_dir,
                f"dashboard_{str(uuid.uuid4())[:8]}.html",
            )
            
            fig.write_html(
                output_file,
                include_plotlyjs="cdn",
                full_html=True,
                include_mathjax="cdn",
            )
            
        if not self.bas:
            fig.show()  # Display in jupyter if applicable
            
        return json.dumps({"html_file": str(Path(output_file).as_posix())}, ensure_ascii=False)
        
    async def _arun(
        self, table_names: List[str], layout: Optional[List[Dict[str, Any]]] = None,
        title: Optional[str] = "Time Series Dashboard", theme: Optional[str] = "light",
        output_dir: Optional[str] = None, include_filters: Optional[bool] = True,
        height: Optional[int] = 900, width: Optional[int] = 1200,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            table_names, layout, title, theme, output_dir, include_filters,
            height, width, run_manager=run_manager
        )
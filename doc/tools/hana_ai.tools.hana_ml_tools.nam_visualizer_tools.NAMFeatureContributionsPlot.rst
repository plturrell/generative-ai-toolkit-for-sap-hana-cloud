:py:mod:`hana_ai.tools.hana_ml_tools.nam_visualizer_tools.NAMFeatureContributionsPlot`
==================================================================================

.. py:class:: NAMFeatureContributionsPlot

   This tool creates elegant visualizations for Neural Additive Models that make complex model behaviors easy to understand and communicate.

   .. image:: ../../image/nam_visualizations.png
      :alt: NAM Visualization Gallery
      :width: 700px
      :align: center

   **Visualization Types**

   The tool provides three types of visualizations:

   1. **Feature Importance** - Ranks features by their overall impact on predictions

      .. image:: ../../image/nam_importance.png
         :alt: NAM Feature Importance
         :width: 400px
         :align: center

   2. **Feature Contributions** - Shows how each feature contributes to individual predictions

      .. image:: ../../image/nam_individual_contributions.png
         :alt: NAM Individual Contributions
         :width: 500px
         :align: center

   3. **Shape Functions** - Displays how each feature affects predictions across its range of values

      .. image:: ../../image/nam_shapes.png
         :alt: NAM Shape Functions
         :width: 600px
         :align: center

   **Visual Design**

   The visualizations are designed with these principles:

   * **Clarity** - Clear visual hierarchy and minimal chartjunk
   * **Consistency** - Consistent color schemes and layouts
   * **Accessibility** - Colorblind-friendly palettes available
   * **Interactivity** - Output can be returned as data for custom visualization

   **Parameters**

   * **connection_context** (*ConnectionContext*) – Connection context to the HANA database.

   **Returns**

   * **str** – Visualization data in the requested format.

     .. note::

         args_schema is used to define the schema of the inputs:

         .. list-table::
             :widths: 15 50
             :header-rows: 1

             * - Field
               - Description
             * - name
               - Name of the model
             * - version
               - Model version
             * - data_table
               - Table with sample data for visualization
             * - sample_size
               - Number of samples to visualize
             * - visualization_type
               - Type of visualization: 'feature_importance', 'contributions', 'shape_functions'
             * - output_format
               - Output format: 'image', 'html', or 'data'
             * - theme
               - Visual theme: 'light', 'dark', or 'colorblind'

   **Example Usage**

   .. code-block:: python

      # Create a feature importance visualization
      agent.run('''
          Create a visualization of feature importance for the "sales_predictor" model
          using a dark theme.
      ''')
      
      # Create a feature contributions visualization
      agent.run('''
          Show me how each feature contributes to the predictions for the first 5 samples
          in the NEW_SALES_DATA table using the "sales_predictor" model.
      ''')
      
      # Create a shape functions visualization
      agent.run('''
          Visualize how each feature affects predictions across its range of values
          for the "sales_predictor" model. Use the SALES_DATA table for feature ranges.
      ''')

   These visualizations can be used to communicate model behavior to stakeholders, explain predictions to end-users, and validate model behavior during development.
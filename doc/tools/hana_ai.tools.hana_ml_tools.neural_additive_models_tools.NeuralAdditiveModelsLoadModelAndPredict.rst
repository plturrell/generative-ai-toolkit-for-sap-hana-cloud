:py:mod:`hana_ai.tools.hana_ml_tools.neural_additive_models_tools.NeuralAdditiveModelsLoadModelAndPredict`
==================================================================================

.. py:class:: NeuralAdditiveModelsLoadModelAndPredict

   This tool generates predictions using a trained Neural Additive Model, with options to visualize feature contributions for enhanced transparency.

   .. image:: ../../image/nam_predictions.png
      :alt: NAM Prediction Visualization
      :width: 600px
      :align: center

   **Visual Explainability**

   One of the most powerful aspects of Neural Additive Models is their ability to provide a clear breakdown of how each feature contributes to a prediction. This makes it easy to understand why the model made a specific prediction and which features had the most impact.

   .. image:: ../../image/nam_contributions.png
      :alt: Feature Contributions Visualization
      :width: 600px
      :align: center

   **Parameters**

   * **connection_context** (*ConnectionContext*) – Connection context to the HANA database.

   **Returns**

   * **str** – Prediction results information in JSON format.

     .. note::

         args_schema is used to define the schema of the inputs:

         .. list-table::
             :widths: 15 50
             :header-rows: 1

             * - Field
               - Description
             * - predict_table
               - Table containing data to predict
             * - name
               - Name of the model
             * - version
               - Model version
             * - features
               - Feature columns to use (comma-separated, uses training features if omitted)
             * - include_contributions
               - Include individual feature contributions
             * - output_format
               - Output format: 'standard' or 'detailed'

   **Example Usage**

   .. code-block:: python

      # Make predictions with feature contributions
      result = agent.run('''
          Generate predictions using the "sales_predictor" model on the NEW_SALES_DATA table.
          Include feature contributions to understand which factors influence the predictions.
      ''')
      
      # Create visualization from the feature contributions
      agent.run('''
          Create a visualization showing how each feature contributes to the predictions
          for the "sales_predictor" model.
      ''')

   .. image:: ../../image/nam_shape_functions.png
      :alt: NAM Shape Functions
      :width: 600px
      :align: center
      
   The above visualization shows how each feature affects the prediction across its range of values. This enables business users to understand not just which features are important, but exactly how they affect the outcome.
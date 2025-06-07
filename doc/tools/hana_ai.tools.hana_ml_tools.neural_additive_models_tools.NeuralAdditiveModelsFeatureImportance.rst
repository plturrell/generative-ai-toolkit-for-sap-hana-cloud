:py:mod:`hana_ai.tools.hana_ml_tools.neural_additive_models_tools.NeuralAdditiveModelsFeatureImportance`
====================================================================================

.. py:class:: NeuralAdditiveModelsFeatureImportance

   This tool is used to get feature importance scores from a neural additive model.

   A key advantage of Neural Additive Models is their interpretability. This tool extracts and
   ranks the importance of each feature, helping users understand which inputs most significantly
   impact the model's predictions.

   **Parameters**

   * **connection_context** (*ConnectionContext*) – Connection context to the HANA database.

   **Returns**

   * **str** – Feature importance information in JSON format.

     .. note::

         args_schema is used to define the schema of the inputs as follows:

         .. list-table::
             :widths: 15 50
             :header-rows: 1

             * - Field
               - Description
             * - name
               - The name of the model.
             * - version
               - The version of the model.
             * - top_n
               - Show only the top N most important features.
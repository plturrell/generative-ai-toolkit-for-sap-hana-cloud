:py:mod:`hana_ai.tools.hana_ml_tools.neural_additive_models_tools.NeuralAdditiveModelsFitAndSave`
===============================================================================

.. py:class:: NeuralAdditiveModelsFitAndSave

   This tool trains a Neural Additive Model (NAM) on your HANA data and saves it to model storage.

   .. image:: ../../image/nam_architecture.png
      :alt: Neural Additive Model Architecture
      :width: 600px
      :align: center

   **Key Benefits of Neural Additive Models**

   * **Interpretability with Power:** NAMs provide feature-level transparency similar to linear models, but with neural networks' ability to capture complex patterns
   * **Visual Explainability:** Each feature's contribution can be visualized independently, making the model easy to explain to business users
   * **Competitive Performance:** NAMs often outperform traditional GAMs while approaching the accuracy of black-box neural networks

   **How It Works**

   Neural Additive Models use a separate neural network for each feature. This allows complex non-linear relationships to be captured while maintaining interpretability. The overall prediction is simply the sum of each feature network's output plus a bias term.

   **Parameters**

   * **connection_context** (*ConnectionContext*) – Connection context to the HANA database.

   **Returns**

   * **str** – The result string containing model information.

     .. note::

         args_schema is used to define the schema of the inputs:

         .. list-table::
             :widths: 15 50
             :header-rows: 1

             * - Field
               - Description
             * - fit_table
               - Table containing training data
             * - name
               - Name for the model in storage
             * - target
               - Target column to predict
             * - version
               - Model version in storage (auto-incremented if omitted)
             * - features
               - Feature columns to use (comma-separated, uses all columns except target if omitted)
             * - complexity
               - Model complexity: 'simple', 'balanced', or 'complex'
             * - training_mode
               - Training mode: 'fast', 'balanced', or 'thorough'
             * - include_interpretability
               - Enable advanced interpretability features

   **Example Usage**

   .. code-block:: python

      # Train a NAM model with balanced complexity
      result = agent.run('''
          Train a neural additive model on the SALES_DATA table.
          Use "REVENUE" as the target column.
          Name the model "sales_predictor".
          Use "balanced" complexity and "thorough" training mode.
      ''')
      
      # Print model information
      print(result)
      
   .. image:: ../../image/nam_training.png
      :alt: NAM Training Process
      :width: 600px
      :align: center
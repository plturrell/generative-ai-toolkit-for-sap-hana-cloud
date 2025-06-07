/**
 * Neural Additive Models Design System
 * 
 * This is the main entry point for the NAM design system that connects
 * the backend Python implementation with the React-based frontend.
 * It provides a cohesive, emotionally resonant experience for users
 * interacting with Neural Additive Models.
 */

import React, { useState, useEffect, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { designTokens } from './design_tokens';
import { ComplexitySelector } from './complexity_selector';
import { FeatureContributions } from './feature_contributions';
import { TrainingVisualizer } from './training_visualizer';
import { ModelExplorer } from './model_explorer';

/**
 * NAM Application
 * 
 * The main component that integrates all aspects of the Neural Additive Models
 * experience into a cohesive, emotionally intelligent interface.
 */
const NAMApp = ({ 
  initialData = null,
  apiEndpoint = '/api/nam',
  theme = 'light',
  onTrainingComplete = () => {},
  onPrediction = () => {}
}) => {
  // Application state
  const [activeView, setActiveView] = useState('configure'); // configure, train, predict, explore
  const [modelConfig, setModelConfig] = useState({
    complexity: 'balanced',
    trainingMode: 'balanced',
    includeInterpretability: true
  });
  const [trainingStatus, setTrainingStatus] = useState({
    isTraining: false,
    progress: 0,
    modelName: '',
    modelVersion: null
  });
  const [predictionData, setPredictionData] = useState(null);
  const [featureImportance, setFeatureImportance] = useState([]);
  
  // Handle switching between views with animations
  const navigateTo = (view) => {
    if (view === activeView) return;
    
    // Provide subtle haptic feedback for navigation
    if (window.navigator && window.navigator.vibrate) {
      window.navigator.vibrate(5);
    }
    
    setActiveView(view);
  };
  
  // Handle model configuration changes
  const handleConfigChange = (key, value) => {
    setModelConfig(prev => ({
      ...prev,
      [key]: value
    }));
  };
  
  // Handle training initiation
  const startTraining = async (trainingParams) => {
    setTrainingStatus({
      isTraining: true,
      progress: 0,
      modelName: trainingParams.name || 'nam_model',
      modelVersion: null
    });
    
    navigateTo('train');
    
    // Training would be handled via API calls to the Python backend
    // This is a simplified representation
    try {
      // Simulated training progress updates
      const progressInterval = setInterval(() => {
        setTrainingStatus(prev => ({
          ...prev,
          progress: Math.min(prev.progress + 2, 98) // Cap at 98% until complete
        }));
      }, 500);
      
      // Simulate API call to Python backend
      const response = await fetch(`${apiEndpoint}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...trainingParams,
          complexity: modelConfig.complexity,
          training_mode: modelConfig.trainingMode,
          include_interpretability: modelConfig.includeInterpretability
        })
      });
      
      clearInterval(progressInterval);
      
      if (response.ok) {
        const result = await response.json();
        setTrainingStatus({
          isTraining: false,
          progress: 100,
          modelName: result.model.name,
          modelVersion: result.model.version
        });
        
        // Fetch feature importance after training
        await fetchFeatureImportance(result.model.name, result.model.version);
        
        onTrainingComplete(result);
      } else {
        throw new Error('Training failed');
      }
    } catch (error) {
      console.error('Error during training:', error);
      setTrainingStatus(prev => ({
        ...prev,
        isTraining: false
      }));
    }
  };
  
  // Fetch feature importance from the backend
  const fetchFeatureImportance = async (modelName, modelVersion) => {
    try {
      const response = await fetch(`${apiEndpoint}/feature-importance?name=${modelName}&version=${modelVersion}`);
      
      if (response.ok) {
        const result = await response.json();
        
        // Transform to the format expected by our components
        const formattedImportance = result.feature_importance.map(item => ({
          name: item.feature,
          contribution: item.importance,
          // In a real implementation, we would include explanations
          explanation: `This feature is ${item.importance > 0.2 ? 'very important' : 'somewhat important'} for predictions.`
        }));
        
        setFeatureImportance(formattedImportance);
      }
    } catch (error) {
      console.error('Error fetching feature importance:', error);
    }
  };
  
  // Make predictions with the trained model
  const makePrediction = async (predictionParams) => {
    try {
      const response = await fetch(`${apiEndpoint}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...predictionParams,
          name: trainingStatus.modelName,
          version: trainingStatus.modelVersion,
          include_contributions: true
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        setPredictionData(result);
        navigateTo('predict');
        onPrediction(result);
      }
    } catch (error) {
      console.error('Error making prediction:', error);
    }
  };
  
  // Define animation variants
  const pageVariants = {
    initial: {
      opacity: 0,
      y: 20
    },
    animate: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.4,
        ease: designTokens.animation.easing.entrance
      }
    },
    exit: {
      opacity: 0,
      y: -20,
      transition: {
        duration: 0.3,
        ease: designTokens.animation.easing.exit
      }
    }
  };
  
  return (
    <div 
      className="nam-application"
      style={{
        fontFamily: designTokens.typography.fontFamily.primary,
        color: theme === 'dark' ? designTokens.colors.neutral[200] : designTokens.colors.neutral[800],
        backgroundColor: theme === 'dark' ? designTokens.colors.neutral[900] : designTokens.colors.neutral[100],
        padding: designTokens.spacing[6],
        borderRadius: designTokens.borderRadius.xl,
        boxShadow: designTokens.elevation.shadow.xl,
        maxWidth: '960px',
        margin: '0 auto'
      }}
    >
      {/* Navigation */}
      <nav style={{
        display: 'flex',
        gap: designTokens.spacing[2],
        marginBottom: designTokens.spacing[6]
      }}>
        {[
          { id: 'configure', label: 'Configure' },
          { id: 'train', label: 'Train', disabled: !modelConfig },
          { id: 'predict', label: 'Predict', disabled: !trainingStatus.modelName },
          { id: 'explore', label: 'Explore', disabled: !trainingStatus.modelName }
        ].map(item => (
          <button
            key={item.id}
            onClick={() => !item.disabled && navigateTo(item.id)}
            disabled={item.disabled}
            style={{
              background: 'none',
              border: 'none',
              borderBottom: `2px solid ${activeView === item.id 
                ? designTokens.colors.primary.base 
                : 'transparent'}`,
              color: activeView === item.id
                ? (theme === 'dark' ? designTokens.colors.primary.light : designTokens.colors.primary.base)
                : (theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600]),
              opacity: item.disabled ? 0.5 : 1,
              padding: `${designTokens.spacing[2]} ${designTokens.spacing[3]}`,
              fontSize: designTokens.typography.fontSize.md,
              fontWeight: activeView === item.id 
                ? designTokens.typography.fontWeight.semibold 
                : designTokens.typography.fontWeight.medium,
              cursor: item.disabled ? 'not-allowed' : 'pointer',
              transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
            }}
          >
            {item.label}
          </button>
        ))}
      </nav>
      
      {/* Main content area with view transitions */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeView}
          initial="initial"
          animate="animate"
          exit="exit"
          variants={pageVariants}
          style={{
            minHeight: '400px'
          }}
        >
          {activeView === 'configure' && (
            <ConfigureView 
              config={modelConfig}
              onChange={handleConfigChange}
              onStartTraining={startTraining}
              theme={theme}
            />
          )}
          
          {activeView === 'train' && (
            <TrainingView 
              status={trainingStatus}
              theme={theme}
              onComplete={() => navigateTo('predict')}
            />
          )}
          
          {activeView === 'predict' && (
            <PredictView 
              modelName={trainingStatus.modelName}
              modelVersion={trainingStatus.modelVersion}
              predictionData={predictionData}
              featureImportance={featureImportance}
              onMakePrediction={makePrediction}
              theme={theme}
            />
          )}
          
          {activeView === 'explore' && (
            <ExploreView 
              modelName={trainingStatus.modelName}
              modelVersion={trainingStatus.modelVersion}
              featureImportance={featureImportance}
              theme={theme}
            />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

/**
 * Configuration View
 * 
 * Allows users to configure model parameters with intuitive controls.
 */
const ConfigureView = ({ 
  config, 
  onChange, 
  onStartTraining,
  theme
}) => {
  const [tableOptions, setTableOptions] = useState([]);
  const [selectedTable, setSelectedTable] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [modelName, setModelName] = useState('nam_model');
  
  // Fetch available tables (would be implemented with actual API calls)
  useEffect(() => {
    // Simulated API call to get tables
    const fetchTables = async () => {
      // In a real implementation, this would be an API call
      setTableOptions([
        { name: 'SALES_DATA', columns: ['ID', 'PRODUCT', 'REGION', 'QUANTITY', 'PRICE', 'TOTAL'] },
        { name: 'CUSTOMER_INFO', columns: ['CUSTOMER_ID', 'NAME', 'AGE', 'LOCATION', 'SEGMENT', 'LOYALTY_SCORE'] },
        { name: 'MARKETING_CAMPAIGNS', columns: ['CAMPAIGN_ID', 'CHANNEL', 'BUDGET', 'IMPRESSIONS', 'CLICKS', 'CONVERSIONS'] }
      ]);
    };
    
    fetchTables();
  }, []);
  
  // Handle table selection
  const handleTableChange = (event) => {
    const selected = event.target.value;
    setSelectedTable(selected);
    setTargetColumn(''); // Reset target column when table changes
  };
  
  // Get columns for selected table
  const getColumnsForTable = () => {
    const table = tableOptions.find(t => t.name === selectedTable);
    return table ? table.columns : [];
  };
  
  // Handle form submission
  const handleSubmit = (event) => {
    event.preventDefault();
    
    if (!selectedTable || !targetColumn) return;
    
    onStartTraining({
      fit_table: selectedTable,
      target: targetColumn,
      name: modelName
    });
  };
  
  return (
    <div className="nam-configure-view">
      <h2 style={{
        fontSize: designTokens.typography.fontSize.xl,
        fontWeight: designTokens.typography.fontWeight.semibold,
        marginBottom: designTokens.spacing[4]
      }}>
        Configure Your Neural Additive Model
      </h2>
      
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: designTokens.spacing[6] }}>
          <label 
            htmlFor="model-name"
            style={{
              display: 'block',
              fontSize: designTokens.typography.fontSize.md,
              fontWeight: designTokens.typography.fontWeight.medium,
              marginBottom: designTokens.spacing[2]
            }}
          >
            Model Name
          </label>
          <input
            id="model-name"
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            style={{
              width: '100%',
              padding: `${designTokens.spacing[2]} ${designTokens.spacing[3]}`,
              fontSize: designTokens.typography.fontSize.base,
              borderRadius: designTokens.borderRadius.md,
              border: `1px solid ${theme === 'dark' ? designTokens.colors.neutral[600] : designTokens.colors.neutral[400]}`,
              backgroundColor: theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[100],
              color: theme === 'dark' ? designTokens.colors.neutral[200] : designTokens.colors.neutral[800],
              marginBottom: designTokens.spacing[4]
            }}
            required
          />
          
          <div style={{ 
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: designTokens.spacing[4],
            marginBottom: designTokens.spacing[4]
          }}>
            <div>
              <label 
                htmlFor="data-table"
                style={{
                  display: 'block',
                  fontSize: designTokens.typography.fontSize.md,
                  fontWeight: designTokens.typography.fontWeight.medium,
                  marginBottom: designTokens.spacing[2]
                }}
              >
                Data Table
              </label>
              <select
                id="data-table"
                value={selectedTable}
                onChange={handleTableChange}
                style={{
                  width: '100%',
                  padding: `${designTokens.spacing[2]} ${designTokens.spacing[3]}`,
                  fontSize: designTokens.typography.fontSize.base,
                  borderRadius: designTokens.borderRadius.md,
                  border: `1px solid ${theme === 'dark' ? designTokens.colors.neutral[600] : designTokens.colors.neutral[400]}`,
                  backgroundColor: theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[100],
                  color: theme === 'dark' ? designTokens.colors.neutral[200] : designTokens.colors.neutral[800]
                }}
                required
              >
                <option value="">Select a table</option>
                {tableOptions.map(table => (
                  <option key={table.name} value={table.name}>{table.name}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label 
                htmlFor="target-column"
                style={{
                  display: 'block',
                  fontSize: designTokens.typography.fontSize.md,
                  fontWeight: designTokens.typography.fontWeight.medium,
                  marginBottom: designTokens.spacing[2]
                }}
              >
                Target Column
              </label>
              <select
                id="target-column"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                disabled={!selectedTable}
                style={{
                  width: '100%',
                  padding: `${designTokens.spacing[2]} ${designTokens.spacing[3]}`,
                  fontSize: designTokens.typography.fontSize.base,
                  borderRadius: designTokens.borderRadius.md,
                  border: `1px solid ${theme === 'dark' ? designTokens.colors.neutral[600] : designTokens.colors.neutral[400]}`,
                  backgroundColor: theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[100],
                  color: theme === 'dark' ? designTokens.colors.neutral[200] : designTokens.colors.neutral[800],
                  opacity: !selectedTable ? 0.6 : 1
                }}
                required
              >
                <option value="">Select target column</option>
                {getColumnsForTable().map(column => (
                  <option key={column} value={column}>{column}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
        
        {/* Complexity Selector Component */}
        <div style={{ marginBottom: designTokens.spacing[6] }}>
          <ComplexitySelector 
            value={config.complexity}
            onChange={(value) => onChange('complexity', value)}
          />
        </div>
        
        {/* Training Mode Selector */}
        <div style={{ marginBottom: designTokens.spacing[6] }}>
          <label 
            htmlFor="training-mode"
            style={{
              display: 'block',
              fontSize: designTokens.typography.fontSize.md,
              fontWeight: designTokens.typography.fontWeight.medium,
              marginBottom: designTokens.spacing[2]
            }}
          >
            Training Mode
          </label>
          
          <div className="nam-training-mode-options" style={{
            display: 'flex',
            gap: designTokens.spacing[2],
            marginBottom: designTokens.spacing[2]
          }}>
            {['fast', 'balanced', 'thorough'].map(mode => (
              <button
                key={mode}
                type="button"
                onClick={() => onChange('trainingMode', mode)}
                style={{
                  flex: 1,
                  padding: `${designTokens.spacing[3]} ${designTokens.spacing[2]}`,
                  backgroundColor: config.trainingMode === mode
                    ? (theme === 'dark' ? designTokens.colors.primary.dark : designTokens.colors.primary.light)
                    : (theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[200]),
                  color: config.trainingMode === mode
                    ? (theme === 'dark' ? designTokens.colors.neutral[100] : designTokens.colors.primary.dark)
                    : (theme === 'dark' ? designTokens.colors.neutral[300] : designTokens.colors.neutral[700]),
                  border: `1px solid ${config.trainingMode === mode
                    ? (theme === 'dark' ? designTokens.colors.primary.base : designTokens.colors.primary.base)
                    : 'transparent'}`,
                  borderRadius: designTokens.borderRadius.md,
                  fontSize: designTokens.typography.fontSize.sm,
                  fontWeight: config.trainingMode === mode
                    ? designTokens.typography.fontWeight.semibold
                    : designTokens.typography.fontWeight.regular,
                  textTransform: 'capitalize',
                  cursor: 'pointer',
                  transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
                }}
              >
                {mode}
              </button>
            ))}
          </div>
          
          <div style={{ 
            fontSize: designTokens.typography.fontSize.sm,
            color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
          }}>
            {config.trainingMode === 'fast' && "Prioritizes speed with fewer training epochs. Good for quick exploration."}
            {config.trainingMode === 'balanced' && "Optimal balance between training time and model performance for most use cases."}
            {config.trainingMode === 'thorough' && "Extended training for maximum accuracy. Takes longer but may produce better results."}
          </div>
        </div>
        
        {/* Interpretability Toggle */}
        <div style={{ marginBottom: designTokens.spacing[8] }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <label 
              htmlFor="interpretability"
              style={{
                fontSize: designTokens.typography.fontSize.md,
                fontWeight: designTokens.typography.fontWeight.medium
              }}
            >
              Enable Advanced Interpretability
            </label>
            
            <button
              id="interpretability"
              type="button"
              onClick={() => onChange('includeInterpretability', !config.includeInterpretability)}
              aria-pressed={config.includeInterpretability}
              style={{
                width: '50px',
                height: '28px',
                backgroundColor: config.includeInterpretability
                  ? designTokens.colors.primary.base
                  : (theme === 'dark' ? designTokens.colors.neutral[600] : designTokens.colors.neutral[400]),
                borderRadius: '14px',
                position: 'relative',
                border: 'none',
                cursor: 'pointer',
                transition: `background-color ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
              }}
            >
              <span style={{
                position: 'absolute',
                top: '4px',
                left: config.includeInterpretability ? '26px' : '4px',
                width: '20px',
                height: '20px',
                backgroundColor: designTokens.colors.neutral[100],
                borderRadius: '50%',
                transition: `left ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
              }} />
            </button>
          </div>
          
          <div style={{ 
            fontSize: designTokens.typography.fontSize.sm,
            color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
            marginTop: designTokens.spacing[2]
          }}>
            Enables detailed feature contribution analysis and interactive visualizations.
            This may slightly increase model complexity but provides valuable insights.
          </div>
        </div>
        
        {/* Submit Button */}
        <div style={{ textAlign: 'center' }}>
          <motion.button
            type="submit"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            style={{
              backgroundColor: designTokens.colors.primary.base,
              color: designTokens.colors.primary.contrast,
              border: 'none',
              borderRadius: designTokens.borderRadius.md,
              padding: `${designTokens.spacing[3]} ${designTokens.spacing[6]}`,
              fontSize: designTokens.typography.fontSize.md,
              fontWeight: designTokens.typography.fontWeight.semibold,
              cursor: 'pointer',
              boxShadow: designTokens.elevation.shadow.md,
              transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
            }}
            disabled={!selectedTable || !targetColumn}
          >
            Train Model
          </motion.button>
        </div>
      </form>
    </div>
  );
};

/**
 * Training View
 * 
 * Visualizes the training process with elegant animations.
 */
const TrainingView = ({ status, theme, onComplete }) => {
  useEffect(() => {
    if (status.progress === 100) {
      // Use a small delay to show 100% state before transitioning
      const timer = setTimeout(() => {
        onComplete();
      }, 1500);
      
      return () => clearTimeout(timer);
    }
  }, [status.progress, onComplete]);
  
  return (
    <div className="nam-training-view" style={{
      textAlign: 'center',
      padding: designTokens.spacing[6]
    }}>
      <h2 style={{
        fontSize: designTokens.typography.fontSize.xl,
        fontWeight: designTokens.typography.fontWeight.semibold,
        marginBottom: designTokens.spacing[6]
      }}>
        {status.progress < 100 ? 'Training Your Model' : 'Training Complete!'}
      </h2>
      
      <div style={{ 
        maxWidth: '600px',
        margin: '0 auto',
        marginBottom: designTokens.spacing[6]
      }}>
        {/* Progress visualization would go here */}
        {/* In a real implementation, this would be replaced with TrainingVisualizer component */}
        <div style={{
          position: 'relative',
          height: '8px',
          backgroundColor: theme === 'dark' ? designTokens.colors.neutral[700] : designTokens.colors.neutral[300],
          borderRadius: designTokens.borderRadius.full,
          overflow: 'hidden',
          marginBottom: designTokens.spacing[2]
        }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${status.progress}%` }}
            style={{
              height: '100%',
              backgroundColor: status.progress === 100
                ? designTokens.colors.feedback.success
                : designTokens.colors.primary.base,
              borderRadius: designTokens.borderRadius.full
            }}
            transition={{
              duration: 0.5,
              ease: designTokens.animation.easing.standard
            }}
          />
        </div>
        
        <div style={{ 
          fontSize: designTokens.typography.fontSize.md,
          color: theme === 'dark' ? designTokens.colors.neutral[300] : designTokens.colors.neutral[700]
        }}>
          {status.progress < 100 ? `${status.progress}% Complete` : 'Model trained successfully!'}
        </div>
      </div>
      
      {status.progress === 100 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.4 }}
        >
          <h3 style={{
            fontSize: designTokens.typography.fontSize.lg,
            fontWeight: designTokens.typography.fontWeight.medium,
            marginBottom: designTokens.spacing[2]
          }}>
            Model Information
          </h3>
          
          <div style={{
            display: 'inline-block',
            backgroundColor: theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[200],
            padding: designTokens.spacing[3],
            borderRadius: designTokens.borderRadius.md,
            marginBottom: designTokens.spacing[4]
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              gap: designTokens.spacing[4]
            }}>
              <div>
                <div style={{ 
                  fontSize: designTokens.typography.fontSize.sm,
                  color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
                  marginBottom: designTokens.spacing[1]
                }}>
                  Model Name
                </div>
                <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
                  {status.modelName}
                </div>
              </div>
              
              <div>
                <div style={{ 
                  fontSize: designTokens.typography.fontSize.sm,
                  color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
                  marginBottom: designTokens.spacing[1]
                }}>
                  Version
                </div>
                <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
                  {status.modelVersion || 1}
                </div>
              </div>
            </div>
          </div>
          
          <motion.button
            onClick={onComplete}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{
              backgroundColor: designTokens.colors.primary.base,
              color: designTokens.colors.primary.contrast,
              border: 'none',
              borderRadius: designTokens.borderRadius.md,
              padding: `${designTokens.spacing[2]} ${designTokens.spacing[4]}`,
              fontSize: designTokens.typography.fontSize.md,
              fontWeight: designTokens.typography.fontWeight.medium,
              cursor: 'pointer',
              boxShadow: designTokens.elevation.shadow.md
            }}
          >
            Continue to Predictions
          </motion.button>
        </motion.div>
      )}
    </div>
  );
};

/**
 * Prediction View
 * 
 * Allows users to make predictions and visualize feature contributions.
 */
const PredictView = ({ 
  modelName, 
  modelVersion, 
  predictionData,
  featureImportance,
  onMakePrediction,
  theme
}) => {
  const [tableOptions, setTableOptions] = useState([]);
  const [selectedTable, setSelectedTable] = useState('');
  const [sampleSize, setSampleSize] = useState(5);
  const [isLoading, setIsLoading] = useState(false);
  
  // Simulated API call to get tables
  useEffect(() => {
    setTableOptions([
      { name: 'SALES_DATA', columns: ['ID', 'PRODUCT', 'REGION', 'QUANTITY', 'PRICE', 'TOTAL'] },
      { name: 'CUSTOMER_INFO', columns: ['CUSTOMER_ID', 'NAME', 'AGE', 'LOCATION', 'SEGMENT', 'LOYALTY_SCORE'] },
      { name: 'MARKETING_CAMPAIGNS', columns: ['CAMPAIGN_ID', 'CHANNEL', 'BUDGET', 'IMPRESSIONS', 'CLICKS', 'CONVERSIONS'] }
    ]);
  }, []);
  
  // Handle prediction request
  const handlePredict = async (e) => {
    e.preventDefault();
    
    if (!selectedTable) return;
    
    setIsLoading(true);
    
    try {
      await onMakePrediction({
        predict_table: selectedTable,
        sample_size: sampleSize
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Transform prediction data for visualization
  const getContributionData = () => {
    if (!predictionData || !predictionData.sample_predictions) {
      return [];
    }
    
    // In a real implementation, this would transform the actual API response
    // Here we're using a simplified mockup based on feature importance
    return featureImportance.map(feature => ({
      name: feature.name,
      contribution: feature.contribution * 2.5, // Scale for visualization
      explanation: feature.explanation
    }));
  };
  
  return (
    <div className="nam-predict-view">
      <h2 style={{
        fontSize: designTokens.typography.fontSize.xl,
        fontWeight: designTokens.typography.fontWeight.semibold,
        marginBottom: designTokens.spacing[4]
      }}>
        Make Predictions
      </h2>
      
      {!predictionData ? (
        <form onSubmit={handlePredict}>
          <div style={{ marginBottom: designTokens.spacing[4] }}>
            <label 
              htmlFor="predict-table"
              style={{
                display: 'block',
                fontSize: designTokens.typography.fontSize.md,
                fontWeight: designTokens.typography.fontWeight.medium,
                marginBottom: designTokens.spacing[2]
              }}
            >
              Prediction Data Table
            </label>
            <select
              id="predict-table"
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
              style={{
                width: '100%',
                padding: `${designTokens.spacing[2]} ${designTokens.spacing[3]}`,
                fontSize: designTokens.typography.fontSize.base,
                borderRadius: designTokens.borderRadius.md,
                border: `1px solid ${theme === 'dark' ? designTokens.colors.neutral[600] : designTokens.colors.neutral[400]}`,
                backgroundColor: theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[100],
                color: theme === 'dark' ? designTokens.colors.neutral[200] : designTokens.colors.neutral[800],
                marginBottom: designTokens.spacing[4]
              }}
              required
            >
              <option value="">Select a table</option>
              {tableOptions.map(table => (
                <option key={table.name} value={table.name}>{table.name}</option>
              ))}
            </select>
            
            <label 
              htmlFor="sample-size"
              style={{
                display: 'block',
                fontSize: designTokens.typography.fontSize.md,
                fontWeight: designTokens.typography.fontWeight.medium,
                marginBottom: designTokens.spacing[2]
              }}
            >
              Number of samples
            </label>
            <input
              id="sample-size"
              type="number"
              min="1"
              max="20"
              value={sampleSize}
              onChange={(e) => setSampleSize(parseInt(e.target.value, 10))}
              style={{
                width: '100%',
                padding: `${designTokens.spacing[2]} ${designTokens.spacing[3]}`,
                fontSize: designTokens.typography.fontSize.base,
                borderRadius: designTokens.borderRadius.md,
                border: `1px solid ${theme === 'dark' ? designTokens.colors.neutral[600] : designTokens.colors.neutral[400]}`,
                backgroundColor: theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[100],
                color: theme === 'dark' ? designTokens.colors.neutral[200] : designTokens.colors.neutral[800]
              }}
            />
          </div>
          
          <div style={{ textAlign: 'center', marginTop: designTokens.spacing[6] }}>
            <motion.button
              type="submit"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              disabled={isLoading || !selectedTable}
              style={{
                backgroundColor: designTokens.colors.primary.base,
                color: designTokens.colors.primary.contrast,
                border: 'none',
                borderRadius: designTokens.borderRadius.md,
                padding: `${designTokens.spacing[3]} ${designTokens.spacing[6]}`,
                fontSize: designTokens.typography.fontSize.md,
                fontWeight: designTokens.typography.fontWeight.semibold,
                cursor: isLoading || !selectedTable ? 'not-allowed' : 'pointer',
                opacity: isLoading || !selectedTable ? 0.7 : 1,
                boxShadow: designTokens.elevation.shadow.md,
                transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
              }}
            >
              {isLoading ? 'Processing...' : 'Generate Predictions'}
            </motion.button>
          </div>
        </form>
      ) : (
        <div>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: designTokens.spacing[4]
          }}>
            <h3 style={{
              fontSize: designTokens.typography.fontSize.lg,
              fontWeight: designTokens.typography.fontWeight.semibold,
              margin: 0
            }}>
              Prediction Results
            </h3>
            
            <motion.button
              onClick={() => setSelectedTable('')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              style={{
                backgroundColor: 'transparent',
                color: designTokens.colors.primary.base,
                border: `1px solid ${designTokens.colors.primary.base}`,
                borderRadius: designTokens.borderRadius.md,
                padding: `${designTokens.spacing[1]} ${designTokens.spacing[3]}`,
                fontSize: designTokens.typography.fontSize.sm,
                fontWeight: designTokens.typography.fontWeight.medium,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: designTokens.spacing[1]
              }}
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 12L4 8m0 0l4-4M4 8h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              New Prediction
            </motion.button>
          </div>
          
          {/* Feature Contributions Visualization */}
          <FeatureContributions
            data={getContributionData()}
            basePrediction={0.35} // In a real implementation, this would come from the API
            theme={theme}
          />
        </div>
      )}
    </div>
  );
};

/**
 * Explore View
 * 
 * Provides deeper insights into model behavior and feature relationships.
 */
const ExploreView = ({ modelName, modelVersion, featureImportance, theme }) => {
  return (
    <div className="nam-explore-view">
      <h2 style={{
        fontSize: designTokens.typography.fontSize.xl,
        fontWeight: designTokens.typography.fontWeight.semibold,
        marginBottom: designTokens.spacing[4]
      }}>
        Explore Model Insights
      </h2>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: designTokens.spacing[4],
        marginBottom: designTokens.spacing[6]
      }}>
        <div style={{
          backgroundColor: theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[200],
          padding: designTokens.spacing[4],
          borderRadius: designTokens.borderRadius.lg
        }}>
          <h3 style={{
            fontSize: designTokens.typography.fontSize.lg,
            fontWeight: designTokens.typography.fontWeight.semibold,
            marginTop: 0,
            marginBottom: designTokens.spacing[3]
          }}>
            Model Details
          </h3>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: `${designTokens.spacing[2]} ${designTokens.spacing[4]}`,
            fontSize: designTokens.typography.fontSize.sm
          }}>
            <div style={{ color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600] }}>
              Name
            </div>
            <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
              {modelName}
            </div>
            
            <div style={{ color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600] }}>
              Version
            </div>
            <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
              {modelVersion || 1}
            </div>
            
            <div style={{ color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600] }}>
              Type
            </div>
            <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
              Neural Additive Model
            </div>
            
            <div style={{ color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600] }}>
              Features
            </div>
            <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
              {featureImportance.length}
            </div>
          </div>
        </div>
        
        <div style={{
          backgroundColor: theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[200],
          padding: designTokens.spacing[4],
          borderRadius: designTokens.borderRadius.lg
        }}>
          <h3 style={{
            fontSize: designTokens.typography.fontSize.lg,
            fontWeight: designTokens.typography.fontWeight.semibold,
            marginTop: 0,
            marginBottom: designTokens.spacing[3]
          }}>
            Performance Metrics
          </h3>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: `${designTokens.spacing[2]} ${designTokens.spacing[4]}`,
            fontSize: designTokens.typography.fontSize.sm
          }}>
            <div style={{ color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600] }}>
              RMSE
            </div>
            <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
              0.0821
            </div>
            
            <div style={{ color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600] }}>
              MAE
            </div>
            <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
              0.0654
            </div>
            
            <div style={{ color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600] }}>
              R²
            </div>
            <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
              0.8932
            </div>
            
            <div style={{ color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600] }}>
              Training Time
            </div>
            <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
              1m 24s
            </div>
          </div>
        </div>
      </div>
      
      {/* Feature Importance Visualization */}
      <div style={{
        backgroundColor: theme === 'dark' ? designTokens.colors.neutral[800] : designTokens.colors.neutral[200],
        padding: designTokens.spacing[4],
        borderRadius: designTokens.borderRadius.lg,
        marginBottom: designTokens.spacing[4]
      }}>
        <h3 style={{
          fontSize: designTokens.typography.fontSize.lg,
          fontWeight: designTokens.typography.fontWeight.semibold,
          marginTop: 0,
          marginBottom: designTokens.spacing[4]
        }}>
          Feature Importance
        </h3>
        
        <div style={{ height: '300px' }}>
          {/* In a real implementation, this would be a chart component */}
          {/* This is a simplified visual representation */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            justifyContent: 'space-around'
          }}>
            {featureImportance.slice(0, 5).map((feature, index) => (
              <div 
                key={feature.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  margin: `${designTokens.spacing[1]} 0`
                }}
              >
                <div style={{
                  width: '120px',
                  fontSize: designTokens.typography.fontSize.sm,
                  marginRight: designTokens.spacing[2],
                  textAlign: 'right',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}>
                  {feature.name}
                </div>
                
                <div style={{
                  flex: 1,
                  height: '24px',
                  position: 'relative'
                }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${feature.contribution * 100}%` }}
                    transition={{
                      duration: 0.8,
                      delay: index * 0.1,
                      ease: designTokens.animation.easing.entrance
                    }}
                    style={{
                      height: '100%',
                      backgroundColor: designTokens.colors.viz.categorical[index % designTokens.colors.viz.categorical.length],
                      borderRadius: designTokens.borderRadius.md
                    }}
                  />
                </div>
                
                <div style={{
                  width: '60px',
                  textAlign: 'right',
                  marginLeft: designTokens.spacing[2],
                  fontSize: designTokens.typography.fontSize.sm,
                  fontWeight: designTokens.typography.fontWeight.medium
                }}>
                  {(feature.contribution * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div style={{ textAlign: 'center' }}>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          style={{
            backgroundColor: designTokens.colors.primary.base,
            color: designTokens.colors.primary.contrast,
            border: 'none',
            borderRadius: designTokens.borderRadius.md,
            padding: `${designTokens.spacing[2]} ${designTokens.spacing[4]}`,
            fontSize: designTokens.typography.fontSize.md,
            fontWeight: designTokens.typography.fontWeight.medium,
            cursor: 'pointer',
            boxShadow: designTokens.elevation.shadow.md
          }}
        >
          Download Detailed Report
        </motion.button>
      </div>
    </div>
  );
};

/**
 * Initialize NAM Application
 * 
 * This function initializes the Neural Additive Models interface
 * in a specified container element.
 */
export const initNAMApp = (containerId, options = {}) => {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`Container element with ID '${containerId}' not found.`);
    return;
  }
  
  ReactDOM.render(
    <NAMApp {...options} />,
    container
  );
};

// Default export
export default {
  initNAMApp,
  NAMApp,
  designTokens
};
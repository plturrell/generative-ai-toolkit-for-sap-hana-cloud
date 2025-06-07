/**
 * ModelExplorer Component
 * 
 * A sophisticated visualization that allows users to explore and understand
 * the Neural Additive Model's behavior across different feature values.
 * It combines interactive shape function visualizations with precise
 * data representation.
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { designTokens } from './design_tokens';

/**
 * Main ModelExplorer Component
 */
export const ModelExplorer = ({
  modelName,
  modelVersion,
  featureShapeFunctions = [],
  theme = 'light',
  apiEndpoint = '/api/nam'
}) => {
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [shapeFunctions, setShapeFunctions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  
  // Colors based on theme
  const colors = theme === 'dark' 
    ? {
        primary: designTokens.colors.primary.light,
        background: designTokens.colors.neutral[900],
        cardBackground: designTokens.colors.neutral[800],
        text: designTokens.colors.neutral[200],
        muted: designTokens.colors.neutral[500],
        accent: designTokens.colors.viz.categorical
      }
    : {
        primary: designTokens.colors.primary.base,
        background: designTokens.colors.neutral[100],
        cardBackground: designTokens.colors.neutral[200],
        text: designTokens.colors.neutral[800],
        muted: designTokens.colors.neutral[600],
        accent: designTokens.colors.viz.categorical
      };
  
  // Fetch shape functions from backend
  useEffect(() => {
    // If featureShapeFunctions are provided, use them
    if (featureShapeFunctions && featureShapeFunctions.length > 0) {
      setShapeFunctions(featureShapeFunctions);
      setSelectedFeature(featureShapeFunctions[0].name);
      return;
    }
    
    // Otherwise fetch from API
    const fetchShapeFunctions = async () => {
      if (!modelName || !modelVersion) return;
      
      setIsLoading(true);
      
      try {
        // This would be a real API call in production
        // Simulating API response for demonstration
        const simulatedResponse = {
          shape_functions: {
            PRICE: {
              x_values: Array.from({ length: 50 }, (_, i) => 10 + i * 2),
              y_values: Array.from({ length: 50 }, (_, i) => {
                const x = i / 49;
                return -0.5 + Math.sin(x * 4) * 0.5;
              }),
              importance: 0.35
            },
            QUANTITY: {
              x_values: Array.from({ length: 50 }, (_, i) => i * 10),
              y_values: Array.from({ length: 50 }, (_, i) => {
                const x = i / 49;
                return 0.3 * Math.log(x * 9 + 1);
              }),
              importance: 0.25
            },
            REGION: {
              x_values: Array.from({ length: 5 }, (_, i) => i),
              y_values: [0.1, -0.2, 0.15, 0.3, -0.1],
              importance: 0.15
            },
            CUSTOMER_SEGMENT: {
              x_values: Array.from({ length: 4 }, (_, i) => i),
              y_values: [-0.25, 0.1, 0.2, 0.4],
              importance: 0.25
            }
          }
        };
        
        // Transform to our component format
        const formattedShapeFunctions = Object.entries(simulatedResponse.shape_functions).map(([name, data]) => ({
          name,
          xValues: data.x_values,
          yValues: data.y_values,
          importance: data.importance,
          color: colors.accent[Math.floor(Math.random() * colors.accent.length)]
        }));
        
        setShapeFunctions(formattedShapeFunctions);
        if (formattedShapeFunctions.length > 0) {
          setSelectedFeature(formattedShapeFunctions[0].name);
        }
      } catch (error) {
        console.error('Error fetching shape functions:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchShapeFunctions();
  }, [modelName, modelVersion, featureShapeFunctions]);
  
  // Get current feature data
  const getCurrentFeatureData = () => {
    if (!selectedFeature) return null;
    return shapeFunctions.find(f => f.name === selectedFeature);
  };
  
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.1,
        duration: 0.3
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 24
      }
    }
  };
  
  // If still loading or no shape functions available
  if (isLoading) {
    return (
      <div style={{
        fontFamily: designTokens.typography.fontFamily.primary,
        color: colors.text,
        backgroundColor: colors.background,
        borderRadius: designTokens.borderRadius.xl,
        padding: designTokens.spacing[6],
        textAlign: 'center',
        minHeight: '300px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '50%',
          border: `3px solid ${colors.primary}`,
          borderTopColor: 'transparent',
          animation: 'nam-spin 1s linear infinite',
          marginBottom: designTokens.spacing[4]
        }} />
        <style>
          {`
            @keyframes nam-spin {
              to { transform: rotate(360deg); }
            }
          `}
        </style>
        <div style={{
          fontSize: designTokens.typography.fontSize.md,
          fontWeight: designTokens.typography.fontWeight.medium
        }}>
          Loading shape functions...
        </div>
      </div>
    );
  }
  
  if (shapeFunctions.length === 0) {
    return (
      <div style={{
        fontFamily: designTokens.typography.fontFamily.primary,
        color: colors.text,
        backgroundColor: colors.background,
        borderRadius: designTokens.borderRadius.xl,
        padding: designTokens.spacing[6],
        textAlign: 'center',
        minHeight: '300px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <div style={{
          fontSize: designTokens.typography.fontSize.lg,
          fontWeight: designTokens.typography.fontWeight.medium,
          marginBottom: designTokens.spacing[2]
        }}>
          No shape functions available
        </div>
        <div style={{
          fontSize: designTokens.typography.fontSize.base,
          color: colors.muted,
          maxWidth: '400px',
          marginBottom: designTokens.spacing[4]
        }}>
          Shape functions could not be loaded for this model. This may be because the model was trained without interpretability features enabled.
        </div>
      </div>
    );
  }
  
  const currentFeature = getCurrentFeatureData();
  
  return (
    <motion.div
      className="nam-model-explorer"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      style={{
        fontFamily: designTokens.typography.fontFamily.primary,
        color: colors.text,
        backgroundColor: colors.background,
        borderRadius: designTokens.borderRadius.xl,
        padding: designTokens.spacing[4],
        boxShadow: designTokens.elevation.shadow.md,
        width: '100%'
      }}
    >
      <motion.div
        variants={itemVariants}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: designTokens.spacing[4]
        }}
      >
        <div>
          <h3 style={{
            fontSize: designTokens.typography.fontSize.lg,
            fontWeight: designTokens.typography.fontWeight.semibold,
            margin: 0,
            marginBottom: designTokens.spacing[1]
          }}>
            Feature Shape Functions
          </h3>
          <div style={{
            fontSize: designTokens.typography.fontSize.sm,
            color: colors.muted,
            marginBottom: designTokens.spacing[3]
          }}>
            Explore how each feature affects predictions across its range of values
          </div>
        </div>
        
        <div style={{
          fontSize: designTokens.typography.fontSize.sm,
          color: colors.muted,
          display: 'flex',
          alignItems: 'center',
          gap: designTokens.spacing[1]
        }}>
          <span>Model:</span> 
          <span style={{ fontWeight: designTokens.typography.fontWeight.medium, color: colors.text }}>
            {modelName} (v{modelVersion})
          </span>
        </div>
      </motion.div>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: '250px 1fr',
        gap: designTokens.spacing[4]
      }}>
        {/* Feature Selector */}
        <motion.div
          variants={itemVariants}
          style={{
            backgroundColor: colors.cardBackground,
            borderRadius: designTokens.borderRadius.lg,
            padding: designTokens.spacing[3],
            height: 'fit-content'
          }}
        >
          <div style={{
            fontSize: designTokens.typography.fontSize.md,
            fontWeight: designTokens.typography.fontWeight.medium,
            marginBottom: designTokens.spacing[3]
          }}>
            Features
          </div>
          
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: designTokens.spacing[2]
          }}>
            {shapeFunctions.map((feature) => (
              <motion.button
                key={feature.name}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setSelectedFeature(feature.name)}
                style={{
                  padding: designTokens.spacing[2],
                  backgroundColor: selectedFeature === feature.name 
                    ? (theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)')
                    : 'transparent',
                  borderRadius: designTokens.borderRadius.md,
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  color: colors.text,
                  fontFamily: 'inherit',
                  fontSize: designTokens.typography.fontSize.sm,
                  textAlign: 'left',
                  borderLeft: `3px solid ${selectedFeature === feature.name ? feature.color : 'transparent'}`,
                  transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
                }}
              >
                <span style={{
                  fontWeight: selectedFeature === feature.name 
                    ? designTokens.typography.fontWeight.medium 
                    : designTokens.typography.fontWeight.regular
                }}>
                  {feature.name}
                </span>
                
                <span style={{
                  fontSize: designTokens.typography.fontSize.xs,
                  color: colors.muted,
                  backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                  padding: `${designTokens.spacing[1]} ${designTokens.spacing[2]}`,
                  borderRadius: designTokens.borderRadius.full,
                }}>
                  {(feature.importance * 100).toFixed(0)}%
                </span>
              </motion.button>
            ))}
          </div>
        </motion.div>
        
        {/* Shape Function Visualization */}
        <motion.div
          variants={itemVariants}
          style={{
            backgroundColor: colors.cardBackground,
            borderRadius: designTokens.borderRadius.lg,
            padding: designTokens.spacing[4]
          }}
        >
          <AnimatePresence mode="wait">
            {currentFeature && (
              <motion.div
                key={currentFeature.name}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  marginBottom: designTokens.spacing[4]
                }}>
                  <div>
                    <div style={{
                      fontSize: designTokens.typography.fontSize.md,
                      fontWeight: designTokens.typography.fontWeight.semibold,
                      color: currentFeature.color
                    }}>
                      {currentFeature.name}
                    </div>
                    <div style={{
                      fontSize: designTokens.typography.fontSize.sm,
                      color: colors.muted
                    }}>
                      Importance: {(currentFeature.importance * 100).toFixed(1)}%
                    </div>
                  </div>
                  
                  <div style={{
                    display: 'flex',
                    gap: designTokens.spacing[2]
                  }}>
                    <button style={{
                      backgroundColor: 'transparent',
                      border: `1px solid ${colors.muted}`,
                      borderRadius: designTokens.borderRadius.md,
                      padding: `${designTokens.spacing[1]} ${designTokens.spacing[2]}`,
                      color: colors.text,
                      fontSize: designTokens.typography.fontSize.xs,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: designTokens.spacing[1]
                    }}>
                      <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                        <path d="M6 2V10M2 6H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      Zoom
                    </button>
                    
                    <button style={{
                      backgroundColor: 'transparent',
                      border: `1px solid ${colors.muted}`,
                      borderRadius: designTokens.borderRadius.md,
                      padding: `${designTokens.spacing[1]} ${designTokens.spacing[2]}`,
                      color: colors.text,
                      fontSize: designTokens.typography.fontSize.xs,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: designTokens.spacing[1]
                    }}>
                      <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                        <path d="M3 9L9 3M9 3H4M9 3V8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      Export
                    </button>
                  </div>
                </div>
                
                {/* Shape Function Chart */}
                <div style={{
                  height: '250px',
                  position: 'relative',
                  marginBottom: designTokens.spacing[4]
                }}>
                  <svg width="100%" height="100%" viewBox="0 0 500 250" style={{ overflow: 'visible' }}>
                    {/* X and Y axes */}
                    <line 
                      x1="40" y1="200" x2="480" y2="200" 
                      stroke={colors.muted} 
                      strokeWidth="1"
                    />
                    <line 
                      x1="40" y1="30" x2="40" y2="200" 
                      stroke={colors.muted} 
                      strokeWidth="1"
                    />
                    
                    {/* Horizontal grid lines */}
                    {[0, 1, 2, 3, 4].map(i => {
                      const y = 200 - i * 40;
                      return (
                        <line 
                          key={`grid-h-${i}`}
                          x1="40" y1={y} x2="480" y2={y} 
                          stroke={colors.muted} 
                          strokeWidth="0.5"
                          strokeDasharray="4,4"
                          opacity="0.5"
                        />
                      );
                    })}
                    
                    {/* Data path */}
                    <motion.path
                      initial={{ pathLength: 0, opacity: 0 }}
                      animate={{ 
                        pathLength: 1, 
                        opacity: 1,
                        transition: {
                          pathLength: {
                            duration: 1,
                            ease: designTokens.animation.easing.entrance
                          },
                          opacity: {
                            duration: 0.5
                          }
                        }
                      }}
                      d={(() => {
                        // Scale values to chart dimensions
                        const xScale = (440 - 10) / (currentFeature.xValues.length - 1);
                        const yMin = Math.min(...currentFeature.yValues);
                        const yMax = Math.max(...currentFeature.yValues);
                        const yRange = Math.max(Math.abs(yMin), Math.abs(yMax)) * 2;
                        
                        return currentFeature.xValues.map((x, i) => {
                          const xPos = 40 + i * xScale;
                          const yPos = 200 - ((currentFeature.yValues[i] - yMin) / yRange) * 160;
                          return `${i === 0 ? 'M' : 'L'}${xPos},${yPos}`;
                        }).join(' ');
                      })()}
                      fill="none"
                      stroke={currentFeature.color}
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    
                    {/* Zero line */}
                    <line 
                      x1="40" 
                      y1="100" 
                      x2="480" 
                      y2="100" 
                      stroke={currentFeature.color} 
                      strokeWidth="1"
                      strokeDasharray="4,4"
                      opacity="0.5"
                    />
                    
                    {/* X-axis labels */}
                    {[0, 1, 2, 3, 4].map(i => {
                      const xPos = 40 + i * 110;
                      const xValueIndex = Math.floor(i * (currentFeature.xValues.length - 1) / 4);
                      const xValue = currentFeature.xValues[xValueIndex];
                      
                      return (
                        <text 
                          key={`x-label-${i}`}
                          x={xPos} 
                          y="220" 
                          textAnchor="middle" 
                          fontSize="12" 
                          fill={colors.muted}
                        >
                          {typeof xValue === 'number' ? xValue.toFixed(1) : xValue}
                        </text>
                      );
                    })}
                    
                    {/* Y-axis labels */}
                    {[0, 1, 2, 3, 4].map(i => {
                      const y = 200 - i * 40;
                      const yMin = Math.min(...currentFeature.yValues);
                      const yMax = Math.max(...currentFeature.yValues);
                      const yRange = Math.max(Math.abs(yMin), Math.abs(yMax)) * 2;
                      const yValue = yMin + (i * yRange / 4);
                      
                      return (
                        <text 
                          key={`y-label-${i}`}
                          x="30" 
                          y={y + 4} 
                          textAnchor="end" 
                          fontSize="12" 
                          fill={colors.muted}
                        >
                          {yValue.toFixed(2)}
                        </text>
                      );
                    })}
                  </svg>
                </div>
                
                {/* Shape Function Interpretation */}
                <div style={{
                  backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
                  borderRadius: designTokens.borderRadius.md,
                  padding: designTokens.spacing[3],
                  fontSize: designTokens.typography.fontSize.sm,
                  lineHeight: designTokens.typography.lineHeight.relaxed
                }}>
                  <div style={{ 
                    fontWeight: designTokens.typography.fontWeight.medium,
                    marginBottom: designTokens.spacing[1]
                  }}>
                    Interpretation
                  </div>
                  
                  {/* In a real application, this would be generated based on the actual shape function */}
                  <p style={{ margin: 0 }}>
                    {currentFeature.name === 'PRICE' && 
                      "As the price increases, it initially has a positive effect on the prediction, but this effect begins to diminish after reaching a certain threshold, suggesting diminishing returns."}
                    {currentFeature.name === 'QUANTITY' && 
                      "Quantity shows a logarithmic relationship with the prediction, where the impact is strongest at lower values and gradually levels off as quantity increases."}
                    {currentFeature.name === 'REGION' && 
                      "Regions have varying effects on the prediction, with Region 3 showing the strongest positive impact and Region 1 showing a negative effect."}
                    {currentFeature.name === 'CUSTOMER_SEGMENT' && 
                      "Customer segments show a clear pattern where higher segments have an increasingly positive effect on the prediction, with Segment 3 having the strongest impact."}
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default ModelExplorer;
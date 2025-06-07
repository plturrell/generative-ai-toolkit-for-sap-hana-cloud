/**
 * FeatureContributions Component
 * 
 * A sophisticated visualization component that reveals how each feature
 * contributes to predictions. The component incorporates fluid animations,
 * haptic feedback, and precise visual design to make complex model behavior
 * immediately understandable.
 */

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { designTokens } from './design_tokens';

/**
 * Main FeatureContributions Component
 */
export const FeatureContributions = ({
  data,
  basePrediction = 0,
  showTotal = true,
  maxFeatures = 8,
  animate = true,
  onFeatureClick = () => {},
  theme = 'light',
  id = 'feature-contributions'
}) => {
  const [activeFeature, setActiveFeature] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const containerRef = useRef(null);
  
  // Detect if any feature needs detailed inspection
  const hasLargeImpact = data.some(feature => Math.abs(feature.contribution) > 0.5 * Math.max(...data.map(f => Math.abs(f.contribution))));
  
  // Colors based on theme
  const colors = theme === 'dark' 
    ? {
        positive: designTokens.colors.contribution.positive.strong,
        negative: designTokens.colors.contribution.negative.strong,
        text: designTokens.colors.neutral[200],
        background: designTokens.colors.neutral[900],
        divider: designTokens.colors.neutral[700],
        highlight: designTokens.colors.primary.light
      }
    : {
        positive: designTokens.colors.contribution.positive.strong,
        negative: designTokens.colors.contribution.negative.strong,
        text: designTokens.colors.neutral[800],
        background: designTokens.colors.neutral[100],
        divider: designTokens.colors.neutral[400],
        highlight: designTokens.colors.primary.light
      };
  
  // Sort features by absolute contribution
  const sortedData = [...data].sort((a, b) => 
    Math.abs(b.contribution) - Math.abs(a.contribution)
  );
  
  // Limit displayed features if needed
  const displayData = isExpanded 
    ? sortedData 
    : sortedData.slice(0, maxFeatures);
    
  // Calculate total contribution (sum of all features + base)
  const totalContribution = sortedData.reduce(
    (sum, feature) => sum + feature.contribution, 
    basePrediction
  );
  
  // Calculate the maximum contribution for scaling
  const maxContribution = Math.max(
    ...sortedData.map(feature => Math.abs(feature.contribution)),
    Math.abs(basePrediction)
  );
  
  // Handle toggling feature details
  const handleFeatureClick = (feature) => {
    // Provide haptic feedback
    if (window.navigator && window.navigator.vibrate) {
      window.navigator.vibrate(2); // Very subtle feedback
    }
    
    setActiveFeature(activeFeature === feature.name ? null : feature.name);
    onFeatureClick(feature);
  };
  
  // Define animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.05,
        duration: 0.3,
        ease: designTokens.animation.easing.entrance
      }
    }
  };
  
  const featureVariants = {
    hidden: { opacity: 0, y: 20, scale: 0.95 },
    visible: { 
      opacity: 1, 
      y: 0, 
      scale: 1,
      transition: {
        type: 'spring',
        stiffness: 500,
        damping: 25,
        mass: 1
      }
    },
    hover: {
      scale: 1.02,
      transition: {
        duration: 0.2,
        ease: designTokens.animation.easing.standard
      }
    },
    tap: {
      scale: 0.98,
      transition: {
        duration: 0.1
      }
    }
  };
  
  const barVariants = {
    hidden: { scaleX: 0, originX: 0 },
    visible: (custom) => ({ 
      scaleX: 1,
      transition: {
        type: 'spring',
        stiffness: 500,
        damping: 30,
        mass: 1,
        delay: custom * 0.03
      }
    })
  };
  
  const valueVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.3,
        delay: 0.2
      }
    }
  };
  
  return (
    <motion.div
      id={id}
      ref={containerRef}
      className="nam-feature-contributions"
      initial={animate ? "hidden" : "visible"}
      animate="visible"
      variants={containerVariants}
      aria-label="Feature contributions visualization"
      style={{
        fontFamily: designTokens.typography.fontFamily.primary,
        backgroundColor: colors.background,
        borderRadius: designTokens.borderRadius.xl,
        padding: designTokens.spacing[4],
        boxShadow: designTokens.elevation.shadow.lg,
        maxWidth: '680px',
        width: '100%',
        color: colors.text
      }}
    >
      {/* Header */}
      <div className="nam-feature-contributions-header" style={{
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
          Feature Contributions
        </h3>
        
        {showTotal && (
          <motion.div 
            className="nam-prediction-total"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{
              type: 'spring',
              stiffness: 500,
              damping: 30,
              delay: 0.5
            }}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-end'
            }}
          >
            <div style={{ 
              fontSize: designTokens.typography.fontSize.sm,
              color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
              marginBottom: designTokens.spacing[1]
            }}>
              Total Prediction
            </div>
            <div style={{ 
              fontSize: designTokens.typography.fontSize.xl,
              fontWeight: designTokens.typography.fontWeight.bold
            }}>
              {totalContribution.toFixed(2)}
            </div>
          </motion.div>
        )}
      </div>
      
      {/* Base prediction/intercept */}
      {basePrediction !== 0 && (
        <motion.div
          className="nam-feature-base"
          variants={featureVariants}
          style={{
            marginBottom: designTokens.spacing[3],
            padding: designTokens.spacing[2],
            borderRadius: designTokens.borderRadius.md,
            backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: designTokens.spacing[1] }}>
            <div style={{ fontWeight: designTokens.typography.fontWeight.medium }}>Base value</div>
            <motion.div variants={valueVariants}>
              {basePrediction.toFixed(2)}
            </motion.div>
          </div>
          
          <div style={{ 
            position: 'relative',
            height: '8px',
            backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
            borderRadius: designTokens.borderRadius.full,
            overflow: 'hidden'
          }}>
            <motion.div
              variants={barVariants}
              style={{
                height: '100%',
                width: `${(Math.abs(basePrediction) / maxContribution) * 100}%`,
                backgroundColor: basePrediction >= 0 ? colors.positive : colors.negative,
                borderRadius: designTokens.borderRadius.full,
                position: 'absolute',
                left: basePrediction >= 0 ? '50%' : `calc(50% - ${(Math.abs(basePrediction) / maxContribution) * 100}%)`,
                opacity: 0.7
              }}
            />
            <div style={{
              position: 'absolute',
              height: '100%',
              width: '1px',
              backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.2)',
              left: '50%',
              top: 0
            }} />
          </div>
        </motion.div>
      )}
      
      {/* Feature contributions */}
      <div className="nam-features-list" style={{ 
        display: 'flex',
        flexDirection: 'column',
        gap: designTokens.spacing[2]
      }}>
        {displayData.map((feature, index) => (
          <FeatureItem 
            key={feature.name}
            feature={feature}
            maxContribution={maxContribution}
            isActive={activeFeature === feature.name}
            onClick={() => handleFeatureClick(feature)}
            variants={featureVariants}
            barVariants={barVariants}
            valueVariants={valueVariants}
            custom={index}
            theme={theme}
            colors={colors}
          />
        ))}
      </div>
      
      {/* Show more/less toggle */}
      {sortedData.length > maxFeatures && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          onClick={() => {
            setIsExpanded(!isExpanded);
            // Subtle haptic feedback
            if (window.navigator && window.navigator.vibrate) {
              window.navigator.vibrate([2, 10, 2]);
            }
          }}
          className="nam-show-more-button"
          style={{
            background: 'none',
            border: 'none',
            color: designTokens.colors.primary.base,
            fontFamily: 'inherit',
            fontSize: designTokens.typography.fontSize.sm,
            cursor: 'pointer',
            padding: designTokens.spacing[2],
            marginTop: designTokens.spacing[2],
            borderRadius: designTokens.borderRadius.md,
            transition: `background-color ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`,
            alignSelf: 'center',
            fontWeight: designTokens.typography.fontWeight.medium,
            display: 'flex',
            alignItems: 'center',
            gap: designTokens.spacing[1]
          }}
          whileHover={{
            backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)'
          }}
          whileTap={{ scale: 0.98 }}
        >
          {isExpanded ? 'Show Fewer Features' : 'Show All Features'}
          <svg 
            width="12"
            height="12"
            viewBox="0 0 12 12"
            fill="none"
            style={{
              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: `transform ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
            }}
          >
            <path
              d="M2 4L6 8L10 4"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </motion.button>
      )}
      
      {/* Warning about large impact features if needed */}
      {hasLargeImpact && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ delay: 0.8, duration: 0.3 }}
          className="nam-impact-warning"
          style={{
            marginTop: designTokens.spacing[3],
            padding: designTokens.spacing[3],
            borderRadius: designTokens.borderRadius.md,
            backgroundColor: theme === 'dark' 
              ? 'rgba(255, 183, 77, 0.1)' 
              : 'rgba(255, 183, 77, 0.1)',
            borderLeft: `3px solid ${designTokens.colors.feedback.warning}`,
            fontSize: designTokens.typography.fontSize.sm,
            lineHeight: designTokens.typography.lineHeight.relaxed
          }}
        >
          <div style={{ 
            display: 'flex', 
            alignItems: 'center',
            gap: designTokens.spacing[2],
            marginBottom: designTokens.spacing[1],
            fontWeight: designTokens.typography.fontWeight.medium
          }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 14A6 6 0 108 2a6 6 0 000 12z" stroke={designTokens.colors.feedback.warning} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M8 5.5v3" stroke={designTokens.colors.feedback.warning} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M8 11.5v.01" stroke={designTokens.colors.feedback.warning} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Feature Impact Insight
          </div>
          Some features have significantly higher impact on this prediction than others. Click on them to explore details and understand their influence on the model.
        </motion.div>
      )}
    </motion.div>
  );
};

/**
 * Individual feature contribution item
 */
const FeatureItem = ({ 
  feature, 
  maxContribution, 
  isActive,
  onClick,
  variants, 
  barVariants,
  valueVariants,
  custom,
  theme,
  colors
}) => {
  // Normalize contribution for visualization
  const normalizedContribution = feature.contribution / maxContribution;
  const isPositive = feature.contribution >= 0;
  
  return (
    <motion.div
      className={`nam-feature-item ${isActive ? 'active' : ''}`}
      variants={variants}
      custom={custom}
      whileHover="hover"
      whileTap="tap"
      onClick={onClick}
      style={{
        borderRadius: designTokens.borderRadius.lg,
        padding: designTokens.spacing[2],
        backgroundColor: isActive 
          ? (theme === 'dark' ? 'rgba(255,255,255,0.07)' : colors.highlight)
          : 'transparent',
        cursor: 'pointer',
        transition: `background-color ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`,
        position: 'relative',
        overflow: 'hidden'
      }}
      aria-expanded={isActive}
    >
      {/* Feature name and value */}
      <div style={{ 
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: designTokens.spacing[1],
        position: 'relative',
        zIndex: 1
      }}>
        <div style={{
          fontWeight: isActive 
            ? designTokens.typography.fontWeight.semibold 
            : designTokens.typography.fontWeight.medium,
          transition: `font-weight 0.2s ease`
        }}>
          {feature.name}
          
          {/* Original value indicator */}
          {feature.value !== undefined && (
            <span style={{
              marginLeft: designTokens.spacing[2],
              fontSize: designTokens.typography.fontSize.xs,
              opacity: 0.7,
              fontWeight: designTokens.typography.fontWeight.regular
            }}>
              ({typeof feature.value === 'number' ? feature.value.toFixed(2) : feature.value})
            </span>
          )}
        </div>
        
        <motion.div 
          variants={valueVariants}
          style={{
            fontWeight: designTokens.typography.fontWeight.medium,
            color: isPositive ? colors.positive : colors.negative
          }}
        >
          {isPositive ? '+' : ''}{feature.contribution.toFixed(2)}
        </motion.div>
      </div>
      
      {/* Contribution bar visualization */}
      <div style={{ 
        position: 'relative',
        height: '8px',
        backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
        borderRadius: designTokens.borderRadius.full,
        overflow: 'hidden'
      }}>
        <motion.div
          variants={barVariants}
          custom={custom}
          style={{
            height: '100%',
            width: `${Math.abs(normalizedContribution) * 100}%`,
            backgroundColor: isPositive ? colors.positive : colors.negative,
            borderRadius: designTokens.borderRadius.full,
            position: 'absolute',
            left: isPositive ? '50%' : `calc(50% - ${Math.abs(normalizedContribution) * 100}%)`,
            opacity: isActive ? 1 : 0.7,
            transition: `opacity ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
          }}
        />
        <div style={{
          position: 'absolute',
          height: '100%',
          width: '1px',
          backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.2)',
          left: '50%',
          top: 0
        }} />
      </div>
      
      {/* Expanded detail section */}
      <AnimatePresence>
        {isActive && (
          <motion.div
            className="nam-feature-detail"
            initial={{ height: 0, opacity: 0 }}
            animate={{ 
              height: 'auto', 
              opacity: 1,
              transition: {
                height: {
                  duration: 0.3,
                  ease: designTokens.animation.easing.entrance
                },
                opacity: {
                  duration: 0.2,
                  delay: 0.1
                }
              }
            }}
            exit={{ 
              height: 0, 
              opacity: 0,
              transition: {
                height: {
                  duration: 0.2,
                  ease: designTokens.animation.easing.exit
                },
                opacity: {
                  duration: 0.1
                }
              }
            }}
            style={{
              overflow: 'hidden',
              marginTop: designTokens.spacing[3]
            }}
          >
            <div style={{ 
              fontSize: designTokens.typography.fontSize.sm,
              lineHeight: designTokens.typography.lineHeight.relaxed,
              color: theme === 'dark' ? designTokens.colors.neutral[300] : designTokens.colors.neutral[700],
              padding: `${designTokens.spacing[1]} 0`
            }}>
              <p style={{ margin: `0 0 ${designTokens.spacing[2]} 0` }}>
                This feature {isPositive ? 'increases' : 'decreases'} the prediction by{' '}
                <strong style={{ fontWeight: designTokens.typography.fontWeight.semibold }}>
                  {Math.abs(feature.contribution).toFixed(2)}
                </strong>{' '}
                units ({Math.round(Math.abs(normalizedContribution) * 100)}% impact).
              </p>
              
              {feature.explanation && (
                <p style={{ margin: `0 0 ${designTokens.spacing[2]} 0` }}>
                  {feature.explanation}
                </p>
              )}
              
              {/* Additional details or feature-specific insights would go here */}
            </div>
            
            {/* Visual enhancement for expanded state */}
            <motion.div
              className="nam-feature-detail-actions"
              initial={{ opacity: 0, y: 10 }}
              animate={{ 
                opacity: 1, 
                y: 0,
                transition: { delay: 0.2, duration: 0.2 }
              }}
              style={{
                display: 'flex',
                justifyContent: 'flex-end',
                gap: designTokens.spacing[2],
                borderTop: `1px solid ${theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)'}`,
                paddingTop: designTokens.spacing[2],
                marginTop: designTokens.spacing[2]
              }}
            >
              <button style={{
                background: 'none',
                border: 'none',
                color: designTokens.colors.primary.base,
                fontFamily: 'inherit',
                fontSize: designTokens.typography.fontSize.sm,
                cursor: 'pointer',
                padding: `${designTokens.spacing[1]} ${designTokens.spacing[2]}`,
                borderRadius: designTokens.borderRadius.md,
                fontWeight: designTokens.typography.fontWeight.medium,
                display: 'flex',
                alignItems: 'center',
                gap: designTokens.spacing[1]
              }}>
                Explore Feature
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M3.5 8.5L8.5 3.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M3.5 3.5H8.5V8.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default FeatureContributions;
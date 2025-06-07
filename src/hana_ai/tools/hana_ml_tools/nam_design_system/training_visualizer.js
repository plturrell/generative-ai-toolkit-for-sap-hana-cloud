/**
 * TrainingVisualizer Component
 * 
 * An elegant visualization of the NAM training process that makes the
 * complex model training process engaging and insightful. The component
 * combines fluid animations, progressive reveal of information, and
 * meaningful data visualization.
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { designTokens } from './design_tokens';

/**
 * Main TrainingVisualizer Component
 */
export const TrainingVisualizer = ({
  trainingProgress = 0,
  currentEpoch = 0,
  totalEpochs = 100,
  metrics = {},
  modelConfig = {},
  theme = 'light'
}) => {
  const [visibleNetworks, setVisibleNetworks] = useState(3);
  const [highlightedFeature, setHighlightedFeature] = useState(null);
  const networkContainerRef = useRef(null);
  
  // Colors based on theme
  const colors = theme === 'dark' 
    ? {
        primary: designTokens.colors.primary.light,
        background: designTokens.colors.neutral[900],
        text: designTokens.colors.neutral[200],
        accent: designTokens.colors.viz.categorical
      }
    : {
        primary: designTokens.colors.primary.base,
        background: designTokens.colors.neutral[100],
        text: designTokens.colors.neutral[800],
        accent: designTokens.colors.viz.categorical
      };
  
  // As training progresses, reveal more networks
  useEffect(() => {
    // Gradually increase visible networks as training progresses
    const maxNetworks = 10; // Example max
    const newVisible = Math.min(
      Math.ceil((trainingProgress / 100) * maxNetworks),
      maxNetworks
    );
    
    if (newVisible > visibleNetworks) {
      setVisibleNetworks(newVisible);
    }
  }, [trainingProgress]);
  
  // Simulated feature networks for visualization
  const featureNetworks = [
    { name: 'PRICE', color: colors.accent[0], complexity: 0.8 },
    { name: 'QUANTITY', color: colors.accent[1], complexity: 0.6 },
    { name: 'REGION', color: colors.accent[2], complexity: 0.5 },
    { name: 'CUSTOMER_SEGMENT', color: colors.accent[3], complexity: 0.9 },
    { name: 'SEASON', color: colors.accent[4], complexity: 0.7 },
    { name: 'PROMOTION', color: colors.accent[5], complexity: 0.4 },
    { name: 'CHANNEL', color: colors.accent[6], complexity: 0.3 },
    { name: 'WEEKDAY', color: colors.accent[7], complexity: 0.2 },
    { name: 'PRODUCT_CATEGORY', color: colors.accent[0], complexity: 0.8 },
    { name: 'INVENTORY_LEVEL', color: colors.accent[1], complexity: 0.5 }
  ].slice(0, visibleNetworks);
  
  // Training progress label variants
  const progressLabelVariants = {
    initial: { opacity: 0, y: -20 },
    animate: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        ease: designTokens.animation.easing.entrance
      }
    }
  };
  
  // Network item variants
  const networkItemVariants = {
    initial: { opacity: 0, scale: 0.8 },
    animate: i => ({
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.5,
        delay: i * 0.1,
        ease: designTokens.animation.easing.entrance
      }
    }),
    hover: {
      scale: 1.05,
      transition: {
        duration: 0.2,
        ease: designTokens.animation.easing.standard
      }
    }
  };
  
  return (
    <div
      className="nam-training-visualizer"
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
      {/* Training Progress Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: designTokens.spacing[4]
      }}>
        <motion.div
          variants={progressLabelVariants}
          initial="initial"
          animate="animate"
        >
          <div style={{
            fontSize: designTokens.typography.fontSize.sm,
            color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
            marginBottom: designTokens.spacing[1]
          }}>
            Training Progress
          </div>
          <div style={{
            fontSize: designTokens.typography.fontSize.xl,
            fontWeight: designTokens.typography.fontWeight.semibold
          }}>
            {trainingProgress}%
          </div>
        </motion.div>
        
        <motion.div
          variants={progressLabelVariants}
          initial="initial"
          animate="animate"
        >
          <div style={{
            fontSize: designTokens.typography.fontSize.sm,
            color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
            marginBottom: designTokens.spacing[1],
            textAlign: 'right'
          }}>
            Epoch
          </div>
          <div style={{
            fontSize: designTokens.typography.fontSize.xl,
            fontWeight: designTokens.typography.fontWeight.semibold
          }}>
            {currentEpoch} / {totalEpochs}
          </div>
        </motion.div>
      </div>
      
      {/* Progress Bar */}
      <div style={{
        marginBottom: designTokens.spacing[5],
        position: 'relative',
        height: '8px',
        backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
        borderRadius: designTokens.borderRadius.full,
        overflow: 'hidden'
      }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${trainingProgress}%` }}
          transition={{
            duration: 0.5,
            ease: designTokens.animation.easing.standard
          }}
          style={{
            height: '100%',
            backgroundColor: colors.primary,
            borderRadius: designTokens.borderRadius.full
          }}
        />
      </div>
      
      {/* Feature Networks */}
      <div 
        ref={networkContainerRef}
        style={{
          marginBottom: designTokens.spacing[4]
        }}
      >
        <div style={{
          fontSize: designTokens.typography.fontSize.md,
          fontWeight: designTokens.typography.fontWeight.medium,
          marginBottom: designTokens.spacing[3]
        }}>
          Feature Networks
        </div>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
          gap: designTokens.spacing[3]
        }}>
          {featureNetworks.map((feature, index) => (
            <motion.div
              key={feature.name}
              variants={networkItemVariants}
              initial="initial"
              animate="animate"
              whileHover="hover"
              custom={index}
              onClick={() => setHighlightedFeature(feature.name === highlightedFeature ? null : feature.name)}
              style={{
                backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
                borderRadius: designTokens.borderRadius.lg,
                padding: designTokens.spacing[3],
                cursor: 'pointer',
                border: highlightedFeature === feature.name 
                  ? `2px solid ${feature.color}`
                  : '2px solid transparent',
                transition: `border-color ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
              }}
            >
              <div style={{
                fontSize: designTokens.typography.fontSize.sm,
                fontWeight: designTokens.typography.fontWeight.medium,
                marginBottom: designTokens.spacing[2],
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}>
                {feature.name}
              </div>
              
              {/* Simple network visualization */}
              <div style={{
                height: '80px',
                position: 'relative'
              }}>
                <svg width="100%" height="100%" viewBox="0 0 100 80">
                  {/* Input node */}
                  <circle cx="10" cy="40" r="5" fill={feature.color} />
                  
                  {/* Hidden layer nodes */}
                  {[1, 2, 3].map((_, i) => {
                    const y = 20 + i * 20;
                    return (
                      <g key={`hidden-${i}`}>
                        <circle 
                          cx="50" 
                          cy={y} 
                          r="4" 
                          fill={feature.color} 
                          opacity={trainingProgress / 100} 
                        />
                        <line
                          x1="10"
                          y1="40"
                          x2="50"
                          y2={y}
                          stroke={feature.color}
                          strokeWidth="1"
                          opacity={Math.min((trainingProgress - 10) / 100, 1) * 0.7}
                        />
                        <line
                          x1="50"
                          y1={y}
                          x2="90"
                          y2="40"
                          stroke={feature.color}
                          strokeWidth="1"
                          opacity={Math.min((trainingProgress - 30) / 100, 1) * 0.7}
                        />
                      </g>
                    );
                  })}
                  
                  {/* Output node */}
                  <circle 
                    cx="90" 
                    cy="40" 
                    r="5" 
                    fill={feature.color} 
                    opacity={Math.min((trainingProgress - 20) / 100, 1)}
                  />
                </svg>
              </div>
              
              <div style={{
                fontSize: designTokens.typography.fontSize.xs,
                color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: designTokens.spacing[1]
              }}>
                <span>Complexity</span>
                <span style={{ fontWeight: designTokens.typography.fontWeight.medium }}>
                  {(feature.complexity * 100).toFixed(0)}%
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
      
      {/* Metrics */}
      <div style={{
        backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
        borderRadius: designTokens.borderRadius.lg,
        padding: designTokens.spacing[3]
      }}>
        <div style={{
          fontSize: designTokens.typography.fontSize.md,
          fontWeight: designTokens.typography.fontWeight.medium,
          marginBottom: designTokens.spacing[2]
        }}>
          Training Metrics
        </div>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
          gap: designTokens.spacing[3]
        }}>
          <MetricCard 
            label="Loss" 
            value={trainingProgress < 20 ? '---' : (1 - trainingProgress/150).toFixed(4)} 
            theme={theme}
          />
          
          <MetricCard 
            label="Val. Loss" 
            value={trainingProgress < 40 ? '---' : (1.1 - trainingProgress/140).toFixed(4)} 
            theme={theme}
          />
          
          <MetricCard 
            label="Learning Rate" 
            value={trainingProgress < 10 ? '---' : "0.001"} 
            theme={theme}
          />
          
          <MetricCard 
            label="Elapsed Time" 
            value={`${Math.floor(currentEpoch / 10)}m ${(currentEpoch % 10) * 6}s`} 
            theme={theme}
          />
        </div>
      </div>
    </div>
  );
};

/**
 * Metric Card Component
 */
const MetricCard = ({ label, value, theme }) => {
  return (
    <div style={{
      padding: designTokens.spacing[2],
      backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.5)',
      borderRadius: designTokens.borderRadius.md
    }}>
      <div style={{
        fontSize: designTokens.typography.fontSize.xs,
        color: theme === 'dark' ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
        marginBottom: designTokens.spacing[1]
      }}>
        {label}
      </div>
      <div style={{
        fontSize: designTokens.typography.fontSize.md,
        fontWeight: designTokens.typography.fontWeight.medium,
        fontFamily: designTokens.typography.fontFamily.mono
      }}>
        {value}
      </div>
    </div>
  );
};

export default TrainingVisualizer;
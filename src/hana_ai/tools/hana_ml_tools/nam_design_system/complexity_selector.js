/**
 * ComplexitySelector Component
 * 
 * A precision-crafted control for selecting model complexity with
 * visual representation of the neural network structure. The component
 * uses subtle animations, haptic feedback, and visual cues to create
 * an intuitive understanding of the selection's impact.
 */

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { designTokens } from './design_tokens';

// SVG paths for the neural network visualizations
// These are meticulously designed to represent the actual model structures
const networkPaths = {
  simple: {
    nodes: [
      { cx: 20, cy: 30, r: 4 },  // Input nodes
      { cx: 20, cy: 50, r: 4 },
      { cx: 20, cy: 70, r: 4 },
      
      { cx: 80, cy: 30, r: 5 },  // Hidden layer nodes (smaller for simple model)
      { cx: 80, cy: 50, r: 5 },
      { cx: 80, cy: 70, r: 5 },
      
      { cx: 140, cy: 50, r: 4 }  // Output node
    ],
    connections: [
      // Input to hidden connections
      'M20,30 L80,30', 'M20,30 L80,50', 
      'M20,50 L80,30', 'M20,50 L80,50', 'M20,50 L80,70',
      'M20,70 L80,50', 'M20,70 L80,70',
      
      // Hidden to output connections
      'M80,30 L140,50', 'M80,50 L140,50', 'M80,70 L140,50'
    ]
  },
  balanced: {
    nodes: [
      { cx: 20, cy: 20, r: 4 },  // Input nodes
      { cx: 20, cy: 40, r: 4 },
      { cx: 20, cy: 60, r: 4 },
      { cx: 20, cy: 80, r: 4 },
      
      { cx: 65, cy: 25, r: 5 },  // First hidden layer
      { cx: 65, cy: 45, r: 5 },
      { cx: 65, cy: 65, r: 5 },
      { cx: 65, cy: 85, r: 5 },
      
      { cx: 110, cy: 35, r: 5 }, // Second hidden layer
      { cx: 110, cy: 55, r: 5 },
      { cx: 110, cy: 75, r: 5 },
      
      { cx: 155, cy: 55, r: 4 }  // Output node
    ],
    connections: [
      // Too many connections to list individually
      // In a real implementation, these would be generated dynamically
      // based on the network architecture
      'M20,20 L65,25', 'M20,20 L65,45',
      'M20,40 L65,25', 'M20,40 L65,45', 'M20,40 L65,65',
      'M20,60 L65,45', 'M20,60 L65,65', 'M20,60 L65,85',
      'M20,80 L65,65', 'M20,80 L65,85',
      
      // First to second hidden layer
      'M65,25 L110,35', 'M65,25 L110,55',
      'M65,45 L110,35', 'M65,45 L110,55',
      'M65,65 L110,55', 'M65,65 L110,75',
      'M65,85 L110,75',
      
      // Second hidden to output
      'M110,35 L155,55', 'M110,55 L155,55', 'M110,75 L155,55'
    ]
  },
  complex: {
    nodes: [
      { cx: 15, cy: 15, r: 3.5 }, // Input nodes (smaller to fit more)
      { cx: 15, cy: 30, r: 3.5 },
      { cx: 15, cy: 45, r: 3.5 },
      { cx: 15, cy: 60, r: 3.5 },
      { cx: 15, cy: 75, r: 3.5 },
      { cx: 15, cy: 90, r: 3.5 },
      
      { cx: 55, cy: 20, r: 4 },  // First hidden layer
      { cx: 55, cy: 35, r: 4 },
      { cx: 55, cy: 50, r: 4 },
      { cx: 55, cy: 65, r: 4 },
      { cx: 55, cy: 80, r: 4 },
      
      { cx: 95, cy: 25, r: 4 },  // Second hidden layer
      { cx: 95, cy: 40, r: 4 },
      { cx: 95, cy: 55, r: 4 },
      { cx: 95, cy: 70, r: 4 },
      { cx: 95, cy: 85, r: 4 },
      
      { cx: 135, cy: 35, r: 4 }, // Third hidden layer
      { cx: 135, cy: 50, r: 4 },
      { cx: 135, cy: 65, r: 4 },
      { cx: 135, cy: 80, r: 4 },
      
      { cx: 175, cy: 55, r: 3.5 } // Output node
    ],
    connections: [
      // In a real implementation, we would generate these programmatically
      // and use a more sophisticated rendering approach for large networks
      // Simplified for demonstration purposes
      'M15,15 L55,20', 'M15,15 L55,35',
      'M15,30 L55,20', 'M15,30 L55,35', 'M15,30 L55,50',
      'M15,45 L55,35', 'M15,45 L55,50', 'M15,45 L55,65',
      'M15,60 L55,50', 'M15,60 L55,65', 'M15,60 L55,80',
      'M15,75 L55,65', 'M15,75 L55,80',
      'M15,90 L55,80',
      
      // Additional connections would be implemented
      // but simplified for this demonstration
      'M55,20 L95,25', 'M55,35 L95,25', 'M55,35 L95,40',
      'M55,50 L95,40', 'M55,50 L95,55', 'M55,65 L95,55',
      'M55,65 L95,70', 'M55,80 L95,70', 'M55,80 L95,85',
      
      'M95,25 L135,35', 'M95,40 L135,35', 'M95,40 L135,50',
      'M95,55 L135,50', 'M95,55 L135,65', 'M95,70 L135,65',
      'M95,70 L135,80', 'M95,85 L135,80',
      
      'M135,35 L175,55', 'M135,50 L175,55', 'M135,65 L175,55', 'M135,80 L175,55'
    ]
  }
};

/**
 * Renders a visually engaging control for selecting model complexity
 */
export const ComplexitySelector = ({ 
  value = 'balanced', 
  onChange,
  disabled = false,
  id = 'complexity-selector'
}) => {
  const [selectedValue, setSelectedValue] = useState(value);
  const [isAnimating, setIsAnimating] = useState(false);
  const [hoverState, setHoverState] = useState(null);
  const [focusVisible, setFocusVisible] = useState(false);
  const containerRef = useRef(null);

  // Detect keyboard navigation for accessibility
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Tab') {
        setFocusVisible(true);
      }
    };
    
    const handleMouseDown = () => {
      setFocusVisible(false);
    };
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('mousedown', handleMouseDown);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('mousedown', handleMouseDown);
    };
  }, []);

  // Handle selection change with animation
  const handleChange = (newValue) => {
    if (disabled || newValue === selectedValue) return;
    
    setIsAnimating(true);
    
    // Subtle haptic feedback if available
    if (window.navigator && window.navigator.vibrate) {
      window.navigator.vibrate(3); // Subtle 3ms vibration
    }
    
    // Stagger animation to show the change
    setTimeout(() => {
      setSelectedValue(newValue);
      if (onChange) onChange(newValue);
      
      // Allow animation to complete before accepting new changes
      setTimeout(() => {
        setIsAnimating(false);
      }, parseInt(designTokens.animation.duration.normal));
    }, parseInt(designTokens.animation.duration.fast));
  };

  // Get the appropriate SVG paths for the current selection
  const network = networkPaths[selectedValue];

  return (
    <div 
      className="nam-complexity-selector"
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        maxWidth: '480px',
        fontFamily: designTokens.typography.fontFamily.primary,
        userSelect: 'none',
        opacity: disabled ? 0.6 : 1,
        transition: `opacity ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
      }}
      aria-disabled={disabled}
    >
      <div 
        className="nam-complexity-selector-label"
        style={{
          fontSize: designTokens.typography.fontSize.md,
          fontWeight: designTokens.typography.fontWeight.medium,
          color: designTokens.colors.neutral[800],
          marginBottom: designTokens.spacing[2]
        }}
        id={`${id}-label`}
      >
        Model Complexity
      </div>
      
      <div
        role="radiogroup"
        aria-labelledby={`${id}-label`}
        className={`nam-complexity-selector-options ${focusVisible ? 'focus-visible' : ''}`}
        style={{
          display: 'flex',
          borderRadius: designTokens.borderRadius.lg,
          backgroundColor: designTokens.colors.neutral[300],
          padding: designTokens.spacing[1],
          position: 'relative',
          boxShadow: 'inset 0 1px 2px rgba(24, 35, 48, 0.1)',
          outline: focusVisible ? `2px solid ${designTokens.colors.primary.base}` : 'none',
          outlineOffset: '2px'
        }}
      >
        {/* Simple Option */}
        <ComplexityOption 
          id={`${id}-simple`}
          label="Simple"
          value="simple"
          isSelected={selectedValue === 'simple'}
          isAnimating={isAnimating}
          isHovered={hoverState === 'simple'}
          disabled={disabled}
          onSelect={() => handleChange('simple')}
          onHover={() => setHoverState('simple')}
          onHoverEnd={() => setHoverState(null)}
        >
          <NetworkVisualization 
            nodes={network.nodes}
            connections={network.connections}
            animate={isAnimating}
            isCurrent={selectedValue === 'simple'}
          />
        </ComplexityOption>
        
        {/* Balanced Option */}
        <ComplexityOption 
          id={`${id}-balanced`}
          label="Balanced"
          value="balanced"
          isSelected={selectedValue === 'balanced'}
          isAnimating={isAnimating}
          isHovered={hoverState === 'balanced'}
          disabled={disabled}
          onSelect={() => handleChange('balanced')}
          onHover={() => setHoverState('balanced')}
          onHoverEnd={() => setHoverState(null)}
        >
          <NetworkVisualization 
            nodes={network.nodes}
            connections={network.connections}
            animate={isAnimating}
            isCurrent={selectedValue === 'balanced'}
          />
        </ComplexityOption>
        
        {/* Complex Option */}
        <ComplexityOption 
          id={`${id}-complex`}
          label="Complex"
          value="complex"
          isSelected={selectedValue === 'complex'}
          isAnimating={isAnimating}
          isHovered={hoverState === 'complex'}
          disabled={disabled}
          onSelect={() => handleChange('complex')}
          onHover={() => setHoverState('complex')}
          onHoverEnd={() => setHoverState(null)}
        >
          <NetworkVisualization 
            nodes={network.nodes}
            connections={network.connections}
            animate={isAnimating}
            isCurrent={selectedValue === 'complex'}
          />
        </ComplexityOption>
      </div>
      
      {/* Description Text */}
      <AnimatePresence mode="wait">
        <motion.div
          key={`description-${selectedValue}`}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -5 }}
          transition={{
            duration: 0.25,
            ease: designTokens.animation.easing.entrance
          }}
          className="nam-complexity-description"
          style={{
            fontSize: designTokens.typography.fontSize.sm,
            color: designTokens.colors.neutral[700],
            marginTop: designTokens.spacing[2],
            height: '2.5rem', // Fixed height to prevent layout shift
            display: 'flex',
            alignItems: 'center'
          }}
        >
          {selectedValue === 'simple' && (
            "Faster training with good accuracy for straightforward relationships."
          )}
          {selectedValue === 'balanced' && (
            "Optimal balance between model complexity and training time for most use cases."
          )}
          {selectedValue === 'complex' && (
            "Captures nuanced patterns at the cost of longer training time and more resources."
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

/**
 * Individual option within the complexity selector
 */
const ComplexityOption = ({ 
  id,
  label, 
  value, 
  isSelected, 
  isAnimating,
  isHovered,
  disabled,
  onSelect,
  onHover,
  onHoverEnd,
  children 
}) => {
  return (
    <motion.div
      role="radio"
      aria-checked={isSelected}
      aria-disabled={disabled}
      id={id}
      tabIndex={disabled ? -1 : 0}
      whileTap={!disabled ? { scale: 0.98 } : {}}
      className={`nam-complexity-option ${isSelected ? 'selected' : ''}`}
      onClick={() => !disabled && onSelect()}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          !disabled && onSelect();
        }
      }}
      onMouseEnter={onHover}
      onMouseLeave={onHoverEnd}
      onFocus={onHover}
      onBlur={onHoverEnd}
      style={{
        position: 'relative',
        flex: 1,
        borderRadius: designTokens.borderRadius.md,
        padding: `${designTokens.spacing[3]} ${designTokens.spacing[2]}`,
        backgroundColor: isSelected 
          ? designTokens.colors.neutral[100]
          : isHovered
            ? designTokens.colors.neutral[200]
            : 'transparent',
        boxShadow: isSelected 
          ? '0 2px 4px rgba(0, 0, 0, 0.1)'
          : 'none',
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: `background-color ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}, 
                   box-shadow ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`,
        outline: 'none',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: designTokens.spacing[1]
      }}
    >
      <div
        className="nam-complexity-visualization"
        style={{
          width: '100%',
          height: '100px',
          margin: `${designTokens.spacing[1]} 0`,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        {children}
      </div>
      
      <motion.div 
        className="nam-complexity-label"
        animate={{
          fontWeight: isSelected ? designTokens.typography.fontWeight.semibold : designTokens.typography.fontWeight.regular,
        }}
        style={{
          fontSize: designTokens.typography.fontSize.sm,
          color: isSelected ? designTokens.colors.primary.dark : designTokens.colors.neutral[800],
          textAlign: 'center',
          transition: `color ${designTokens.animation.duration.normal} ${designTokens.animation.easing.standard}`
        }}
      >
        {label}
      </motion.div>
    </motion.div>
  );
};

/**
 * Neural network visualization component
 */
const NetworkVisualization = ({ nodes, connections, animate, isCurrent }) => {
  // Animation variants for connections and nodes
  const connectionVariants = {
    hidden: { opacity: 0, pathLength: 0 },
    visible: (custom) => ({
      opacity: 1,
      pathLength: 1,
      transition: {
        pathLength: {
          delay: custom * 0.01,
          duration: 0.5,
          ease: designTokens.animation.easing.custom.anticipate
        },
        opacity: {
          delay: custom * 0.01,
          duration: 0.3
        }
      }
    })
  };
  
  const nodeVariants = {
    hidden: { opacity: 0, scale: 0 },
    visible: (custom) => ({
      opacity: 1,
      scale: 1,
      transition: {
        delay: custom * 0.02 + 0.2, // Nodes appear after connections begin
        duration: 0.3,
        type: 'spring',
        stiffness: 300,
        damping: 15
      }
    })
  };

  return (
    <svg 
      width="100%" 
      height="100%" 
      viewBox="0 0 200 100"
      style={{ 
        overflow: 'visible',
        filter: isCurrent ? 'drop-shadow(0 2px 3px rgba(0, 0, 0, 0.1))' : 'none' 
      }}
    >
      {/* Connections between nodes */}
      {connections.map((path, i) => (
        <motion.path
          key={`connection-${i}`}
          d={path}
          stroke={designTokens.colors.primary.base}
          strokeWidth={isCurrent ? 1.5 : 1}
          strokeOpacity={isCurrent ? 0.7 : 0.4}
          fill="none"
          initial={animate ? "hidden" : "visible"}
          animate="visible"
          custom={i}
          variants={connectionVariants}
        />
      ))}
      
      {/* Nodes */}
      {nodes.map((node, i) => (
        <motion.circle
          key={`node-${i}`}
          cx={node.cx}
          cy={node.cy}
          r={node.r}
          fill={designTokens.colors.primary.base}
          initial={animate ? "hidden" : "visible"}
          animate="visible"
          custom={i}
          variants={nodeVariants}
        />
      ))}
    </svg>
  );
};

export default ComplexitySelector;
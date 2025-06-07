/**
 * Neural Additive Models Design System Tokens
 * 
 * This file defines the fundamental design constants that ensure
 * visual and behavioral consistency throughout the NAM experience.
 * Every value has been carefully considered and calibrated to
 * create a precise, emotionally resonant interface.
 */

// Color System
// ------------
// The color palette is derived from the primary SAP HANA brand colors
// but evolved to create a distinctive feeling specific to the Neural Additive Models
// experience while maintaining a clear connection to the parent brand.

export const colors = {
  // Primary Brand Colors
  primary: {
    base: '#0070D2',    // Core blue - the foundation of our palette
    light: '#CFEAFF',   // Light interaction states
    medium: '#2D88E2',  // Secondary actions
    dark: '#004B98',    // Focused states
    contrast: '#FFFFFF' // Text on primary colors
  },

  // Neutrals - the canvas of our interface
  neutral: {
    100: '#FFFFFF', // Pure white backgrounds
    200: '#F9FBFD', // Subtle off-white for cards
    300: '#F2F6FA', // Background areas
    400: '#E5EAEF', // Subtle borders
    500: '#D1D9E0', // Strong borders
    600: '#B3BEC9', // Muted text
    700: '#8596A5', // Secondary text
    800: '#485C6E', // Primary text
    900: '#243342'  // Emphasized text
  },

  // Feature Contribution Semantic Colors
  contribution: {
    positive: {
      weak: '#D4EDDA',
      medium: '#86CCA5',
      strong: '#28A745'
    },
    negative: {
      weak: '#F8D7DA',
      medium: '#EDA1A8',
      strong: '#DC3545'
    },
    neutral: '#F2F6F8'
  },

  // Feedback States
  feedback: {
    success: '#2E7D32',
    info: '#0288D1',
    warning: '#F57C00',
    error: '#D32F2F'
  },

  // Charts and Visualizations
  viz: {
    categorical: [
      '#2C83EC', // Primary blue  
      '#46BAA3', // Teal
      '#9379F2', // Purple
      '#FF8162', // Coral
      '#FFB74A', // Amber
      '#2DB7B5', // Sea green
      '#6871DE', // Indigo
      '#EE6868'  // Red
    ],
    sequential: [
      '#E5F2FF',
      '#CFEAFF',
      '#A1CDFF',
      '#70B0FF',
      '#4C95FC',
      '#2D88E2',
      '#1A6DBC',
      '#004B98'
    ],
    diverging: [
      '#DC3545', // Negative extreme
      '#EDA1A8',
      '#F8D7DA',
      '#F2F6F8', // Neutral center
      '#D4EDDA', 
      '#86CCA5',
      '#28A745'  // Positive extreme
    ]
  },

  // Special UI Elements
  special: {
    focus: 'rgba(44, 131, 236, 0.4)', // Focus ring with 40% opacity
    overlay: 'rgba(36, 51, 66, 0.7)',  // Modal overlay
    shadow: 'rgba(24, 35, 48, 0.2)'    // Elevation shadow
  }
};

// Typography
// ----------
// A harmonious type system that balances legibility with character.
// Base sizes use a dynamic modular scale with careful optical adjustments.

export const typography = {
  fontFamily: {
    primary: '"-apple-system", "BlinkMacSystemFont", "San Francisco", "Segoe UI", "Roboto", "Helvetica Neue", sans-serif',
    mono: '"SFMono-Regular", "Menlo", "Monaco", "Consolas", "Liberation Mono", "Courier New", monospace',
    display: '"SF Pro Display", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", sans-serif'
  },
  
  // Base size scales with 1.2 ratio
  fontSize: {
    xs: '0.694rem',    // 11.1px at 16px base
    sm: '0.833rem',    // 13.33px
    base: '1rem',      // 16px
    md: '1.2rem',      // 19.2px
    lg: '1.44rem',     // 23.04px
    xl: '1.728rem',    // 27.65px
    '2xl': '2.074rem', // 33.18px
    '3xl': '2.488rem'  // 39.81px
  },
  
  fontWeight: {
    light: 300,
    regular: 400,
    medium: 500,
    semibold: 600,
    bold: 700
  },
  
  letterSpacing: {
    tight: '-0.01em',
    normal: '0',
    wide: '0.02em',
    wider: '0.05em'
  },
  
  lineHeight: {
    none: 1,
    tight: 1.2,
    snug: 1.375,
    normal: 1.5,
    relaxed: 1.625,
    loose: 2
  }
};

// Spacing
// -------
// An 8px-based spacing system with 4px refinements for precision adjustments
// where necessary for optical alignment and balance.

export const spacing = {
  '0': '0',
  '1': '0.25rem',   // 4px
  '2': '0.5rem',    // 8px
  '3': '0.75rem',   // 12px
  '4': '1rem',      // 16px
  '5': '1.25rem',   // 20px
  '6': '1.5rem',    // 24px
  '8': '2rem',      // 32px
  '10': '2.5rem',   // 40px
  '12': '3rem',     // 48px
  '16': '4rem',     // 64px
  '20': '5rem',     // 80px
  '24': '6rem',     // 96px
  '32': '8rem'      // 128px
};

// Layers and Elevation
// -------------------
// Carefully calibrated shadows and z-indices to create meaningful
// spatial relationships between elements.

export const elevation = {
  z: {
    base: 0,
    raised: 10,
    dropdown: 20,
    sticky: 30,
    fixed: 40,
    modal: 50,
    popover: 60,
    tooltip: 70,
    max: 9999
  },
  
  shadow: {
    none: 'none',
    xs: '0 1px 2px rgba(24, 35, 48, 0.05)',
    sm: '0 1px 3px rgba(24, 35, 48, 0.1), 0 1px 2px rgba(24, 35, 48, 0.06)',
    md: '0 4px 6px -1px rgba(24, 35, 48, 0.1), 0 2px 4px -1px rgba(24, 35, 48, 0.06)',
    lg: '0 10px 15px -3px rgba(24, 35, 48, 0.1), 0 4px 6px -2px rgba(24, 35, 48, 0.05)',
    xl: '0 20px 25px -5px rgba(24, 35, 48, 0.1), 0 10px 10px -5px rgba(24, 35, 48, 0.04)',
    '2xl': '0 25px 50px -12px rgba(24, 35, 48, 0.25)',
    inner: 'inset 0 2px 4px 0 rgba(24, 35, 48, 0.06)'
  }
};

// Animation & Timing
// -----------------
// Precise timing and easing functions that create a specific
// personality and feel through motion.

export const animation = {
  easing: {
    // Entrance animations - fluid and welcoming
    entrance: 'cubic-bezier(0.0, 0.0, 0.2, 1)',
    // Exit animations - quick but not abrupt
    exit: 'cubic-bezier(0.4, 0.0, 1, 1)',
    // Emphasis animations - slightly elastic for delight
    emphasis: 'cubic-bezier(0.2, 0.0, 0.0, 1.0)',
    // Standard animations - balanced and neutral
    standard: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
    // Precise control for custom interactions
    custom: {
      anticipate: 'cubic-bezier(0.38, -0.4, 0.88, 0.65)',
      overshoot: 'cubic-bezier(0.34, 1.56, 0.64, 1)'
    }
  },

  duration: {
    fastest: '50ms',
    faster: '100ms',
    fast: '150ms',
    normal: '200ms',
    slow: '300ms',
    slower: '400ms',
    slowest: '500ms',
    
    // Semantic durations for specific actions
    emphasis: '350ms',      // For attention-grabbing animations
    transition: '200ms',    // For UI state changes
    entrance: '250ms',      // For elements entering the screen
    exit: '150ms',          // For elements leaving the screen
    complex: '400ms'        // For more intricate animations
  }
};

// Border Radius
// ------------
// Consistent rounding that creates a distinctive yet subtle personality.

export const borderRadius = {
  none: '0',
  sm: '0.125rem',   // 2px
  md: '0.25rem',    // 4px
  lg: '0.375rem',   // 6px
  xl: '0.5rem',     // 8px
  '2xl': '0.75rem', // 12px
  '3xl': '1rem',    // 16px
  full: '9999px'    // Fully rounded (for circles or pills)
};

// Breakpoints
// ----------
// Responsive design breakpoints for consistent adaptive layouts.

export const breakpoints = {
  xs: '480px',   // Small phones
  sm: '640px',   // Larger phones
  md: '768px',   // Tablets
  lg: '1024px',  // Laptops/desktops
  xl: '1280px',  // Large desktops
  '2xl': '1536px' // Extra large screens
};

// Focus States
// -----------
// Distinctive focus indicators that maintain accessibility while
// fitting into the design language.

export const focusRing = {
  default: `0 0 0 2px ${colors.primary.light}, 0 0 0 4px ${colors.primary.base}`,
  inset: `inset 0 0 0 2px ${colors.primary.base}`,
  highContrast: `0 0 0 2px white, 0 0 0 4px ${colors.primary.dark}`,
  subtle: `0 0 0 1px ${colors.neutral[100]}, 0 0 0 3px ${colors.primary.light}`
};

// Export all tokens as a unified design system
export const designTokens = {
  colors,
  typography,
  spacing,
  elevation,
  animation,
  borderRadius,
  breakpoints,
  focusRing
};

// Default export
export default designTokens;
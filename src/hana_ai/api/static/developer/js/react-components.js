/**
 * React Components for Developer Studio
 * 
 * This module provides React-based UI components for the Developer Studio.
 * These components are designed to be used with the existing Developer Studio infrastructure.
 */

// Create root element for React components if not exists
if (!document.getElementById('react-root')) {
    const reactRoot = document.createElement('div');
    reactRoot.id = 'react-root';
    document.body.appendChild(reactRoot);
}

/**
 * Button Component
 */
const Button = React.forwardRef(({ 
    children, 
    variant = 'primary', 
    size = 'medium', 
    icon, 
    disabled = false,
    fullWidth = false,
    onClick,
    className = '',
    ...props 
}, ref) => {
    // Button style classes
    const variantClasses = {
        primary: 'bg-blue-600 hover:bg-blue-700 text-white',
        secondary: 'bg-gray-200 hover:bg-gray-300 text-gray-800',
        success: 'bg-green-600 hover:bg-green-700 text-white',
        danger: 'bg-red-600 hover:bg-red-700 text-white',
        outline: 'bg-transparent border border-blue-600 text-blue-600 hover:bg-blue-50',
        ghost: 'bg-transparent text-gray-700 hover:bg-gray-100',
    };
    
    const sizeClasses = {
        small: 'px-2 py-1 text-sm',
        medium: 'px-4 py-2',
        large: 'px-6 py-3 text-lg',
    };
    
    const baseClasses = 'rounded font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 flex items-center justify-center';
    const widthClass = fullWidth ? 'w-full' : '';
    const disabledClass = disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer';
    
    const buttonClasses = `${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${widthClass} ${disabledClass} ${className}`;
    
    return (
        <button 
            className={buttonClasses}
            disabled={disabled}
            onClick={onClick}
            ref={ref}
            {...props}
        >
            {icon && <span className="mr-2">{icon}</span>}
            {children}
        </button>
    );
});

/**
 * Input Component
 */
const Input = React.forwardRef(({
    label,
    id,
    type = 'text',
    placeholder,
    error,
    helperText,
    className = '',
    disabled = false,
    required = false,
    fullWidth = false,
    startIcon,
    endIcon,
    ...props
}, ref) => {
    const uniqueId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
    const widthClass = fullWidth ? 'w-full' : '';
    
    return (
        <div className={`mb-4 ${widthClass} ${className}`}>
            {label && (
                <label 
                    htmlFor={uniqueId}
                    className="block text-sm font-medium text-gray-700 mb-1"
                >
                    {label}
                    {required && <span className="text-red-500 ml-1">*</span>}
                </label>
            )}
            
            <div className="relative">
                {startIcon && (
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-500">
                        {startIcon}
                    </div>
                )}
                
                <input
                    id={uniqueId}
                    type={type}
                    className={`block border ${error ? 'border-red-500' : 'border-gray-300'} rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${widthClass} ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''} ${startIcon ? 'pl-10' : ''} ${endIcon ? 'pr-10' : ''}`}
                    placeholder={placeholder}
                    disabled={disabled}
                    required={required}
                    ref={ref}
                    {...props}
                />
                
                {endIcon && (
                    <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none text-gray-500">
                        {endIcon}
                    </div>
                )}
            </div>
            
            {error && helperText && (
                <p className="mt-1 text-sm text-red-600">{helperText}</p>
            )}
            
            {!error && helperText && (
                <p className="mt-1 text-sm text-gray-500">{helperText}</p>
            )}
        </div>
    );
});

/**
 * Select Component
 */
const Select = React.forwardRef(({
    label,
    id,
    options = [],
    error,
    helperText,
    className = '',
    disabled = false,
    required = false,
    fullWidth = false,
    ...props
}, ref) => {
    const uniqueId = id || `select-${Math.random().toString(36).substr(2, 9)}`;
    const widthClass = fullWidth ? 'w-full' : '';
    
    return (
        <div className={`mb-4 ${widthClass} ${className}`}>
            {label && (
                <label 
                    htmlFor={uniqueId}
                    className="block text-sm font-medium text-gray-700 mb-1"
                >
                    {label}
                    {required && <span className="text-red-500 ml-1">*</span>}
                </label>
            )}
            
            <select
                id={uniqueId}
                className={`block border ${error ? 'border-red-500' : 'border-gray-300'} rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${widthClass} ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''} appearance-none bg-no-repeat bg-right pr-8`}
                disabled={disabled}
                required={required}
                ref={ref}
                {...props}
            >
                {options.map((option) => (
                    <option 
                        key={option.value} 
                        value={option.value}
                        disabled={option.disabled}
                    >
                        {option.label}
                    </option>
                ))}
            </select>
            
            {error && helperText && (
                <p className="mt-1 text-sm text-red-600">{helperText}</p>
            )}
            
            {!error && helperText && (
                <p className="mt-1 text-sm text-gray-500">{helperText}</p>
            )}
        </div>
    );
});

/**
 * Checkbox Component
 */
const Checkbox = React.forwardRef(({
    label,
    id,
    checked,
    onChange,
    error,
    helperText,
    className = '',
    disabled = false,
    required = false,
    ...props
}, ref) => {
    const uniqueId = id || `checkbox-${Math.random().toString(36).substr(2, 9)}`;
    
    return (
        <div className={`mb-4 ${className}`}>
            <div className="flex items-start">
                <div className="flex items-center h-5">
                    <input
                        id={uniqueId}
                        type="checkbox"
                        className={`w-4 h-4 text-blue-600 border ${error ? 'border-red-500' : 'border-gray-300'} rounded focus:ring-2 focus:ring-blue-500 ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`}
                        checked={checked}
                        onChange={onChange}
                        disabled={disabled}
                        required={required}
                        ref={ref}
                        {...props}
                    />
                </div>
                
                <div className="ml-3 text-sm">
                    {label && (
                        <label 
                            htmlFor={uniqueId}
                            className={`font-medium ${disabled ? 'text-gray-500' : 'text-gray-700'}`}
                        >
                            {label}
                            {required && <span className="text-red-500 ml-1">*</span>}
                        </label>
                    )}
                    
                    {error && helperText && (
                        <p className="mt-1 text-sm text-red-600">{helperText}</p>
                    )}
                    
                    {!error && helperText && (
                        <p className="mt-1 text-sm text-gray-500">{helperText}</p>
                    )}
                </div>
            </div>
        </div>
    );
});

/**
 * Radio Group Component
 */
const RadioGroup = React.forwardRef(({
    label,
    name,
    options = [],
    value,
    onChange,
    error,
    helperText,
    className = '',
    disabled = false,
    required = false,
    ...props
}, ref) => {
    return (
        <div className={`mb-4 ${className}`} ref={ref} {...props}>
            {label && (
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    {label}
                    {required && <span className="text-red-500 ml-1">*</span>}
                </label>
            )}
            
            <div className="space-y-2">
                {options.map((option) => {
                    const uniqueId = `radio-${name}-${option.value}`;
                    
                    return (
                        <div key={option.value} className="flex items-center">
                            <input
                                id={uniqueId}
                                type="radio"
                                name={name}
                                value={option.value}
                                checked={value === option.value}
                                onChange={onChange}
                                disabled={disabled || option.disabled}
                                required={required}
                                className={`w-4 h-4 text-blue-600 border ${error ? 'border-red-500' : 'border-gray-300'} focus:ring-2 focus:ring-blue-500 ${disabled || option.disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`}
                            />
                            
                            <label 
                                htmlFor={uniqueId}
                                className={`ml-3 text-sm font-medium ${disabled || option.disabled ? 'text-gray-500' : 'text-gray-700'}`}
                            >
                                {option.label}
                            </label>
                        </div>
                    );
                })}
            </div>
            
            {error && helperText && (
                <p className="mt-1 text-sm text-red-600">{helperText}</p>
            )}
            
            {!error && helperText && (
                <p className="mt-1 text-sm text-gray-500">{helperText}</p>
            )}
        </div>
    );
});

/**
 * Card Component
 */
const Card = ({
    children,
    title,
    subtitle,
    actions,
    className = '',
    variant = 'default',
    ...props
}) => {
    const variantClasses = {
        default: 'bg-white',
        outlined: 'bg-white border border-gray-300',
        elevated: 'bg-white shadow-md',
        flat: 'bg-gray-50',
    };
    
    return (
        <div 
            className={`rounded-lg ${variantClasses[variant]} overflow-hidden ${className}`}
            {...props}
        >
            {(title || subtitle) && (
                <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                    {title && <h3 className="text-lg font-medium text-gray-900">{title}</h3>}
                    {subtitle && <p className="mt-1 text-sm text-gray-500">{subtitle}</p>}
                </div>
            )}
            
            <div className="p-4">
                {children}
            </div>
            
            {actions && (
                <div className="px-4 py-3 bg-gray-50 border-t border-gray-200 flex justify-end space-x-2">
                    {actions}
                </div>
            )}
        </div>
    );
};

/**
 * Tabs Component
 */
const Tabs = ({ 
    tabs = [], 
    activeTab, 
    onChange,
    variant = 'default',
    className = '',
    ...props 
}) => {
    const variantClasses = {
        default: 'border-b border-gray-200',
        pills: 'space-x-2',
        underline: 'border-b border-gray-200',
    };
    
    const getTabClasses = (tab, index) => {
        const isActive = activeTab === index;
        
        if (variant === 'default') {
            return `px-4 py-2 border-b-2 ${isActive ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} font-medium`;
        } else if (variant === 'pills') {
            return `px-4 py-2 rounded-full ${isActive ? 'bg-blue-100 text-blue-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'} font-medium`;
        } else if (variant === 'underline') {
            return `px-4 py-2 ${isActive ? 'text-blue-600 border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-700'} font-medium`;
        }
        
        return '';
    };
    
    return (
        <div className={`${variantClasses[variant]} ${className}`} {...props}>
            <div className="flex space-x-4">
                {tabs.map((tab, index) => (
                    <button
                        key={index}
                        className={getTabClasses(tab, index)}
                        onClick={() => onChange(index)}
                    >
                        {typeof tab === 'string' ? tab : tab.label}
                    </button>
                ))}
            </div>
        </div>
    );
};

/**
 * Modal Component
 */
const Modal = ({
    isOpen,
    onClose,
    title,
    children,
    footer,
    size = 'medium',
    className = '',
    overlayClassName = '',
    ...props
}) => {
    const [isVisible, setIsVisible] = React.useState(isOpen);
    
    React.useEffect(() => {
        if (isOpen) {
            setIsVisible(true);
        } else {
            // Delay hiding to allow animation
            const timer = setTimeout(() => setIsVisible(false), 300);
            return () => clearTimeout(timer);
        }
    }, [isOpen]);
    
    if (!isVisible) return null;
    
    const sizeClasses = {
        small: 'max-w-md',
        medium: 'max-w-lg',
        large: 'max-w-2xl',
        extraLarge: 'max-w-4xl',
        fullWidth: 'max-w-full mx-4',
    };
    
    const modalClasses = `
        transform transition-all duration-300 ease-out
        ${isOpen ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}
        bg-white rounded-lg shadow-xl overflow-hidden
        ${sizeClasses[size]} w-full
        mx-auto my-8 z-50
        ${className}
    `;
    
    const overlayClasses = `
        fixed inset-0 bg-black bg-opacity-50
        flex items-center justify-center p-4
        transition-opacity duration-300 ease-out
        ${isOpen ? 'opacity-100' : 'opacity-0'}
        ${overlayClassName}
    `;
    
    return (
        <div className={overlayClasses} onClick={onClose}>
            <div 
                className={modalClasses} 
                onClick={e => e.stopPropagation()}
                {...props}
            >
                {title && (
                    <div className="px-6 py-4 border-b border-gray-200">
                        <div className="flex items-center justify-between">
                            <h3 className="text-lg font-medium text-gray-900">{title}</h3>
                            <button
                                type="button"
                                className="text-gray-400 hover:text-gray-500 focus:outline-none"
                                onClick={onClose}
                            >
                                <span className="sr-only">Close</span>
                                <i className="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                )}
                
                <div className="px-6 py-4">
                    {children}
                </div>
                
                {footer && (
                    <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
                        {footer}
                    </div>
                )}
            </div>
        </div>
    );
};

/**
 * Alert Component
 */
const Alert = ({
    children,
    title,
    severity = 'info',
    onClose,
    className = '',
    ...props
}) => {
    const severityClasses = {
        info: 'bg-blue-50 text-blue-800 border-blue-200',
        success: 'bg-green-50 text-green-800 border-green-200',
        warning: 'bg-yellow-50 text-yellow-800 border-yellow-200',
        error: 'bg-red-50 text-red-800 border-red-200',
    };
    
    const severityIcons = {
        info: <i className="fas fa-info-circle text-blue-500"></i>,
        success: <i className="fas fa-check-circle text-green-500"></i>,
        warning: <i className="fas fa-exclamation-triangle text-yellow-500"></i>,
        error: <i className="fas fa-exclamation-circle text-red-500"></i>,
    };
    
    return (
        <div 
            className={`rounded border p-4 flex ${severityClasses[severity]} ${className}`}
            role="alert"
            {...props}
        >
            <div className="flex-shrink-0 mr-3">
                {severityIcons[severity]}
            </div>
            <div className="flex-1">
                {title && <div className="font-medium mb-1">{title}</div>}
                <div className="text-sm">{children}</div>
            </div>
            {onClose && (
                <button
                    type="button"
                    className="flex-shrink-0 ml-3 -mt-0.5 -mr-1 text-gray-400 hover:text-gray-500 focus:outline-none"
                    onClick={onClose}
                >
                    <span className="sr-only">Close</span>
                    <i className="fas fa-times"></i>
                </button>
            )}
        </div>
    );
};

/**
 * Toast Component
 */
const Toast = ({
    message,
    severity = 'info',
    duration = 3000,
    position = 'bottom-right',
    onClose,
    className = '',
    ...props
}) => {
    const [isVisible, setIsVisible] = React.useState(true);
    
    React.useEffect(() => {
        if (duration !== null) {
            const timer = setTimeout(() => {
                setIsVisible(false);
                if (onClose) {
                    setTimeout(onClose, 300); // After animation
                }
            }, duration);
            
            return () => clearTimeout(timer);
        }
    }, [duration, onClose]);
    
    const severityClasses = {
        info: 'bg-blue-100 text-blue-800 border-blue-200',
        success: 'bg-green-100 text-green-800 border-green-200',
        warning: 'bg-yellow-100 text-yellow-800 border-yellow-200',
        error: 'bg-red-100 text-red-800 border-red-200',
    };
    
    const severityIcons = {
        info: <i className="fas fa-info-circle text-blue-500"></i>,
        success: <i className="fas fa-check-circle text-green-500"></i>,
        warning: <i className="fas fa-exclamation-triangle text-yellow-500"></i>,
        error: <i className="fas fa-exclamation-circle text-red-500"></i>,
    };
    
    const positionClasses = {
        'top-left': 'top-4 left-4',
        'top-right': 'top-4 right-4',
        'bottom-left': 'bottom-4 left-4',
        'bottom-right': 'bottom-4 right-4',
        'top-center': 'top-4 left-1/2 transform -translate-x-1/2',
        'bottom-center': 'bottom-4 left-1/2 transform -translate-x-1/2',
    };
    
    const toastClasses = `
        fixed z-50 rounded px-4 py-3 shadow-lg border
        transition-all duration-300 ease-in-out
        ${positionClasses[position]}
        ${severityClasses[severity]}
        ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}
        ${className}
    `;
    
    return (
        <div className={toastClasses} role="alert" {...props}>
            <div className="flex items-center">
                <div className="flex-shrink-0 mr-3">
                    {severityIcons[severity]}
                </div>
                <div className="flex-1 mr-2">
                    {message}
                </div>
                <button
                    type="button"
                    className="flex-shrink-0 text-gray-400 hover:text-gray-500 focus:outline-none"
                    onClick={() => {
                        setIsVisible(false);
                        if (onClose) {
                            setTimeout(onClose, 300); // After animation
                        }
                    }}
                >
                    <span className="sr-only">Close</span>
                    <i className="fas fa-times"></i>
                </button>
            </div>
        </div>
    );
};

/**
 * Tooltip Component
 */
const Tooltip = ({
    children,
    content,
    position = 'top',
    className = '',
    ...props
}) => {
    const [isVisible, setIsVisible] = React.useState(false);
    const [tooltipPosition, setTooltipPosition] = React.useState({ top: 0, left: 0 });
    const childRef = React.useRef(null);
    const tooltipRef = React.useRef(null);
    
    const showTooltip = () => {
        if (childRef.current && tooltipRef.current) {
            const childRect = childRef.current.getBoundingClientRect();
            const tooltipRect = tooltipRef.current.getBoundingClientRect();
            
            let top = 0;
            let left = 0;
            
            switch (position) {
                case 'top':
                    top = childRect.top - tooltipRect.height - 8;
                    left = childRect.left + (childRect.width / 2) - (tooltipRect.width / 2);
                    break;
                case 'bottom':
                    top = childRect.bottom + 8;
                    left = childRect.left + (childRect.width / 2) - (tooltipRect.width / 2);
                    break;
                case 'left':
                    top = childRect.top + (childRect.height / 2) - (tooltipRect.height / 2);
                    left = childRect.left - tooltipRect.width - 8;
                    break;
                case 'right':
                    top = childRect.top + (childRect.height / 2) - (tooltipRect.height / 2);
                    left = childRect.right + 8;
                    break;
            }
            
            setTooltipPosition({ top, left });
        }
        
        setIsVisible(true);
    };
    
    const hideTooltip = () => {
        setIsVisible(false);
    };
    
    const positionArrowClasses = {
        'top': 'after:bottom-[-6px] after:border-t-gray-700',
        'bottom': 'after:top-[-6px] after:border-b-gray-700',
        'left': 'after:right-[-6px] after:border-l-gray-700',
        'right': 'after:left-[-6px] after:border-r-gray-700',
    };
    
    const tooltipClasses = `
        fixed z-50 py-1 px-2 bg-gray-700 text-white text-sm rounded
        transition-opacity duration-200 ease-in-out pointer-events-none
        after:content-[''] after:absolute after:border-[6px] after:border-transparent
        ${positionArrowClasses[position]}
        ${isVisible ? 'opacity-100' : 'opacity-0'}
        ${className}
    `;
    
    return (
        <>
            <div
                ref={childRef}
                onMouseEnter={showTooltip}
                onMouseLeave={hideTooltip}
                onFocus={showTooltip}
                onBlur={hideTooltip}
                className="inline-block"
            >
                {children}
            </div>
            
            <div
                ref={tooltipRef}
                className={tooltipClasses}
                style={{
                    top: `${tooltipPosition.top}px`,
                    left: `${tooltipPosition.left}px`,
                }}
                {...props}
            >
                {content}
            </div>
        </>
    );
};

/**
 * Badge Component
 */
const Badge = ({
    children,
    content,
    color = 'primary',
    variant = 'standard',
    overlap = false,
    className = '',
    ...props
}) => {
    const colorClasses = {
        primary: {
            standard: 'bg-blue-500 text-white',
            outlined: 'bg-transparent text-blue-500 border border-blue-500',
            dot: 'bg-blue-500',
        },
        secondary: {
            standard: 'bg-gray-500 text-white',
            outlined: 'bg-transparent text-gray-500 border border-gray-500',
            dot: 'bg-gray-500',
        },
        success: {
            standard: 'bg-green-500 text-white',
            outlined: 'bg-transparent text-green-500 border border-green-500',
            dot: 'bg-green-500',
        },
        warning: {
            standard: 'bg-yellow-500 text-white',
            outlined: 'bg-transparent text-yellow-500 border border-yellow-500',
            dot: 'bg-yellow-500',
        },
        error: {
            standard: 'bg-red-500 text-white',
            outlined: 'bg-transparent text-red-500 border border-red-500',
            dot: 'bg-red-500',
        },
    };
    
    const variantClasses = {
        standard: 'min-w-[20px] h-5 px-1 rounded-full text-xs flex items-center justify-center',
        outlined: 'min-w-[20px] h-5 px-1 rounded-full text-xs flex items-center justify-center',
        dot: 'w-2 h-2 rounded-full',
    };
    
    const badgeClasses = `
        ${colorClasses[color][variant]}
        ${variantClasses[variant]}
        ${className}
    `;
    
    const positionClasses = overlap
        ? 'absolute -top-1 -right-1 transform translate-x-1/2 -translate-y-1/2'
        : 'absolute -top-1 -right-1';
    
    return (
        <div className="relative inline-flex">
            {children}
            
            <span className={`${badgeClasses} ${positionClasses}`} {...props}>
                {variant !== 'dot' && content}
            </span>
        </div>
    );
};

/**
 * Progress Component
 */
const Progress = ({
    value = 0,
    variant = 'determinate',
    color = 'primary',
    thickness = 'medium',
    label = false,
    className = '',
    ...props
}) => {
    const colorClasses = {
        primary: 'bg-blue-500',
        secondary: 'bg-gray-500',
        success: 'bg-green-500',
        warning: 'bg-yellow-500',
        error: 'bg-red-500',
    };
    
    const thicknessClasses = {
        thin: 'h-1',
        medium: 'h-2',
        thick: 'h-3',
    };
    
    const baseClasses = 'w-full bg-gray-200 rounded overflow-hidden';
    
    const progressClasses = `
        ${baseClasses}
        ${thicknessClasses[thickness]}
        ${className}
    `;
    
    // For indeterminate animation
    const indeterminateClasses = variant === 'indeterminate'
        ? 'relative before:absolute before:inset-0 before:translate-x-[-100%] before:animate-[progress-indeterminate_1.5s_infinite]'
        : '';
    
    return (
        <div className="w-full">
            <div className={progressClasses} {...props}>
                <div
                    className={`${colorClasses[color]} ${indeterminateClasses}`}
                    style={{
                        width: variant === 'determinate' ? `${Math.min(100, Math.max(0, value))}%` : '100%',
                        transition: 'width 0.4s ease',
                    }}
                ></div>
            </div>
            
            {label && (
                <div className="mt-1 text-xs text-right text-gray-500">
                    {variant === 'determinate' ? `${Math.round(value)}%` : ''}
                </div>
            )}
        </div>
    );
};

/**
 * Dropdown Component
 */
const Dropdown = ({
    trigger,
    items = [],
    isOpen,
    onToggle,
    className = '',
    ...props
}) => {
    const [open, setOpen] = React.useState(isOpen || false);
    const dropdownRef = React.useRef(null);
    
    React.useEffect(() => {
        if (isOpen !== undefined) {
            setOpen(isOpen);
        }
    }, [isOpen]);
    
    const handleToggle = () => {
        const newState = !open;
        setOpen(newState);
        
        if (onToggle) {
            onToggle(newState);
        }
    };
    
    // Close dropdown when clicking outside
    React.useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setOpen(false);
                
                if (onToggle) {
                    onToggle(false);
                }
            }
        };
        
        document.addEventListener('mousedown', handleClickOutside);
        
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [onToggle]);
    
    return (
        <div className={`relative inline-block ${className}`} ref={dropdownRef} {...props}>
            <div onClick={handleToggle}>
                {trigger}
            </div>
            
            {open && (
                <div className="absolute right-0 z-10 mt-2 w-48 rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                    <div className="py-1">
                        {items.map((item, index) => {
                            if (item.divider) {
                                return <hr key={index} className="my-1 border-gray-200" />;
                            }
                            
                            return (
                                <div
                                    key={index}
                                    className={`block px-4 py-2 text-sm ${item.disabled ? 'text-gray-400 cursor-not-allowed' : 'text-gray-700 hover:bg-gray-100 cursor-pointer'}`}
                                    onClick={() => {
                                        if (!item.disabled && item.onClick) {
                                            item.onClick();
                                            setOpen(false);
                                            
                                            if (onToggle) {
                                                onToggle(false);
                                            }
                                        }
                                    }}
                                >
                                    <div className="flex items-center">
                                        {item.icon && <span className="mr-2">{item.icon}</span>}
                                        {item.label}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};

/**
 * Data Table Component
 */
const DataTable = ({
    columns = [],
    data = [],
    pagination = false,
    pageSize = 10,
    className = '',
    ...props
}) => {
    const [currentPage, setCurrentPage] = React.useState(1);
    const [sortConfig, setSortConfig] = React.useState(null);
    
    // Sort data based on sort config
    const sortedData = React.useMemo(() => {
        if (!sortConfig) return data;
        
        return [...data].sort((a, b) => {
            const aValue = a[sortConfig.key];
            const bValue = b[sortConfig.key];
            
            if (aValue < bValue) {
                return sortConfig.direction === 'ascending' ? -1 : 1;
            }
            if (aValue > bValue) {
                return sortConfig.direction === 'ascending' ? 1 : -1;
            }
            return 0;
        });
    }, [data, sortConfig]);
    
    // Paginate data
    const paginatedData = React.useMemo(() => {
        if (!pagination) return sortedData;
        
        const startIdx = (currentPage - 1) * pageSize;
        return sortedData.slice(startIdx, startIdx + pageSize);
    }, [sortedData, currentPage, pageSize, pagination]);
    
    // Handle sorting
    const handleSort = (key) => {
        let direction = 'ascending';
        
        if (sortConfig && sortConfig.key === key) {
            direction = sortConfig.direction === 'ascending' ? 'descending' : 'ascending';
        }
        
        setSortConfig({ key, direction });
    };
    
    // Total pages
    const totalPages = pagination ? Math.ceil(data.length / pageSize) : 1;
    
    return (
        <div className={`overflow-x-auto ${className}`} {...props}>
            <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                    <tr>
                        {columns.map((column, index) => (
                            <th
                                key={index}
                                scope="col"
                                className={`px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${column.sortable ? 'cursor-pointer select-none' : ''}`}
                                onClick={() => column.sortable && handleSort(column.key)}
                            >
                                <div className="flex items-center">
                                    {column.header || column.key}
                                    
                                    {column.sortable && (
                                        <span className="ml-2">
                                            {sortConfig && sortConfig.key === column.key ? (
                                                sortConfig.direction === 'ascending' ? (
                                                    <i className="fas fa-sort-up"></i>
                                                ) : (
                                                    <i className="fas fa-sort-down"></i>
                                                )
                                            ) : (
                                                <i className="fas fa-sort text-gray-300"></i>
                                            )}
                                        </span>
                                    )}
                                </div>
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                    {paginatedData.length > 0 ? (
                        paginatedData.map((row, rowIndex) => (
                            <tr key={rowIndex} className="hover:bg-gray-50">
                                {columns.map((column, colIndex) => (
                                    <td key={colIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                        {column.render ? column.render(row[column.key], row) : row[column.key]}
                                    </td>
                                ))}
                            </tr>
                        ))
                    ) : (
                        <tr>
                            <td colSpan={columns.length} className="px-6 py-4 text-center text-sm text-gray-500">
                                No data available
                            </td>
                        </tr>
                    )}
                </tbody>
            </table>
            
            {pagination && (
                <div className="flex items-center justify-between px-4 py-3 bg-white border-t border-gray-200 sm:px-6">
                    <div className="flex items-center">
                        <p className="text-sm text-gray-700">
                            Showing <span className="font-medium">{paginatedData.length > 0 ? (currentPage - 1) * pageSize + 1 : 0}</span> to <span className="font-medium">{Math.min(currentPage * pageSize, data.length)}</span> of <span className="font-medium">{data.length}</span> results
                        </p>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                        <button
                            className={`px-3 py-1 rounded border ${currentPage === 1 ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-white text-gray-700 hover:bg-gray-50'}`}
                            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                            disabled={currentPage === 1}
                        >
                            <i className="fas fa-chevron-left"></i>
                        </button>
                        
                        <span className="text-sm text-gray-700">
                            Page {currentPage} of {totalPages}
                        </span>
                        
                        <button
                            className={`px-3 py-1 rounded border ${currentPage === totalPages ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-white text-gray-700 hover:bg-gray-50'}`}
                            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                            disabled={currentPage === totalPages}
                        >
                            <i className="fas fa-chevron-right"></i>
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

/**
 * DragDropContainer Component
 */
const DragDropContainer = ({
    children,
    onDrop,
    acceptTypes = [],
    className = '',
    ...props
}) => {
    const [isDragging, setIsDragging] = React.useState(false);
    const containerRef = React.useRef(null);
    
    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        if (!isDragging) {
            setIsDragging(true);
        }
    };
    
    const handleDragEnter = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };
    
    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        // Only set isDragging to false if the mouse leaves the container
        if (containerRef.current && !containerRef.current.contains(e.relatedTarget)) {
            setIsDragging(false);
        }
    };
    
    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        
        const files = Array.from(e.dataTransfer.files);
        
        // Filter files by accepted types if specified
        const filteredFiles = acceptTypes.length > 0
            ? files.filter(file => acceptTypes.some(type => file.type.match(type)))
            : files;
        
        if (onDrop && filteredFiles.length > 0) {
            onDrop(filteredFiles);
        }
    };
    
    const containerClasses = `
        ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-white'}
        border-2 border-dashed rounded-lg p-6
        transition-colors duration-200 ease-in-out
        ${className}
    `;
    
    return (
        <div
            ref={containerRef}
            className={containerClasses}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            {...props}
        >
            {children}
        </div>
    );
};

// Export all components to window object
window.ReactComponents = {
    Button,
    Input,
    Select,
    Checkbox,
    RadioGroup,
    Card,
    Tabs,
    Modal,
    Alert,
    Toast,
    Tooltip,
    Badge,
    Progress,
    Dropdown,
    DataTable,
    DragDropContainer,
};

// Create a React DOM root for rendering React components
window.reactRoot = ReactDOM.createRoot(document.getElementById('react-root'));

// Render a test component to verify React is working
window.renderReactComponent = (component) => {
    window.reactRoot.render(component);
};

console.log('React Components initialized successfully');
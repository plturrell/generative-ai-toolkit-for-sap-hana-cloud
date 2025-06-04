/**
 * Comprehensive error handling system for Developer Studio
 * 
 * This module provides error handling utilities, error types, and reporting
 * mechanisms for the Developer Studio application.
 */

/**
 * Error types used throughout the application
 */
const ErrorTypes = {
    // API related errors
    API_CONNECTION: 'api_connection',
    API_TIMEOUT: 'api_timeout',
    API_AUTHENTICATION: 'api_authentication',
    API_PERMISSION: 'api_permission',
    API_RATE_LIMIT: 'api_rate_limit',
    API_SERVER: 'api_server',
    
    // Flow related errors
    FLOW_VALIDATION: 'flow_validation',
    FLOW_CYCLE: 'flow_cycle',
    FLOW_INCOMPATIBLE: 'flow_incompatible',
    FLOW_MISSING_PROPERTY: 'flow_missing_property',
    
    // Code generation and execution errors
    CODE_GENERATION: 'code_generation',
    CODE_EXECUTION: 'code_execution',
    CODE_SYNTAX: 'code_syntax',
    
    // Database errors
    DB_CONNECTION: 'db_connection',
    DB_QUERY: 'db_query',
    
    // User input errors
    VALIDATION: 'validation',
    
    // System errors
    SYSTEM: 'system',
    UNKNOWN: 'unknown'
};

/**
 * Application error class for standardized error handling
 */
class AppError extends Error {
    /**
     * Create a new application error
     * 
     * @param {string} message - Human readable error message
     * @param {string} type - Error type from ErrorTypes
     * @param {Object} [details={}] - Additional details about the error
     * @param {Error} [originalError=null] - Original error that caused this error
     */
    constructor(message, type, details = {}, originalError = null) {
        super(message);
        this.name = 'AppError';
        this.type = type;
        this.details = details;
        this.originalError = originalError;
        this.timestamp = new Date();
        
        // Capture stack trace
        if (Error.captureStackTrace) {
            Error.captureStackTrace(this, AppError);
        }
    }
    
    /**
     * Get a user-friendly message for this error
     * 
     * @returns {string} User-friendly error message
     */
    getUserMessage() {
        // Return custom user message based on error type
        switch(this.type) {
            case ErrorTypes.API_CONNECTION:
                return 'Unable to connect to the server. Please check your internet connection and try again.';
                
            case ErrorTypes.API_TIMEOUT:
                return 'The server took too long to respond. Please try again later.';
                
            case ErrorTypes.API_AUTHENTICATION:
                return 'Authentication failed. Please sign in again.';
                
            case ErrorTypes.API_PERMISSION:
                return 'You do not have permission to perform this action.';
                
            case ErrorTypes.API_RATE_LIMIT:
                return 'You have exceeded the rate limit. Please try again later.';
                
            case ErrorTypes.API_SERVER:
                return 'The server encountered an error. Please try again later.';
                
            case ErrorTypes.FLOW_VALIDATION:
                return 'Your flow contains validation errors. Please check the highlighted nodes.';
                
            case ErrorTypes.FLOW_CYCLE:
                return 'Your flow contains a cycle. Flows must be directed acyclic graphs.';
                
            case ErrorTypes.FLOW_INCOMPATIBLE:
                return 'Two connected nodes have incompatible types. Please check your connections.';
                
            case ErrorTypes.FLOW_MISSING_PROPERTY:
                return 'One or more nodes are missing required properties. Please check the highlighted nodes.';
                
            case ErrorTypes.CODE_GENERATION:
                return 'Failed to generate code from your flow. Please check your flow configuration.';
                
            case ErrorTypes.CODE_EXECUTION:
                return 'Failed to execute the generated code. Please check the error message for details.';
                
            case ErrorTypes.CODE_SYNTAX:
                return 'The code contains syntax errors. Please check the highlighted lines.';
                
            case ErrorTypes.DB_CONNECTION:
                return 'Failed to connect to the database. Please check your connection settings.';
                
            case ErrorTypes.DB_QUERY:
                return 'Failed to execute the database query. Please check your query syntax.';
                
            case ErrorTypes.VALIDATION:
                return 'Please check your input and try again.';
                
            case ErrorTypes.SYSTEM:
                return 'A system error occurred. Please try again later.';
                
            case ErrorTypes.UNKNOWN:
            default:
                return this.message || 'An unexpected error occurred. Please try again later.';
        }
    }
    
    /**
     * Get a detailed description of this error for logging
     * 
     * @returns {Object} Detailed error information
     */
    getDetails() {
        return {
            message: this.message,
            type: this.type,
            details: this.details,
            timestamp: this.timestamp,
            stack: this.stack,
            originalError: this.originalError ? {
                message: this.originalError.message,
                name: this.originalError.name,
                stack: this.originalError.stack
            } : null
        };
    }
}

/**
 * Error handler for API calls
 * 
 * @param {Response} response - Fetch API response
 * @returns {Promise<any>} - Parsed response data or throws an AppError
 */
async function handleApiResponse(response) {
    // If response is ok, return the data
    if (response.ok) {
        return await response.json();
    }
    
    // Otherwise, create an appropriate error
    let errorType = ErrorTypes.API_SERVER;
    let errorDetails = {};
    let errorMessage = 'An error occurred while communicating with the server';
    
    // Try to parse the error response
    try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorMessage;
        errorDetails = errorData;
    } catch (e) {
        // Failed to parse JSON, use response text
        errorMessage = await response.text();
    }
    
    // Determine error type based on status code
    switch (response.status) {
        case 401:
            errorType = ErrorTypes.API_AUTHENTICATION;
            break;
        case 403:
            errorType = ErrorTypes.API_PERMISSION;
            break;
        case 404:
            errorType = ErrorTypes.API_SERVER;
            errorMessage = 'The requested resource was not found';
            break;
        case 408:
        case 504:
            errorType = ErrorTypes.API_TIMEOUT;
            break;
        case 429:
            errorType = ErrorTypes.API_RATE_LIMIT;
            break;
        default:
            if (response.status >= 500) {
                errorType = ErrorTypes.API_SERVER;
            }
            break;
    }
    
    throw new AppError(errorMessage, errorType, {
        status: response.status,
        url: response.url,
        ...errorDetails
    });
}

/**
 * Retry a function with exponential backoff
 * 
 * @param {Function} fn - Function to retry (should return a Promise)
 * @param {Object} options - Retry options
 * @param {number} [options.maxRetries=3] - Maximum number of retries
 * @param {number} [options.initialDelay=1000] - Initial delay in milliseconds
 * @param {number} [options.maxDelay=10000] - Maximum delay in milliseconds
 * @param {Function} [options.shouldRetry] - Function to determine if retry should be attempted
 * @returns {Promise<any>} - Result of the function or throws the last error
 */
async function retryWithBackoff(fn, options = {}) {
    const {
        maxRetries = 3,
        initialDelay = 1000,
        maxDelay = 10000,
        shouldRetry = (error) => {
            // By default, retry on connection errors and server errors
            const retryableTypes = [
                ErrorTypes.API_CONNECTION,
                ErrorTypes.API_TIMEOUT,
                ErrorTypes.API_SERVER
            ];
            
            if (error instanceof AppError) {
                return retryableTypes.includes(error.type);
            }
            
            return false;
        }
    } = options;
    
    let lastError;
    let delay = initialDelay;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            
            // If we've used all retries or shouldn't retry, throw the error
            if (attempt >= maxRetries || !shouldRetry(error)) {
                throw error;
            }
            
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, delay));
            
            // Increase delay for next attempt (exponential backoff)
            delay = Math.min(delay * 2, maxDelay);
        }
    }
    
    // This should never happen but just in case
    throw lastError;
}

/**
 * Report an error to the error reporting service
 * 
 * @param {Error|AppError} error - Error to report
 * @param {Object} [context={}] - Additional context information
 */
function reportError(error, context = {}) {
    // In a production environment, this would send the error to a reporting service
    // For now, we'll just log it to the console
    const errorDetails = error instanceof AppError ? error.getDetails() : {
        message: error.message,
        name: error.name,
        stack: error.stack,
        timestamp: new Date()
    };
    
    console.error('Error Report:', {
        ...errorDetails,
        context
    });
    
    // In the future, this would send the error to a service like Sentry
    // Example:
    // Sentry.captureException(error, { extra: { ...context } });
}

/**
 * Display an error to the user
 * 
 * @param {Error|AppError} error - Error to display
 */
function displayError(error) {
    const message = error instanceof AppError ? error.getUserMessage() : error.message;
    
    // Use the toast notification system
    if (typeof showToast === 'function') {
        showToast(message, 'error');
    } else {
        // Fallback if toast is not available
        alert(message);
    }
}

/**
 * Creates an enhanced version of fetch with error handling, retries, and timeouts
 * 
 * @param {string} baseUrl - Base URL for all requests
 * @returns {Function} Enhanced fetch function
 */
function createApiClient(baseUrl) {
    return async function apiClient(endpoint, options = {}) {
        const {
            method = 'GET',
            data = null,
            headers = {},
            timeout = 30000,
            retry = true,
            maxRetries = 3,
            credentials = 'same-origin'
        } = options;
        
        // Prepare URL
        const url = endpoint.startsWith('http') ? endpoint : `${baseUrl}${endpoint}`;
        
        // Prepare headers
        const defaultHeaders = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
        
        // Prepare request
        const requestOptions = {
            method,
            headers: { ...defaultHeaders, ...headers },
            credentials
        };
        
        // Add body if data is provided
        if (data !== null) {
            requestOptions.body = JSON.stringify(data);
        }
        
        // Create a timeout promise
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
                reject(new AppError('Request timed out', ErrorTypes.API_TIMEOUT, { url }));
            }, timeout);
        });
        
        // Function to execute the fetch
        const executeFetch = async () => {
            try {
                // Race between fetch and timeout
                const response = await Promise.race([
                    fetch(url, requestOptions),
                    timeoutPromise
                ]);
                
                return await handleApiResponse(response);
            } catch (error) {
                // Convert fetch errors to AppErrors
                if (!(error instanceof AppError)) {
                    // Network errors
                    if (error.name === 'TypeError' && error.message.includes('Network')) {
                        throw new AppError('Network connection failed', ErrorTypes.API_CONNECTION, { url }, error);
                    }
                    
                    // Unknown errors
                    throw new AppError(error.message, ErrorTypes.UNKNOWN, { url }, error);
                }
                
                throw error;
            }
        };
        
        // Execute the fetch with or without retry
        try {
            if (retry) {
                return await retryWithBackoff(executeFetch, { maxRetries });
            } else {
                return await executeFetch();
            }
        } catch (error) {
            // Report the error
            reportError(error, { url, method });
            
            // Rethrow
            throw error;
        }
    };
}

/**
 * Create error boundary for React components
 * 
 * @param {string} fallbackMessage - Message to display when an error occurs
 * @returns {Function} Error boundary component
 */
function createErrorBoundary(fallbackMessage = 'Something went wrong') {
    // This is a placeholder for the React error boundary
    // In a real implementation, this would use React's error boundary functionality
    
    console.log('Error boundary created with fallback message:', fallbackMessage);
    
    return function ErrorBoundary(props) {
        // This would be implemented with React
        return props.children;
    };
}

// Export all utilities
window.ErrorHandler = {
    ErrorTypes,
    AppError,
    handleApiResponse,
    retryWithBackoff,
    reportError,
    displayError,
    createApiClient,
    createErrorBoundary
};
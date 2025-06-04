/**
 * API Client Module
 * 
 * Handles all API communication with standardized error handling and retries.
 */

class APIClient {
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || '';
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            ...(options.headers || {})
        };
        this.timeout = options.timeout || 30000; // 30 seconds default
        this.maxRetries = options.maxRetries || 3;
        this.retryDelay = options.retryDelay || 1000; // 1 second default
        this.eventListeners = {};
    }

    /**
     * Send a request to the API
     * 
     * @param {String} url - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<any>} - API response
     */
    async request(url, options = {}) {
        const requestUrl = this.baseUrl ? `${this.baseUrl}${url}` : url;
        const headers = {
            ...this.defaultHeaders,
            ...(options.headers || {})
        };
        
        const requestOptions = {
            method: options.method || 'GET',
            headers,
            ...options,
        };
        
        // Add body if provided
        if (options.body !== undefined) {
            requestOptions.body = typeof options.body === 'string' 
                ? options.body 
                : JSON.stringify(options.body);
        }
        
        // Create AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        requestOptions.signal = controller.signal;
        
        // Trigger requestStart event
        this.triggerEvent('requestStart', {
            url: requestUrl,
            method: requestOptions.method,
            headers: requestOptions.headers,
            body: options.body
        });
        
        let retries = 0;
        let lastError = null;
        
        while (retries <= this.maxRetries) {
            try {
                const response = await fetch(requestUrl, requestOptions);
                
                // Clear timeout
                clearTimeout(timeoutId);
                
                // Get response body
                const data = await this.parseResponseData(response);
                
                // Check if response is ok
                if (!response.ok) {
                    const error = new AppError(
                        data.detail || data.error || `Request failed with status ${response.status}`,
                        'API_ERROR',
                        {
                            status: response.status,
                            statusText: response.statusText,
                            data,
                            url: requestUrl,
                            method: requestOptions.method
                        }
                    );
                    
                    // Trigger requestError event
                    this.triggerEvent('requestError', {
                        url: requestUrl,
                        method: requestOptions.method,
                        status: response.status,
                        error: error
                    });
                    
                    throw error;
                }
                
                // Trigger requestSuccess event
                this.triggerEvent('requestSuccess', {
                    url: requestUrl,
                    method: requestOptions.method,
                    status: response.status,
                    data
                });
                
                return data;
            } catch (error) {
                lastError = error;
                
                // Clear timeout
                clearTimeout(timeoutId);
                
                // Don't retry if aborted or max retries reached
                if (error.name === 'AbortError') {
                    const timeoutError = new AppError(
                        `Request timed out after ${this.timeout}ms`,
                        'REQUEST_TIMEOUT',
                        { originalError: error, url: requestUrl, method: requestOptions.method }
                    );
                    
                    // Trigger requestError event
                    this.triggerEvent('requestError', {
                        url: requestUrl,
                        method: requestOptions.method,
                        error: timeoutError
                    });
                    
                    throw timeoutError;
                }
                
                // If max retries reached, throw the error
                if (retries >= this.maxRetries) {
                    const retriesError = new AppError(
                        `Request failed after ${this.maxRetries} retries`,
                        'MAX_RETRIES_EXCEEDED',
                        { originalError: error, url: requestUrl, method: requestOptions.method }
                    );
                    
                    // Trigger requestError event
                    this.triggerEvent('requestError', {
                        url: requestUrl,
                        method: requestOptions.method,
                        error: retriesError
                    });
                    
                    throw retriesError;
                }
                
                // Trigger retryRequest event
                this.triggerEvent('retryRequest', {
                    url: requestUrl,
                    method: requestOptions.method,
                    error,
                    retryCount: retries + 1
                });
                
                // Wait before retrying (with exponential backoff)
                const delay = this.retryDelay * Math.pow(2, retries);
                await new Promise(resolve => setTimeout(resolve, delay));
                
                retries++;
            }
        }
        
        // This should never happen but just in case
        throw lastError;
    }

    /**
     * Parse response data based on content type
     * 
     * @param {Response} response - Fetch Response object
     * @returns {Promise<any>} - Parsed response data
     */
    async parseResponseData(response) {
        const contentType = response.headers.get('Content-Type') || '';
        
        if (contentType.includes('application/json')) {
            return await response.json();
        } else if (contentType.includes('text/plain') || contentType.includes('text/html')) {
            return await response.text();
        } else {
            // For binary data or other types, return as is
            return await response.blob();
        }
    }

    /**
     * Send a GET request
     * 
     * @param {String} url - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<any>} - API response
     */
    async get(url, options = {}) {
        return this.request(url, { ...options, method: 'GET' });
    }

    /**
     * Send a POST request
     * 
     * @param {String} url - API endpoint
     * @param {Object} body - Request body
     * @param {Object} options - Request options
     * @returns {Promise<any>} - API response
     */
    async post(url, body = {}, options = {}) {
        return this.request(url, { ...options, method: 'POST', body });
    }

    /**
     * Send a PUT request
     * 
     * @param {String} url - API endpoint
     * @param {Object} body - Request body
     * @param {Object} options - Request options
     * @returns {Promise<any>} - API response
     */
    async put(url, body = {}, options = {}) {
        return this.request(url, { ...options, method: 'PUT', body });
    }

    /**
     * Send a DELETE request
     * 
     * @param {String} url - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<any>} - API response
     */
    async delete(url, options = {}) {
        return this.request(url, { ...options, method: 'DELETE' });
    }

    /**
     * Add event listener
     * 
     * @param {String} event - Event name
     * @param {Function} callback - Event callback
     */
    addEventListener(event, callback) {
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        
        this.eventListeners[event].push(callback);
    }

    /**
     * Remove event listener
     * 
     * @param {String} event - Event name
     * @param {Function} callback - Event callback
     */
    removeEventListener(event, callback) {
        if (!this.eventListeners[event]) return;
        
        this.eventListeners[event] = this.eventListeners[event].filter(cb => cb !== callback);
    }

    /**
     * Trigger an event
     * 
     * @param {String} event - Event name
     * @param {*} data - Event data
     */
    triggerEvent(event, data) {
        if (!this.eventListeners[event]) return;
        
        for (const callback of this.eventListeners[event]) {
            callback(data);
        }
    }
}

// Export the class
window.APIClient = APIClient;
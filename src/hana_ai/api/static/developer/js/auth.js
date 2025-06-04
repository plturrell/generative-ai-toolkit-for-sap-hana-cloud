/**
 * Authentication and security module for Developer Studio
 * 
 * This module provides secure authentication, token management, and
 * API security utilities for the Developer Studio application.
 */

/**
 * Token storage key in local storage
 */
const TOKEN_STORAGE_KEY = 'dev_studio_auth_token';

/**
 * Token refresh interval in milliseconds (10 minutes)
 */
const TOKEN_REFRESH_INTERVAL = 10 * 60 * 1000;

/**
 * Authentication state
 */
const authState = {
    token: null,
    refreshToken: null,
    expiresAt: null,
    user: null,
    isAuthenticated: false,
    isAdmin: false,
    refreshInterval: null
};

/**
 * Event listeners for auth events
 */
const listeners = {
    login: [],
    logout: [],
    tokenRefresh: []
};

/**
 * Initialize authentication
 * 
 * @returns {Promise<boolean>} Whether authentication was initialized successfully
 */
async function initAuth() {
    // Try to load token from storage
    loadTokenFromStorage();
    
    // If we have a token, validate it
    if (authState.token) {
        try {
            await validateToken();
            startTokenRefresh();
            return true;
        } catch (error) {
            // Token is invalid, clear it
            clearAuthState();
            return false;
        }
    }
    
    return false;
}

/**
 * Load token from storage
 */
function loadTokenFromStorage() {
    try {
        const storedAuth = localStorage.getItem(TOKEN_STORAGE_KEY);
        
        if (storedAuth) {
            const parsedAuth = JSON.parse(storedAuth);
            
            // Verify expiration
            if (parsedAuth.expiresAt && new Date(parsedAuth.expiresAt) > new Date()) {
                // Token is still valid
                authState.token = parsedAuth.token;
                authState.refreshToken = parsedAuth.refreshToken;
                authState.expiresAt = parsedAuth.expiresAt;
                authState.user = parsedAuth.user;
                authState.isAuthenticated = true;
                authState.isAdmin = parsedAuth.isAdmin || false;
            } else {
                // Token has expired, clear it
                clearAuthState();
            }
        }
    } catch (error) {
        console.error('Failed to load auth from storage:', error);
        clearAuthState();
    }
}

/**
 * Save token to storage
 */
function saveTokenToStorage() {
    try {
        localStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify({
            token: authState.token,
            refreshToken: authState.refreshToken,
            expiresAt: authState.expiresAt,
            user: authState.user,
            isAdmin: authState.isAdmin
        }));
    } catch (error) {
        console.error('Failed to save auth to storage:', error);
    }
}

/**
 * Clear authentication state
 */
function clearAuthState() {
    authState.token = null;
    authState.refreshToken = null;
    authState.expiresAt = null;
    authState.user = null;
    authState.isAuthenticated = false;
    authState.isAdmin = false;
    
    try {
        localStorage.removeItem(TOKEN_STORAGE_KEY);
    } catch (error) {
        console.error('Failed to clear auth from storage:', error);
    }
    
    stopTokenRefresh();
}

/**
 * Validate token with server
 * 
 * @returns {Promise<boolean>} Whether token is valid
 */
async function validateToken() {
    if (!authState.token) {
        return false;
    }
    
    try {
        // In a real implementation, this would validate the token with the server
        // For now, we'll just check if it hasn't expired
        if (authState.expiresAt && new Date(authState.expiresAt) > new Date()) {
            return true;
        }
        
        throw new Error('Token has expired');
    } catch (error) {
        console.error('Token validation failed:', error);
        clearAuthState();
        return false;
    }
}

/**
 * Start token refresh interval
 */
function startTokenRefresh() {
    // Clear any existing interval
    stopTokenRefresh();
    
    // Start new interval
    authState.refreshInterval = setInterval(async () => {
        try {
            await refreshToken();
            notifyListeners('tokenRefresh');
        } catch (error) {
            console.error('Token refresh failed:', error);
            clearAuthState();
            notifyListeners('logout', { reason: 'refresh_failed' });
        }
    }, TOKEN_REFRESH_INTERVAL);
}

/**
 * Stop token refresh interval
 */
function stopTokenRefresh() {
    if (authState.refreshInterval) {
        clearInterval(authState.refreshInterval);
        authState.refreshInterval = null;
    }
}

/**
 * Refresh token
 * 
 * @returns {Promise<boolean>} Whether token was refreshed successfully
 */
async function refreshToken() {
    if (!authState.refreshToken) {
        return false;
    }
    
    try {
        // In a real implementation, this would call the token refresh endpoint
        // For now, we'll just extend the expiration
        authState.expiresAt = new Date(Date.now() + TOKEN_REFRESH_INTERVAL * 2).toISOString();
        saveTokenToStorage();
        return true;
    } catch (error) {
        console.error('Token refresh failed:', error);
        clearAuthState();
        return false;
    }
}

/**
 * Log in with username and password
 * 
 * @param {string} username - Username
 * @param {string} password - Password
 * @returns {Promise<Object>} User information
 */
async function login(username, password) {
    try {
        // In a real implementation, this would call the login endpoint
        // For now, we'll just simulate a successful login
        
        // Validate credentials (this is just for demonstration)
        if (username !== 'admin' && username !== 'user') {
            throw new Error('Invalid username or password');
        }
        
        // Create a token with 1 hour expiration
        const expiresAt = new Date(Date.now() + 60 * 60 * 1000).toISOString();
        
        // Set auth state
        authState.token = `mock-token-${Date.now()}`;
        authState.refreshToken = `mock-refresh-token-${Date.now()}`;
        authState.expiresAt = expiresAt;
        authState.user = {
            id: username === 'admin' ? '1' : '2',
            username,
            name: username === 'admin' ? 'Administrator' : 'Regular User',
            email: `${username}@example.com`
        };
        authState.isAuthenticated = true;
        authState.isAdmin = username === 'admin';
        
        // Save to storage
        saveTokenToStorage();
        
        // Start token refresh
        startTokenRefresh();
        
        // Notify listeners
        notifyListeners('login', authState.user);
        
        return authState.user;
    } catch (error) {
        // Clear auth state
        clearAuthState();
        
        // Rethrow error
        throw error;
    }
}

/**
 * Log out
 */
async function logout() {
    // In a real implementation, this would call the logout endpoint
    
    // Clear auth state
    clearAuthState();
    
    // Notify listeners
    notifyListeners('logout', { reason: 'user_logout' });
}

/**
 * Get current user
 * 
 * @returns {Object|null} Current user or null if not authenticated
 */
function getCurrentUser() {
    return authState.user;
}

/**
 * Check if user is authenticated
 * 
 * @returns {boolean} Whether user is authenticated
 */
function isAuthenticated() {
    return authState.isAuthenticated;
}

/**
 * Check if user is an admin
 * 
 * @returns {boolean} Whether user is an admin
 */
function isAdmin() {
    return authState.isAdmin;
}

/**
 * Get auth headers for API requests
 * 
 * @returns {Object} Headers object with authorization
 */
function getAuthHeaders() {
    if (!authState.token) {
        return {};
    }
    
    return {
        'Authorization': `Bearer ${authState.token}`
    };
}

/**
 * Add event listener
 * 
 * @param {string} event - Event name (login, logout, tokenRefresh)
 * @param {Function} callback - Callback function
 */
function addEventListener(event, callback) {
    if (!listeners[event]) {
        listeners[event] = [];
    }
    
    listeners[event].push(callback);
}

/**
 * Remove event listener
 * 
 * @param {string} event - Event name (login, logout, tokenRefresh)
 * @param {Function} callback - Callback function
 */
function removeEventListener(event, callback) {
    if (!listeners[event]) {
        return;
    }
    
    listeners[event] = listeners[event].filter(cb => cb !== callback);
}

/**
 * Notify listeners of an event
 * 
 * @param {string} event - Event name
 * @param {*} data - Event data
 */
function notifyListeners(event, data) {
    if (!listeners[event]) {
        return;
    }
    
    listeners[event].forEach(callback => {
        try {
            callback(data);
        } catch (error) {
            console.error(`Error in ${event} listener:`, error);
        }
    });
}

/**
 * Generate a secure random token
 * 
 * @param {number} [length=32] - Token length
 * @returns {string} Random token
 */
function generateSecureToken(length = 32) {
    const array = new Uint8Array(length);
    window.crypto.getRandomValues(array);
    return Array.from(array, byte => ('0' + (byte & 0xFF).toString(16)).slice(-2)).join('');
}

/**
 * Generate a CSRF token for forms
 * 
 * @returns {string} CSRF token
 */
function generateCsrfToken() {
    const token = generateSecureToken();
    
    try {
        sessionStorage.setItem('csrf_token', token);
    } catch (error) {
        console.error('Failed to save CSRF token to session storage:', error);
    }
    
    return token;
}

/**
 * Validate a CSRF token
 * 
 * @param {string} token - Token to validate
 * @returns {boolean} Whether token is valid
 */
function validateCsrfToken(token) {
    try {
        const storedToken = sessionStorage.getItem('csrf_token');
        return storedToken === token;
    } catch (error) {
        console.error('Failed to validate CSRF token:', error);
        return false;
    }
}

/**
 * Create a login form with CSRF protection
 * 
 * @param {string} containerId - ID of container element
 * @param {Function} [onSuccess] - Callback function on successful login
 */
function createLoginForm(containerId, onSuccess) {
    const container = document.getElementById(containerId);
    
    if (!container) {
        console.error(`Container with ID ${containerId} not found`);
        return;
    }
    
    // Generate CSRF token
    const csrfToken = generateCsrfToken();
    
    // Create form HTML
    container.innerHTML = `
        <form id="login-form" class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
            <div class="mb-4">
                <label class="block text-gray-700 text-sm font-bold mb-2" for="username">
                    Username
                </label>
                <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="username" type="text" placeholder="Username">
            </div>
            <div class="mb-6">
                <label class="block text-gray-700 text-sm font-bold mb-2" for="password">
                    Password
                </label>
                <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline" id="password" type="password" placeholder="******************">
            </div>
            <div class="flex items-center justify-between">
                <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline" type="submit">
                    Sign In
                </button>
            </div>
            <input type="hidden" name="csrf_token" value="${csrfToken}">
        </form>
    `;
    
    // Add event listener
    const form = document.getElementById('login-form');
    
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const formCsrfToken = form.querySelector('input[name="csrf_token"]').value;
        
        // Validate CSRF token
        if (!validateCsrfToken(formCsrfToken)) {
            alert('Security validation failed. Please refresh the page and try again.');
            return;
        }
        
        try {
            const user = await login(username, password);
            
            if (onSuccess) {
                onSuccess(user);
            }
        } catch (error) {
            alert(error.message || 'Login failed. Please try again.');
        }
    });
}

// Export all utilities
window.Auth = {
    initAuth,
    login,
    logout,
    getCurrentUser,
    isAuthenticated,
    isAdmin,
    getAuthHeaders,
    addEventListener,
    removeEventListener,
    createLoginForm
};
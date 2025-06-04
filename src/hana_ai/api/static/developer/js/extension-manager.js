/**
 * Extension Manager Module
 * 
 * Provides extension marketplace and extension management capabilities.
 */

class ExtensionManager {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.extensions = [];
        this.installedExtensions = [];
        this.isInitialized = false;
        this.eventListeners = {};
        this.extensionPoints = {
            'editor': [],       // Extensions for the code editor
            'flow': [],         // Extensions for the flow builder
            'tool': [],         // New tools for the toolkit
            'language': [],     // Language support extensions
            'theme': [],        // UI themes
            'formatter': []     // Code formatters
        };
    }

    /**
     * Initialize the Extension Manager
     * 
     * @returns {Promise<boolean>} - True if initialized successfully
     */
    async initialize() {
        try {
            // Fetch available extensions
            await this.fetchExtensions();
            
            // Fetch installed extensions
            await this.fetchInstalledExtensions();
            
            // Initialize extensions
            await this.initializeExtensions();
            
            // Create extension UI
            this.createExtensionUI();
            
            // Set initialized flag
            this.isInitialized = true;
            
            // Trigger initialized event
            this.triggerEvent('initialized', {
                extensions: this.extensions,
                installedExtensions: this.installedExtensions
            });
            
            return true;
        } catch (error) {
            console.error('Failed to initialize Extension Manager:', error);
            throw new AppError(
                'Failed to initialize Extension Manager', 
                'EXTENSION_INITIALIZATION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Fetch available extensions from the marketplace
     */
    async fetchExtensions() {
        try {
            // Call API to get available extensions
            const response = await this.apiClient.get('/api/v1/developer/extensions');
            
            // Store extensions
            this.extensions = response.extensions || [];
            
            // Trigger extensions fetched event
            this.triggerEvent('extensionsFetched', this.extensions);
            
            return this.extensions;
        } catch (error) {
            console.error('Failed to fetch extensions:', error);
            throw new AppError(
                'Failed to fetch extensions', 
                'EXTENSION_FETCH_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Fetch installed extensions
     */
    async fetchInstalledExtensions() {
        try {
            // Call API to get installed extensions
            const response = await this.apiClient.get('/api/v1/developer/extensions/installed');
            
            // Store installed extensions
            this.installedExtensions = response.extensions || [];
            
            // Trigger installed extensions fetched event
            this.triggerEvent('installedExtensionsFetched', this.installedExtensions);
            
            return this.installedExtensions;
        } catch (error) {
            console.error('Failed to fetch installed extensions:', error);
            throw new AppError(
                'Failed to fetch installed extensions', 
                'INSTALLED_EXTENSIONS_FETCH_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Initialize installed extensions
     */
    async initializeExtensions() {
        try {
            // Reset extension points
            for (const key in this.extensionPoints) {
                this.extensionPoints[key] = [];
            }
            
            // Load extension scripts
            for (const extension of this.installedExtensions) {
                if (extension.enabled) {
                    await this.loadExtension(extension);
                }
            }
            
            // Trigger extensions initialized event
            this.triggerEvent('extensionsInitialized', this.extensionPoints);
        } catch (error) {
            console.error('Failed to initialize extensions:', error);
            throw new AppError(
                'Failed to initialize extensions', 
                'EXTENSION_INITIALIZATION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Load an extension
     * 
     * @param {Object} extension - Extension to load
     */
    async loadExtension(extension) {
        try {
            // Check if extension is already loaded
            if (window[`extension_${extension.id}`]) {
                console.log(`Extension ${extension.id} already loaded`);
                return;
            }
            
            // Call API to get extension script
            const response = await this.apiClient.get(`/api/v1/developer/extensions/${extension.id}/script`);
            
            // Create script element
            const script = document.createElement('script');
            script.id = `extension-${extension.id}`;
            script.textContent = response.script;
            
            // Add load and error event listeners
            script.addEventListener('load', () => {
                console.log(`Extension ${extension.id} loaded successfully`);
                
                // Register extension
                this.registerExtension(extension);
                
                // Trigger extension loaded event
                this.triggerEvent('extensionLoaded', extension);
            });
            
            script.addEventListener('error', (error) => {
                console.error(`Failed to load extension ${extension.id}:`, error);
                
                // Trigger extension load failed event
                this.triggerEvent('extensionLoadFailed', {
                    extension,
                    error
                });
            });
            
            // Append script to document
            document.head.appendChild(script);
        } catch (error) {
            console.error(`Failed to load extension ${extension.id}:`, error);
            throw new AppError(
                'Failed to load extension', 
                'EXTENSION_LOAD_FAILED',
                { originalError: error, extension }
            );
        }
    }

    /**
     * Register an extension
     * 
     * @param {Object} extension - Extension to register
     */
    registerExtension(extension) {
        // Get extension object from window
        const extObj = window[`extension_${extension.id}`];
        
        if (!extObj) {
            console.error(`Extension ${extension.id} not found in window`);
            return;
        }
        
        // Register extension for each extension point
        for (const point in extObj) {
            if (this.extensionPoints[point]) {
                this.extensionPoints[point].push({
                    id: extension.id,
                    extension: extObj[point]
                });
            }
        }
        
        // Call extension's initialize method if it exists
        if (extObj.initialize && typeof extObj.initialize === 'function') {
            extObj.initialize();
        }
    }

    /**
     * Create extension UI
     */
    createExtensionUI() {
        // Create extension panel
        const extensionPanel = document.createElement('div');
        extensionPanel.id = 'extension-panel';
        extensionPanel.className = 'extension-panel';
        
        // Create extension panel header
        const extensionHeader = document.createElement('div');
        extensionHeader.className = 'extension-panel-header';
        extensionHeader.innerHTML = `
            <div class="extension-panel-title">
                <i class="fas fa-puzzle-piece"></i>
                Extensions
            </div>
            <div class="extension-panel-actions">
                <button class="extension-action refresh" title="Refresh">
                    <i class="fas fa-sync-alt"></i>
                </button>
                <button class="extension-action close" title="Close Extension Panel">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Create extension panel content with tabs
        const extensionContent = document.createElement('div');
        extensionContent.className = 'extension-panel-content';
        extensionContent.innerHTML = `
            <div class="extension-tabs">
                <div class="extension-tab active" data-tab="installed">Installed</div>
                <div class="extension-tab" data-tab="marketplace">Marketplace</div>
            </div>
            <div class="extension-content">
                <div class="extension-tab-content installed active">
                    <div class="extension-search">
                        <input type="text" class="extension-search-input" placeholder="Search installed extensions...">
                    </div>
                    <div class="extension-list installed-extensions">
                        <div class="extension-empty-message">No extensions installed</div>
                    </div>
                </div>
                <div class="extension-tab-content marketplace">
                    <div class="extension-search">
                        <input type="text" class="extension-search-input" placeholder="Search marketplace...">
                    </div>
                    <div class="extension-list marketplace-extensions">
                        <div class="extension-empty-message">Loading marketplace extensions...</div>
                    </div>
                </div>
            </div>
        `;
        
        // Add header and content to panel
        extensionPanel.appendChild(extensionHeader);
        extensionPanel.appendChild(extensionContent);
        
        // Add panel to the workspace
        const workspace = document.querySelector('.developer-workspace');
        if (workspace) {
            workspace.appendChild(extensionPanel);
        }
        
        // Add event listeners
        this.addExtensionPanelEventListeners();
        
        // Update extension UI with initial data
        this.updateExtensionUI();
    }

    /**
     * Add event listeners to extension panel
     */
    addExtensionPanelEventListeners() {
        // Panel header actions
        const refreshButton = document.querySelector('.extension-panel-header .extension-action.refresh');
        const closeButton = document.querySelector('.extension-panel-header .extension-action.close');
        
        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                this.refreshExtensions();
            });
        }
        
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                this.hideExtensionPanel();
            });
        }
        
        // Tab switching
        const extensionTabs = document.querySelectorAll('.extension-tab');
        const tabContents = document.querySelectorAll('.extension-tab-content');
        
        extensionTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                extensionTabs.forEach(t => t.classList.remove('active'));
                
                // Add active class to clicked tab
                tab.classList.add('active');
                
                // Show corresponding tab content
                const tabName = tab.getAttribute('data-tab');
                tabContents.forEach(content => {
                    content.classList.remove('active');
                    if (content.classList.contains(tabName)) {
                        content.classList.add('active');
                    }
                });
            });
        });
        
        // Search
        const searchInputs = document.querySelectorAll('.extension-search-input');
        searchInputs.forEach(input => {
            input.addEventListener('input', (e) => {
                const searchText = e.target.value.toLowerCase();
                const tabContent = e.target.closest('.extension-tab-content');
                const tabName = tabContent.classList.contains('installed') ? 'installed' : 'marketplace';
                
                this.filterExtensions(searchText, tabName);
            });
        });
    }

    /**
     * Show extension panel
     */
    showExtensionPanel() {
        const extensionPanel = document.getElementById('extension-panel');
        if (extensionPanel) {
            extensionPanel.style.display = 'flex';
        }
    }

    /**
     * Hide extension panel
     */
    hideExtensionPanel() {
        const extensionPanel = document.getElementById('extension-panel');
        if (extensionPanel) {
            extensionPanel.style.display = 'none';
        }
    }

    /**
     * Toggle extension panel
     */
    toggleExtensionPanel() {
        const extensionPanel = document.getElementById('extension-panel');
        if (extensionPanel) {
            if (extensionPanel.style.display === 'none') {
                this.showExtensionPanel();
            } else {
                this.hideExtensionPanel();
            }
        } else {
            this.createExtensionUI();
            this.showExtensionPanel();
        }
    }

    /**
     * Update extension UI with current data
     */
    updateExtensionUI() {
        // Update installed extensions
        this.updateInstalledExtensionsUI();
        
        // Update marketplace extensions
        this.updateMarketplaceExtensionsUI();
    }

    /**
     * Update installed extensions UI
     */
    updateInstalledExtensionsUI() {
        const installedExtensionsContainer = document.querySelector('.installed-extensions');
        if (!installedExtensionsContainer) return;
        
        // Clear container
        installedExtensionsContainer.innerHTML = '';
        
        // Check if there are installed extensions
        if (this.installedExtensions.length === 0) {
            installedExtensionsContainer.innerHTML = '<div class="extension-empty-message">No extensions installed</div>';
            return;
        }
        
        // Add installed extensions
        this.installedExtensions.forEach(extension => {
            const extensionItem = document.createElement('div');
            extensionItem.className = 'extension-item';
            extensionItem.innerHTML = `
                <div class="extension-item-header">
                    <div class="extension-icon">
                        <i class="${extension.icon || 'fas fa-puzzle-piece'}"></i>
                    </div>
                    <div class="extension-info">
                        <div class="extension-name">${extension.name}</div>
                        <div class="extension-version">v${extension.version}</div>
                    </div>
                    <div class="extension-actions">
                        <label class="extension-toggle">
                            <input type="checkbox" class="extension-toggle-checkbox" data-extension-id="${extension.id}" ${extension.enabled ? 'checked' : ''}>
                            <span class="extension-toggle-slider"></span>
                        </label>
                        <button class="extension-action uninstall" data-extension-id="${extension.id}" title="Uninstall">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="extension-item-content">
                    <div class="extension-description">${extension.description}</div>
                    <div class="extension-details">
                        <div class="extension-author">
                            <i class="fas fa-user"></i>
                            ${extension.author}
                        </div>
                        <div class="extension-type">
                            <i class="fas fa-tag"></i>
                            ${extension.type}
                        </div>
                    </div>
                </div>
            `;
            
            installedExtensionsContainer.appendChild(extensionItem);
        });
        
        // Add event listeners
        const toggleCheckboxes = installedExtensionsContainer.querySelectorAll('.extension-toggle-checkbox');
        toggleCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const extensionId = e.target.getAttribute('data-extension-id');
                const enabled = e.target.checked;
                
                this.toggleExtension(extensionId, enabled);
            });
        });
        
        const uninstallButtons = installedExtensionsContainer.querySelectorAll('.extension-action.uninstall');
        uninstallButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const extensionId = e.target.closest('button').getAttribute('data-extension-id');
                
                // Show confirmation dialog
                if (confirm('Are you sure you want to uninstall this extension?')) {
                    this.uninstallExtension(extensionId);
                }
            });
        });
    }

    /**
     * Update marketplace extensions UI
     */
    updateMarketplaceExtensionsUI() {
        const marketplaceExtensionsContainer = document.querySelector('.marketplace-extensions');
        if (!marketplaceExtensionsContainer) return;
        
        // Clear container
        marketplaceExtensionsContainer.innerHTML = '';
        
        // Check if there are marketplace extensions
        if (this.extensions.length === 0) {
            marketplaceExtensionsContainer.innerHTML = '<div class="extension-empty-message">No extensions found in marketplace</div>';
            return;
        }
        
        // Add marketplace extensions
        this.extensions.forEach(extension => {
            // Check if extension is already installed
            const isInstalled = this.installedExtensions.some(e => e.id === extension.id);
            
            const extensionItem = document.createElement('div');
            extensionItem.className = 'extension-item';
            extensionItem.innerHTML = `
                <div class="extension-item-header">
                    <div class="extension-icon">
                        <i class="${extension.icon || 'fas fa-puzzle-piece'}"></i>
                    </div>
                    <div class="extension-info">
                        <div class="extension-name">${extension.name}</div>
                        <div class="extension-version">v${extension.version}</div>
                    </div>
                    <div class="extension-actions">
                        <button class="extension-action ${isInstalled ? 'installed' : 'install'}" data-extension-id="${extension.id}" ${isInstalled ? 'disabled' : ''}>
                            <i class="fas ${isInstalled ? 'fa-check' : 'fa-download'}"></i>
                            ${isInstalled ? 'Installed' : 'Install'}
                        </button>
                    </div>
                </div>
                <div class="extension-item-content">
                    <div class="extension-description">${extension.description}</div>
                    <div class="extension-details">
                        <div class="extension-author">
                            <i class="fas fa-user"></i>
                            ${extension.author}
                        </div>
                        <div class="extension-type">
                            <i class="fas fa-tag"></i>
                            ${extension.type}
                        </div>
                        <div class="extension-rating">
                            <i class="fas fa-star"></i>
                            ${extension.rating || '0.0'} (${extension.downloads || '0'} downloads)
                        </div>
                    </div>
                </div>
            `;
            
            marketplaceExtensionsContainer.appendChild(extensionItem);
        });
        
        // Add event listeners
        const installButtons = marketplaceExtensionsContainer.querySelectorAll('.extension-action.install');
        installButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const extensionId = e.target.closest('button').getAttribute('data-extension-id');
                
                this.installExtension(extensionId);
            });
        });
    }

    /**
     * Filter extensions based on search text
     * 
     * @param {string} searchText - Text to search for
     * @param {string} tabName - Tab name ('installed' or 'marketplace')
     */
    filterExtensions(searchText, tabName) {
        const container = document.querySelector(`.${tabName}-extensions`);
        if (!container) return;
        
        const extensionItems = container.querySelectorAll('.extension-item');
        
        extensionItems.forEach(item => {
            const name = item.querySelector('.extension-name').textContent.toLowerCase();
            const description = item.querySelector('.extension-description').textContent.toLowerCase();
            const author = item.querySelector('.extension-author').textContent.toLowerCase();
            
            const matches = name.includes(searchText) || 
                           description.includes(searchText) || 
                           author.includes(searchText);
            
            item.style.display = matches ? 'block' : 'none';
        });
        
        // Show message if no matches
        const hasVisibleItems = Array.from(extensionItems).some(item => item.style.display !== 'none');
        
        // Remove existing empty message
        const existingEmptyMessage = container.querySelector('.extension-empty-message');
        if (existingEmptyMessage) {
            container.removeChild(existingEmptyMessage);
        }
        
        // Add empty message if no matches
        if (!hasVisibleItems) {
            const emptyMessage = document.createElement('div');
            emptyMessage.className = 'extension-empty-message';
            emptyMessage.textContent = 'No extensions match your search';
            container.appendChild(emptyMessage);
        }
    }

    /**
     * Refresh extensions
     */
    async refreshExtensions() {
        try {
            // Fetch available extensions
            await this.fetchExtensions();
            
            // Fetch installed extensions
            await this.fetchInstalledExtensions();
            
            // Update extension UI
            this.updateExtensionUI();
            
            // Show success notification
            showToast('Extensions refreshed successfully', 'success');
        } catch (error) {
            console.error('Failed to refresh extensions:', error);
            
            // Show error notification
            showToast(`Failed to refresh extensions: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to refresh extensions', 
                'EXTENSION_REFRESH_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Install an extension
     * 
     * @param {string} extensionId - ID of the extension to install
     */
    async installExtension(extensionId) {
        try {
            // Check if extension is already installed
            if (this.installedExtensions.some(e => e.id === extensionId)) {
                throw new Error('Extension already installed');
            }
            
            // Install extension
            await this.apiClient.post(`/api/v1/developer/extensions/${extensionId}/install`);
            
            // Refresh installed extensions
            await this.fetchInstalledExtensions();
            
            // Update extension UI
            this.updateExtensionUI();
            
            // Load extension
            const extension = this.installedExtensions.find(e => e.id === extensionId);
            if (extension) {
                await this.loadExtension(extension);
            }
            
            // Trigger extension installed event
            this.triggerEvent('extensionInstalled', extension);
            
            // Show success notification
            showToast(`Extension '${extension.name}' installed successfully`, 'success');
        } catch (error) {
            console.error(`Failed to install extension ${extensionId}:`, error);
            
            // Show error notification
            showToast(`Failed to install extension: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to install extension', 
                'EXTENSION_INSTALL_FAILED',
                { originalError: error, extensionId }
            );
        }
    }

    /**
     * Uninstall an extension
     * 
     * @param {string} extensionId - ID of the extension to uninstall
     */
    async uninstallExtension(extensionId) {
        try {
            // Check if extension is installed
            const extension = this.installedExtensions.find(e => e.id === extensionId);
            if (!extension) {
                throw new Error('Extension not installed');
            }
            
            // Uninstall extension
            await this.apiClient.post(`/api/v1/developer/extensions/${extensionId}/uninstall`);
            
            // Refresh installed extensions
            await this.fetchInstalledExtensions();
            
            // Update extension UI
            this.updateExtensionUI();
            
            // Remove extension script
            const script = document.getElementById(`extension-${extensionId}`);
            if (script) {
                document.head.removeChild(script);
            }
            
            // Cleanup extension
            if (window[`extension_${extensionId}`]) {
                // Call cleanup method if it exists
                const extObj = window[`extension_${extensionId}`];
                if (extObj.cleanup && typeof extObj.cleanup === 'function') {
                    extObj.cleanup();
                }
                
                // Remove extension from window
                delete window[`extension_${extensionId}`];
            }
            
            // Remove extension from extension points
            for (const point in this.extensionPoints) {
                this.extensionPoints[point] = this.extensionPoints[point].filter(e => e.id !== extensionId);
            }
            
            // Trigger extension uninstalled event
            this.triggerEvent('extensionUninstalled', extension);
            
            // Show success notification
            showToast(`Extension '${extension.name}' uninstalled successfully`, 'success');
        } catch (error) {
            console.error(`Failed to uninstall extension ${extensionId}:`, error);
            
            // Show error notification
            showToast(`Failed to uninstall extension: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to uninstall extension', 
                'EXTENSION_UNINSTALL_FAILED',
                { originalError: error, extensionId }
            );
        }
    }

    /**
     * Toggle an extension
     * 
     * @param {string} extensionId - ID of the extension to toggle
     * @param {boolean} enabled - Whether the extension should be enabled
     */
    async toggleExtension(extensionId, enabled) {
        try {
            // Check if extension is installed
            const extension = this.installedExtensions.find(e => e.id === extensionId);
            if (!extension) {
                throw new Error('Extension not installed');
            }
            
            // Toggle extension
            await this.apiClient.post(`/api/v1/developer/extensions/${extensionId}/toggle`, {
                enabled
            });
            
            // Update extension
            extension.enabled = enabled;
            
            // If enabling, load extension
            if (enabled) {
                await this.loadExtension(extension);
            } else {
                // If disabling, cleanup extension
                if (window[`extension_${extensionId}`]) {
                    // Call cleanup method if it exists
                    const extObj = window[`extension_${extensionId}`];
                    if (extObj.cleanup && typeof extObj.cleanup === 'function') {
                        extObj.cleanup();
                    }
                }
                
                // Remove extension from extension points
                for (const point in this.extensionPoints) {
                    this.extensionPoints[point] = this.extensionPoints[point].filter(e => e.id !== extensionId);
                }
            }
            
            // Trigger extension toggled event
            this.triggerEvent('extensionToggled', {
                extension,
                enabled
            });
            
            // Show success notification
            showToast(`Extension '${extension.name}' ${enabled ? 'enabled' : 'disabled'} successfully`, 'success');
        } catch (error) {
            console.error(`Failed to toggle extension ${extensionId}:`, error);
            
            // Show error notification
            showToast(`Failed to toggle extension: ${error.message}`, 'error');
            
            // Revert toggle in UI
            const checkbox = document.querySelector(`.extension-toggle-checkbox[data-extension-id="${extensionId}"]`);
            if (checkbox) {
                checkbox.checked = !enabled;
            }
            
            throw new AppError(
                'Failed to toggle extension', 
                'EXTENSION_TOGGLE_FAILED',
                { originalError: error, extensionId, enabled }
            );
        }
    }

    /**
     * Get extensions for a specific extension point
     * 
     * @param {string} point - Extension point
     * @returns {Array} - Extensions for the extension point
     */
    getExtensionsForPoint(point) {
        return this.extensionPoints[point] || [];
    }

    /**
     * Call method on all extensions for a specific extension point
     * 
     * @param {string} point - Extension point
     * @param {string} method - Method to call
     * @param {...any} args - Arguments to pass to the method
     * @returns {Array} - Results from all extensions
     */
    callExtensionMethod(point, method, ...args) {
        const extensions = this.getExtensionsForPoint(point);
        
        return extensions.map(ext => {
            try {
                // Check if method exists
                if (ext.extension[method] && typeof ext.extension[method] === 'function') {
                    return ext.extension[method](...args);
                }
                
                return null;
            } catch (error) {
                console.error(`Error calling method ${method} on extension ${ext.id}:`, error);
                return null;
            }
        }).filter(result => result !== null);
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

// Helper function to show toast notification
function showToast(message, type = 'success') {
    // Check if the developer studio has a showToast function
    if (window.developerStudio && typeof window.developerStudio.showToast === 'function') {
        window.developerStudio.showToast(message, type);
    } else {
        // Create toast element if it doesn't exist
        let toast = document.getElementById('toast-notification');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'toast-notification';
            toast.className = 'fixed bottom-4 right-4 px-4 py-2 rounded shadow-lg transform transition-all duration-300 ease-in-out translate-y-20 opacity-0 z-50';
            document.body.appendChild(toast);
        }
        
        // Set toast content
        toast.innerHTML = `
            <div class="flex items-center">
                <i class="fas ${type === 'success' ? 'fa-check-circle text-green-500' : 
                              type === 'error' ? 'fa-exclamation-circle text-red-500' : 
                              type === 'warning' ? 'fa-exclamation-triangle text-yellow-500' : 
                              'fa-info-circle text-blue-500'} mr-2"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Set toast style based on type
        toast.className = 'fixed bottom-4 right-4 px-4 py-2 rounded shadow-lg transform transition-all duration-300 ease-in-out z-50';
        
        switch (type) {
            case 'success':
                toast.classList.add('bg-green-100', 'text-green-800', 'border', 'border-green-200');
                break;
            case 'error':
                toast.classList.add('bg-red-100', 'text-red-800', 'border', 'border-red-200');
                break;
            case 'warning':
                toast.classList.add('bg-yellow-100', 'text-yellow-800', 'border', 'border-yellow-200');
                break;
            case 'info':
                toast.classList.add('bg-blue-100', 'text-blue-800', 'border', 'border-blue-200');
                break;
        }
        
        // Show toast
        setTimeout(() => {
            toast.classList.remove('translate-y-20', 'opacity-0');
            toast.classList.add('translate-y-0', 'opacity-100');
        }, 10);
        
        // Hide toast after 3 seconds
        setTimeout(() => {
            toast.classList.remove('translate-y-0', 'opacity-100');
            toast.classList.add('translate-y-20', 'opacity-0');
        }, 3000);
    }
}

// Export the class
window.ExtensionManager = ExtensionManager;
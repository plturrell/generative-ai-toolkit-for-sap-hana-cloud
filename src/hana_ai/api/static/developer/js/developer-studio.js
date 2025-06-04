/**
 * Developer Studio Module
 * 
 * Main application logic for the Developer Studio.
 */

class DeveloperStudio {
    constructor(apiClient, aiIntegration, flowBuilder, monacoIntegration) {
        this.apiClient = apiClient;
        this.aiIntegration = aiIntegration;
        this.flowBuilder = flowBuilder;
        this.monacoIntegration = monacoIntegration;
        this.currentFlowId = null;
        this.eventListeners = {};
    }

    /**
     * Initialize the Developer Studio
     */
    async initialize() {
        try {
            // Initialize flow list
            await this.loadFlowList();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Show startup toast
            this.showToast('Developer Studio initialized successfully', 'success');
            
            // Trigger initialized event
            this.triggerEvent('initialized');
            
            return true;
        } catch (error) {
            console.error('Failed to initialize Developer Studio:', error);
            this.showToast('Failed to initialize Developer Studio', 'error');
            throw new AppError(
                'Failed to initialize Developer Studio', 
                'INITIALIZATION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Set up event listeners for the UI
     */
    setupEventListeners() {
        // New flow button
        const newFlowBtn = document.getElementById('btn-new-flow');
        if (newFlowBtn) {
            newFlowBtn.addEventListener('click', () => this.createNewFlow());
        }
        
        // Open flow button
        const openFlowBtn = document.getElementById('btn-open-flow');
        if (openFlowBtn) {
            openFlowBtn.addEventListener('click', () => this.showOpenFlowDialog());
        }
        
        // Dashboard button
        const dashboardBtn = document.getElementById('btn-dashboard');
        if (dashboardBtn) {
            dashboardBtn.addEventListener('click', () => this.showDashboard());
        }
        
        // Help button
        const helpBtn = document.getElementById('btn-help');
        if (helpBtn) {
            helpBtn.addEventListener('click', () => this.showHelpDialog());
        }
        
        // Flow builder events
        this.flowBuilder.addEventListener('codeGenerated', (generationResponse) => {
            // Set code in Monaco editor
            this.monacoIntegration.setCode(generationResponse.code);
            
            // Set language
            this.monacoIntegration.setLanguage(generationResponse.language);
            
            // Show toast
            this.showToast('Code generated successfully', 'success');
        });
        
        // Connect Monaco integration with AI integration
        const executeBtn = document.querySelector('.btn-execute');
        if (executeBtn) {
            executeBtn.addEventListener('click', async () => {
                try {
                    // Get code from Monaco editor
                    const code = this.monacoIntegration.getCode();
                    
                    // Get language
                    const language = this.monacoIntegration.language;
                    
                    // Execute code
                    const result = await this.aiIntegration.executeCode(code, language);
                    
                    // Show result in output panel
                    const outputPanel = document.querySelector('.results-panel-content');
                    if (outputPanel) {
                        outputPanel.textContent = result.output;
                        
                        // Add appropriate class
                        outputPanel.className = 'results-panel-content';
                        if (!result.success) {
                            outputPanel.classList.add('error');
                        } else {
                            outputPanel.classList.add('success');
                        }
                    }
                    
                    // Show toast
                    if (result.success) {
                        this.showToast('Code executed successfully', 'success');
                    } else {
                        this.showToast(`Execution failed: ${result.error}`, 'error');
                    }
                } catch (error) {
                    console.error('Code execution error:', error);
                    this.showToast('Code execution failed', 'error');
                    
                    // Show error in output panel
                    const outputPanel = document.querySelector('.results-panel-content');
                    if (outputPanel) {
                        outputPanel.textContent = `Error: ${error.message}`;
                        outputPanel.className = 'results-panel-content error';
                    }
                }
            });
        }
    }

    /**
     * Create a new flow
     */
    async createNewFlow() {
        try {
            // Reset flow builder
            this.flowBuilder.flowName = 'Untitled Flow';
            this.flowBuilder.flowDescription = '';
            this.flowBuilder.flow = { nodes: [], edges: [] };
            this.flowBuilder.flowId = null;
            this.flowBuilder.isModified = false;
            
            // Update flow visualization
            this.flowBuilder.updateFlowVisualization();
            
            // Update UI
            const flowTitle = document.querySelector('.flow-title');
            if (flowTitle) {
                flowTitle.value = 'Untitled Flow';
            }
            
            // Clear current selection
            this.flowBuilder.currentSelection = null;
            this.flowBuilder.updatePropertiesPanel();
            
            // Show toast
            this.showToast('Created new flow', 'success');
            
            // Trigger event
            this.triggerEvent('newFlowCreated');
            
            return true;
        } catch (error) {
            console.error('Failed to create new flow:', error);
            this.showToast('Failed to create new flow', 'error');
            throw new AppError(
                'Failed to create new flow', 
                'CREATE_FLOW_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Show open flow dialog
     */
    async showOpenFlowDialog() {
        try {
            // Get flow list
            const flows = await this.apiClient.get('/api/v1/developer/flows');
            
            // Create dialog
            const dialog = document.createElement('div');
            dialog.className = 'dialog-overlay';
            dialog.innerHTML = `
                <div class="dialog">
                    <div class="dialog-header">
                        <div class="dialog-title">Open Flow</div>
                        <button class="dialog-close">&times;</button>
                    </div>
                    <div class="dialog-content">
                        <div class="file-browser">
                            ${flows.length > 0 ? 
                                flows.map(flow => `
                                    <div class="file-list-item" data-flow-id="${flow.id}">
                                        <i class="fas fa-project-diagram"></i>
                                        <span>${flow.name}</span>
                                    </div>
                                `).join('') : 
                                '<div class="p-4 text-center text-gray-400">No flows found</div>'
                            }
                        </div>
                    </div>
                    <div class="dialog-footer">
                        <button class="btn-cancel">Cancel</button>
                        <button class="btn-primary" disabled>Open</button>
                    </div>
                </div>
            `;
            
            // Add dialog to DOM
            document.body.appendChild(dialog);
            
            // Add event listeners
            const closeBtn = dialog.querySelector('.dialog-close');
            const cancelBtn = dialog.querySelector('.btn-cancel');
            const openBtn = dialog.querySelector('.btn-primary');
            const fileItems = dialog.querySelectorAll('.file-list-item');
            
            // Close dialog
            const closeDialog = () => {
                document.body.removeChild(dialog);
            };
            
            closeBtn.addEventListener('click', closeDialog);
            cancelBtn.addEventListener('click', closeDialog);
            
            // File selection
            let selectedFlowId = null;
            
            fileItems.forEach(item => {
                item.addEventListener('click', () => {
                    // Remove selected class from all items
                    fileItems.forEach(i => i.classList.remove('selected'));
                    
                    // Add selected class to clicked item
                    item.classList.add('selected');
                    
                    // Enable open button
                    openBtn.removeAttribute('disabled');
                    
                    // Set selected flow ID
                    selectedFlowId = item.getAttribute('data-flow-id');
                });
                
                // Double click to open
                item.addEventListener('dblclick', async () => {
                    selectedFlowId = item.getAttribute('data-flow-id');
                    closeDialog();
                    await this.openFlow(selectedFlowId);
                });
            });
            
            // Open button
            openBtn.addEventListener('click', async () => {
                if (selectedFlowId) {
                    closeDialog();
                    await this.openFlow(selectedFlowId);
                }
            });
        } catch (error) {
            console.error('Failed to show open flow dialog:', error);
            this.showToast('Failed to load flows', 'error');
            throw new AppError(
                'Failed to show open flow dialog', 
                'OPEN_DIALOG_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Open a flow
     * 
     * @param {String} flowId - ID of the flow to open
     */
    async openFlow(flowId) {
        try {
            // Check if current flow is modified
            if (this.flowBuilder.isModified) {
                const confirmResult = confirm('Current flow has unsaved changes. Do you want to save them?');
                if (confirmResult) {
                    await this.flowBuilder.saveFlow();
                }
            }
            
            // Load flow from API
            await this.flowBuilder.loadFlow(flowId);
            
            // Update current flow ID
            this.currentFlowId = flowId;
            
            // Trigger code generation for the flow
            const generationResponse = await this.aiIntegration.generateCode(this.flowBuilder.flow);
            
            // Set code in Monaco editor
            this.monacoIntegration.setCode(generationResponse.code);
            
            // Set language
            this.monacoIntegration.setLanguage(generationResponse.language);
            
            // Trigger event
            this.triggerEvent('flowOpened', { flowId });
            
            return true;
        } catch (error) {
            console.error('Failed to open flow:', error);
            this.showToast(`Failed to open flow: ${error.message}`, 'error');
            throw new AppError(
                'Failed to open flow', 
                'OPEN_FLOW_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Show dashboard
     */
    showDashboard() {
        try {
            // Create dialog
            const dialog = document.createElement('div');
            dialog.className = 'dialog-overlay';
            dialog.innerHTML = `
                <div class="dialog dashboard-dialog">
                    <div class="dialog-header">
                        <div class="dialog-title">Dashboard</div>
                        <button class="dialog-close">&times;</button>
                    </div>
                    <div class="dialog-content">
                        <div class="dashboard-tabs">
                            <div class="dashboard-tab active" data-tab="overview">Overview</div>
                            <div class="dashboard-tab" data-tab="performance">Performance</div>
                            <div class="dashboard-tab" data-tab="models">Models</div>
                            <div class="dashboard-tab" data-tab="flows">Flows</div>
                        </div>
                        
                        <div class="dashboard-content">
                            <!-- Overview Tab (default active) -->
                            <div class="dashboard-tab-content active" id="tab-overview">
                                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                    <div class="stat-card bg-blue-50 border border-blue-100 rounded-lg p-4">
                                        <div class="text-blue-500 text-lg font-bold">10</div>
                                        <div class="text-sm text-gray-600">Active Flows</div>
                                    </div>
                                    <div class="stat-card bg-green-50 border border-green-100 rounded-lg p-4">
                                        <div class="text-green-500 text-lg font-bold">3</div>
                                        <div class="text-sm text-gray-600">Active Models</div>
                                    </div>
                                    <div class="stat-card bg-purple-50 border border-purple-100 rounded-lg p-4">
                                        <div class="text-purple-500 text-lg font-bold">1,256</div>
                                        <div class="text-sm text-gray-600">API Requests (24h)</div>
                                    </div>
                                </div>
                                
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div class="chart-container bg-white border border-gray-200 rounded-lg p-4">
                                        <div class="chart-title text-sm font-bold text-gray-700 mb-2">API Usage</div>
                                        <div class="chart-placeholder h-60 bg-gray-50 flex items-center justify-center">
                                            <div class="text-center">
                                                <i class="fas fa-chart-line text-gray-400 text-3xl mb-2"></i>
                                                <div class="text-sm text-gray-500">API usage chart</div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="chart-container bg-white border border-gray-200 rounded-lg p-4">
                                        <div class="chart-title text-sm font-bold text-gray-700 mb-2">Model Performance</div>
                                        <div class="chart-placeholder h-60 bg-gray-50 flex items-center justify-center">
                                            <div class="text-center">
                                                <i class="fas fa-chart-bar text-gray-400 text-3xl mb-2"></i>
                                                <div class="text-sm text-gray-500">Performance metrics chart</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Performance Tab -->
                            <div class="dashboard-tab-content" id="tab-performance">
                                <div class="bg-white border border-gray-200 rounded-lg p-4 mb-6">
                                    <div class="text-sm font-bold text-gray-700 mb-4">Response Time (ms)</div>
                                    <div class="chart-placeholder h-60 bg-gray-50 flex items-center justify-center">
                                        <div class="text-center">
                                            <i class="fas fa-tachometer-alt text-gray-400 text-3xl mb-2"></i>
                                            <div class="text-sm text-gray-500">Response time chart</div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div class="bg-white border border-gray-200 rounded-lg p-4">
                                        <div class="text-sm font-bold text-gray-700 mb-4">GPU Utilization</div>
                                        <div class="chart-placeholder h-40 bg-gray-50 flex items-center justify-center">
                                            <div class="text-center">
                                                <i class="fas fa-microchip text-gray-400 text-3xl mb-2"></i>
                                                <div class="text-sm text-gray-500">GPU usage chart</div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="bg-white border border-gray-200 rounded-lg p-4">
                                        <div class="text-sm font-bold text-gray-700 mb-4">Memory Usage</div>
                                        <div class="chart-placeholder h-40 bg-gray-50 flex items-center justify-center">
                                            <div class="text-center">
                                                <i class="fas fa-memory text-gray-400 text-3xl mb-2"></i>
                                                <div class="text-sm text-gray-500">Memory usage chart</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Models Tab -->
                            <div class="dashboard-tab-content" id="tab-models">
                                <div class="bg-white border border-gray-200 rounded-lg p-4">
                                    <div class="flex justify-between items-center mb-4">
                                        <div class="text-sm font-bold text-gray-700">Deployed Models</div>
                                        <button class="text-xs bg-blue-500 text-white px-2 py-1 rounded">
                                            <i class="fas fa-plus mr-1"></i> New Model
                                        </button>
                                    </div>
                                    <table class="w-full text-sm">
                                        <thead>
                                            <tr class="border-b">
                                                <th class="text-left py-2">Name</th>
                                                <th class="text-left py-2">Type</th>
                                                <th class="text-left py-2">Status</th>
                                                <th class="text-left py-2">Updated</th>
                                                <th class="text-left py-2">Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr class="border-b">
                                                <td class="py-2">Sales Prediction</td>
                                                <td class="py-2">Time Series</td>
                                                <td class="py-2"><span class="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Active</span></td>
                                                <td class="py-2">2 days ago</td>
                                                <td class="py-2">
                                                    <button class="text-gray-600 hover:text-gray-900 mr-2"><i class="fas fa-edit"></i></button>
                                                    <button class="text-gray-600 hover:text-gray-900"><i class="fas fa-trash"></i></button>
                                                </td>
                                            </tr>
                                            <tr class="border-b">
                                                <td class="py-2">Customer Segmentation</td>
                                                <td class="py-2">Clustering</td>
                                                <td class="py-2"><span class="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Active</span></td>
                                                <td class="py-2">5 days ago</td>
                                                <td class="py-2">
                                                    <button class="text-gray-600 hover:text-gray-900 mr-2"><i class="fas fa-edit"></i></button>
                                                    <button class="text-gray-600 hover:text-gray-900"><i class="fas fa-trash"></i></button>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td class="py-2">Product Recommendation</td>
                                                <td class="py-2">Neural Network</td>
                                                <td class="py-2"><span class="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs">Training</span></td>
                                                <td class="py-2">1 hour ago</td>
                                                <td class="py-2">
                                                    <button class="text-gray-600 hover:text-gray-900 mr-2"><i class="fas fa-edit"></i></button>
                                                    <button class="text-gray-600 hover:text-gray-900"><i class="fas fa-trash"></i></button>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            
                            <!-- Flows Tab -->
                            <div class="dashboard-tab-content" id="tab-flows">
                                <div class="bg-white border border-gray-200 rounded-lg p-4">
                                    <div class="flex justify-between items-center mb-4">
                                        <div class="text-sm font-bold text-gray-700">Deployed Flows</div>
                                        <button class="text-xs bg-blue-500 text-white px-2 py-1 rounded" id="dashboard-new-flow">
                                            <i class="fas fa-plus mr-1"></i> New Flow
                                        </button>
                                    </div>
                                    <table class="w-full text-sm">
                                        <thead>
                                            <tr class="border-b">
                                                <th class="text-left py-2">Name</th>
                                                <th class="text-left py-2">Type</th>
                                                <th class="text-left py-2">Status</th>
                                                <th class="text-left py-2">Created</th>
                                                <th class="text-left py-2">Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr class="border-b">
                                                <td class="py-2">Customer Analysis</td>
                                                <td class="py-2">Data Analysis</td>
                                                <td class="py-2"><span class="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Active</span></td>
                                                <td class="py-2">3 days ago</td>
                                                <td class="py-2">
                                                    <button class="text-gray-600 hover:text-gray-900 mr-2 btn-edit-flow" data-flow-id="1"><i class="fas fa-edit"></i></button>
                                                    <button class="text-gray-600 hover:text-gray-900 mr-2"><i class="fas fa-play"></i></button>
                                                    <button class="text-gray-600 hover:text-gray-900"><i class="fas fa-trash"></i></button>
                                                </td>
                                            </tr>
                                            <tr class="border-b">
                                                <td class="py-2">Sales Forecast</td>
                                                <td class="py-2">Predictive</td>
                                                <td class="py-2"><span class="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Active</span></td>
                                                <td class="py-2">1 week ago</td>
                                                <td class="py-2">
                                                    <button class="text-gray-600 hover:text-gray-900 mr-2 btn-edit-flow" data-flow-id="2"><i class="fas fa-edit"></i></button>
                                                    <button class="text-gray-600 hover:text-gray-900 mr-2"><i class="fas fa-play"></i></button>
                                                    <button class="text-gray-600 hover:text-gray-900"><i class="fas fa-trash"></i></button>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td class="py-2">Product Inventory</td>
                                                <td class="py-2">Reporting</td>
                                                <td class="py-2"><span class="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs">Inactive</span></td>
                                                <td class="py-2">2 weeks ago</td>
                                                <td class="py-2">
                                                    <button class="text-gray-600 hover:text-gray-900 mr-2 btn-edit-flow" data-flow-id="3"><i class="fas fa-edit"></i></button>
                                                    <button class="text-gray-600 hover:text-gray-900 mr-2"><i class="fas fa-play"></i></button>
                                                    <button class="text-gray-600 hover:text-gray-900"><i class="fas fa-trash"></i></button>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="dialog-footer">
                        <button class="btn-primary">Close</button>
                    </div>
                </div>
            `;
            
            // Add dialog to DOM
            document.body.appendChild(dialog);
            
            // Add event listeners
            const closeBtn = dialog.querySelector('.dialog-close');
            const primaryBtn = dialog.querySelector('.btn-primary');
            
            // Close dialog
            const closeDialog = () => {
                document.body.removeChild(dialog);
            };
            
            closeBtn.addEventListener('click', closeDialog);
            primaryBtn.addEventListener('click', closeDialog);
            
            // Tab switching
            const tabs = dialog.querySelectorAll('.dashboard-tab');
            const tabContents = dialog.querySelectorAll('.dashboard-tab-content');
            
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    const tabId = tab.getAttribute('data-tab');
                    
                    // Hide all tab contents
                    tabContents.forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    // Show selected tab content
                    const activeContent = document.getElementById(`tab-${tabId}`);
                    if (activeContent) {
                        activeContent.classList.add('active');
                    }
                    
                    // Update active tab
                    tabs.forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                });
            });
            
            // New flow button in dashboard
            const newFlowBtn = dialog.querySelector('#dashboard-new-flow');
            if (newFlowBtn) {
                newFlowBtn.addEventListener('click', () => {
                    closeDialog();
                    this.createNewFlow();
                });
            }
            
            // Edit flow buttons
            const editFlowBtns = dialog.querySelectorAll('.btn-edit-flow');
            editFlowBtns.forEach(btn => {
                btn.addEventListener('click', async () => {
                    const flowId = btn.getAttribute('data-flow-id');
                    closeDialog();
                    await this.openFlow(flowId);
                });
            });
            
            // Trigger event
            this.triggerEvent('dashboardOpened');
            
        } catch (error) {
            console.error('Failed to show dashboard:', error);
            this.showToast('Failed to show dashboard', 'error');
        }
    }

    /**
     * Show help dialog
     */
    showHelpDialog() {
        // Create dialog
        const dialog = document.createElement('div');
        dialog.className = 'dialog-overlay';
        dialog.innerHTML = `
            <div class="dialog">
                <div class="dialog-header">
                    <div class="dialog-title">Developer Studio Help</div>
                    <button class="dialog-close">&times;</button>
                </div>
                <div class="dialog-content">
                    <h3 class="text-lg font-bold mb-4">Welcome to the SAP HANA AI Toolkit Developer Studio</h3>
                    
                    <div class="mb-4">
                        <h4 class="font-bold text-gray-700 mb-2">Flow Builder</h4>
                        <p class="text-sm text-gray-600 mb-2">The Flow Builder allows you to create data processing flows visually:</p>
                        <ul class="list-disc ml-5 text-sm text-gray-600">
                            <li>Drag nodes from the palette to the canvas</li>
                            <li>Connect nodes by dragging from the right handle to the left handle of another node</li>
                            <li>Select a node to edit its properties</li>
                            <li>Delete selected nodes with the Delete key or the properties panel</li>
                        </ul>
                    </div>
                    
                    <div class="mb-4">
                        <h4 class="font-bold text-gray-700 mb-2">Code Editor</h4>
                        <p class="text-sm text-gray-600 mb-2">The Code Editor shows generated code from your flow:</p>
                        <ul class="list-disc ml-5 text-sm text-gray-600">
                            <li>Code is automatically generated when you modify your flow</li>
                            <li>You can manually edit the code if needed</li>
                            <li>Execute the code to see the results</li>
                            <li>Switch languages using the dropdown</li>
                        </ul>
                    </div>
                    
                    <div class="mb-4">
                        <h4 class="font-bold text-gray-700 mb-2">Keyboard Shortcuts</h4>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div class="font-medium">Ctrl+Z</div><div>Undo</div>
                            <div class="font-medium">Ctrl+Y</div><div>Redo</div>
                            <div class="font-medium">Delete</div><div>Delete selected node or edge</div>
                            <div class="font-medium">Ctrl+S</div><div>Save flow</div>
                            <div class="font-medium">Ctrl+F</div><div>Search in code editor</div>
                            <div class="font-medium">F5</div><div>Execute code</div>
                        </div>
                    </div>
                    
                    <div>
                        <h4 class="font-bold text-gray-700 mb-2">Additional Resources</h4>
                        <ul class="list-disc ml-5 text-sm text-gray-600">
                            <li><a href="https://github.com/SAP-samples/generative-ai-toolkit-for-sap-hana-cloud" class="text-blue-600 hover:underline" target="_blank">GitHub Repository</a></li>
                            <li><a href="https://www.sap.com/products/technology-platform/hana/cloud.html" class="text-blue-600 hover:underline" target="_blank">SAP HANA Cloud Documentation</a></li>
                            <li><a href="https://help.sap.com/docs/hana-cloud/sap-hana-cloud-overview/what-is-sap-hana-cloud" class="text-blue-600 hover:underline" target="_blank">SAP HANA Cloud Overview</a></li>
                        </ul>
                    </div>
                </div>
                <div class="dialog-footer">
                    <button class="btn-primary">Close</button>
                </div>
            </div>
        `;
        
        // Add dialog to DOM
        document.body.appendChild(dialog);
        
        // Add event listeners
        const closeBtn = dialog.querySelector('.dialog-close');
        const primaryBtn = dialog.querySelector('.btn-primary');
        
        // Close dialog
        const closeDialog = () => {
            document.body.removeChild(dialog);
        };
        
        closeBtn.addEventListener('click', closeDialog);
        primaryBtn.addEventListener('click', closeDialog);
    }

    /**
     * Load flow list from API
     */
    async loadFlowList() {
        try {
            // Get flows from API
            const flows = await this.apiClient.get('/api/v1/developer/flows');
            
            // Create navigator items for flows
            const navigatorContent = document.querySelector('.navigator-content');
            if (navigatorContent) {
                // Clear existing flows
                const flowsGroup = navigatorContent.querySelector('.navigator-group:first-child');
                if (flowsGroup) {
                    const groupTitle = flowsGroup.querySelector('.navigator-group-title');
                    flowsGroup.innerHTML = '';
                    flowsGroup.appendChild(groupTitle);
                    
                    // Add flows
                    flows.forEach(flow => {
                        const flowItem = document.createElement('div');
                        flowItem.className = 'navigator-item';
                        flowItem.innerHTML = `
                            <i class="fas fa-project-diagram"></i>
                            <span>${flow.name}</span>
                        `;
                        
                        // Add click event to open flow
                        flowItem.addEventListener('click', async () => {
                            await this.openFlow(flow.id);
                            
                            // Remove active class from all items
                            const items = navigatorContent.querySelectorAll('.navigator-item');
                            items.forEach(item => item.classList.remove('active'));
                            
                            // Add active class to clicked item
                            flowItem.classList.add('active');
                        });
                        
                        flowsGroup.appendChild(flowItem);
                    });
                }
            }
            
            // Trigger event
            this.triggerEvent('flowListLoaded', flows);
            
            return flows;
        } catch (error) {
            console.error('Failed to load flow list:', error);
            this.showToast('Failed to load flows', 'error');
            throw new AppError(
                'Failed to load flow list', 
                'LOAD_FLOW_LIST_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Show a toast message
     * 
     * @param {String} message - Message to show
     * @param {String} type - Message type ('success', 'error', 'warning', 'info')
     */
    showToast(message, type = 'success') {
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
window.DeveloperStudio = DeveloperStudio;
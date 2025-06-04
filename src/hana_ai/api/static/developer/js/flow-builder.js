/**
 * Flow Builder Module
 * 
 * Core functionality for the visual flow builder component of the Developer Studio.
 * Uses React Flow to create a node-based visual programming environment.
 */

class FlowBuilder {
    constructor(containerId, apiClient, aiIntegration) {
        this.containerId = containerId;
        this.apiClient = apiClient;
        this.aiIntegration = aiIntegration;
        this.flow = { nodes: [], edges: [] };
        this.reactFlowInstance = null;
        this.nodeTypes = this.getNodeTypes();
        this.edgeTypes = this.getEdgeTypes();
        this.flowId = null;
        this.flowName = 'Untitled Flow';
        this.flowDescription = '';
        this.isModified = false;
        this.currentSelection = null;
        this.undoStack = [];
        this.redoStack = [];
        this.maxUndoStackSize = 50;
        this.eventListeners = {};
    }

    /**
     * Initialize the flow builder
     */
    async initialize() {
        try {
            // Create the React Flow component
            this.createReactFlowComponent();
            
            // Initialize event listeners
            this.initializeEventListeners();
            
            // Load node palette
            await this.loadNodePalette();
            
            // Trigger initialized event
            this.triggerEvent('initialized');
        } catch (error) {
            console.error('Failed to initialize flow builder:', error);
            throw new AppError(
                'Failed to initialize flow builder', 
                'INITIALIZATION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Create the React Flow component
     */
    createReactFlowComponent() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            throw new Error(`Container with ID '${this.containerId}' not found`);
        }

        // This would normally use React and React Flow in a real implementation
        // For this demonstration, we'll simulate the creation
        container.innerHTML = `
            <div class="flow-builder-header">
                <div class="flow-title-container">
                    <input type="text" class="flow-title" value="${this.flowName}" placeholder="Flow Name">
                    <button class="btn-save" title="Save Flow"><i class="fas fa-save"></i></button>
                    <button class="btn-run" title="Run Flow"><i class="fas fa-play"></i></button>
                </div>
                <div class="flow-actions">
                    <button class="btn-undo" title="Undo"><i class="fas fa-undo"></i></button>
                    <button class="btn-redo" title="Redo"><i class="fas fa-redo"></i></button>
                    <button class="btn-delete" title="Delete Selected"><i class="fas fa-trash"></i></button>
                    <button class="btn-export" title="Export Flow"><i class="fas fa-file-export"></i></button>
                    <button class="btn-validate" title="Validate Flow"><i class="fas fa-check-circle"></i></button>
                </div>
            </div>
            <div class="flow-builder-container">
                <div class="node-palette">
                    <h3>Nodes</h3>
                    <div class="node-categories">
                        <div class="node-category">
                            <h4>Data Sources</h4>
                            <div class="node-list" id="data-source-nodes"></div>
                        </div>
                        <div class="node-category">
                            <h4>Transformations</h4>
                            <div class="node-list" id="transformation-nodes"></div>
                        </div>
                        <div class="node-category">
                            <h4>Analysis</h4>
                            <div class="node-list" id="analysis-nodes"></div>
                        </div>
                        <div class="node-category">
                            <h4>Visualization</h4>
                            <div class="node-list" id="visualization-nodes"></div>
                        </div>
                    </div>
                </div>
                <div class="react-flow-container" id="react-flow-canvas">
                    <div class="react-flow-placeholder">Loading flow editor...</div>
                </div>
                <div class="node-properties">
                    <h3>Properties</h3>
                    <div class="properties-content" id="node-properties-content">
                        <div class="no-selection-message">Select a node to edit its properties</div>
                    </div>
                </div>
            </div>
            <div class="flow-builder-footer">
                <div class="status-message"></div>
                <div class="validation-status"></div>
            </div>
        `;
    }

    /**
     * Initialize event listeners for the flow builder
     */
    initializeEventListeners() {
        // In a real implementation, these would be React component events
        // For this demonstration, we'll simulate with DOM event listeners
        
        // Flow title input
        const titleInput = document.querySelector(`#${this.containerId} .flow-title`);
        if (titleInput) {
            titleInput.addEventListener('change', (e) => {
                this.flowName = e.target.value;
                this.isModified = true;
                this.triggerEvent('flowModified');
            });
        }
        
        // Save button
        const saveButton = document.querySelector(`#${this.containerId} .btn-save`);
        if (saveButton) {
            saveButton.addEventListener('click', () => this.saveFlow());
        }
        
        // Run button
        const runButton = document.querySelector(`#${this.containerId} .btn-run`);
        if (runButton) {
            runButton.addEventListener('click', () => this.generateAndExecuteCode());
        }
        
        // Undo button
        const undoButton = document.querySelector(`#${this.containerId} .btn-undo`);
        if (undoButton) {
            undoButton.addEventListener('click', () => this.undo());
        }
        
        // Redo button
        const redoButton = document.querySelector(`#${this.containerId} .btn-redo`);
        if (redoButton) {
            redoButton.addEventListener('click', () => this.redo());
        }
        
        // Delete button
        const deleteButton = document.querySelector(`#${this.containerId} .btn-delete`);
        if (deleteButton) {
            deleteButton.addEventListener('click', () => this.deleteSelected());
        }
        
        // Export button
        const exportButton = document.querySelector(`#${this.containerId} .btn-export`);
        if (exportButton) {
            exportButton.addEventListener('click', () => this.exportFlow());
        }
        
        // Validate button
        const validateButton = document.querySelector(`#${this.containerId} .btn-validate`);
        if (validateButton) {
            validateButton.addEventListener('click', () => this.validateFlow());
        }
    }

    /**
     * Load node palette with available node types
     */
    async loadNodePalette() {
        try {
            // In a real implementation, this might load from an API
            // For this demonstration, we'll use hardcoded node types
            
            // Data Source Nodes
            const dataSourceNodes = [
                { id: 'hana-table', type: 'dataSource', label: 'HANA Table', description: 'Read data from a HANA table' },
                { id: 'sql-query', type: 'dataSource', label: 'SQL Query', description: 'Execute a custom SQL query' },
                { id: 'csv-upload', type: 'dataSource', label: 'CSV Upload', description: 'Import data from a CSV file' }
            ];
            
            // Transformation Nodes
            const transformationNodes = [
                { id: 'filter', type: 'transformation', label: 'Filter', description: 'Filter rows based on conditions' },
                { id: 'join', type: 'transformation', label: 'Join', description: 'Join with another data source' },
                { id: 'group-by', type: 'transformation', label: 'Group By', description: 'Aggregate data by groups' },
                { id: 'sort', type: 'transformation', label: 'Sort', description: 'Sort data by columns' },
                { id: 'project', type: 'transformation', label: 'Select Columns', description: 'Select specific columns' }
            ];
            
            // Analysis Nodes
            const analysisNodes = [
                { id: 'time-series', type: 'analysis', label: 'Time Series', description: 'Time series forecasting' },
                { id: 'regression', type: 'analysis', label: 'Regression', description: 'Linear/Logistic regression' },
                { id: 'clustering', type: 'analysis', label: 'Clustering', description: 'K-means clustering' },
                { id: 'text-analysis', type: 'analysis', label: 'Text Analysis', description: 'Natural language processing' }
            ];
            
            // Visualization Nodes
            const visualizationNodes = [
                { id: 'bar-chart', type: 'visualization', label: 'Bar Chart', description: 'Create a bar chart' },
                { id: 'line-chart', type: 'visualization', label: 'Line Chart', description: 'Create a line chart' },
                { id: 'scatter-plot', type: 'visualization', label: 'Scatter Plot', description: 'Create a scatter plot' },
                { id: 'table-view', type: 'visualization', label: 'Table View', description: 'Display data as a table' }
            ];
            
            // Render node palette items
            this.renderNodePaletteItems('data-source-nodes', dataSourceNodes);
            this.renderNodePaletteItems('transformation-nodes', transformationNodes);
            this.renderNodePaletteItems('analysis-nodes', analysisNodes);
            this.renderNodePaletteItems('visualization-nodes', visualizationNodes);
            
            // Initialize drag-and-drop functionality
            this.initializeDragAndDrop();
        } catch (error) {
            console.error('Failed to load node palette:', error);
            throw new AppError(
                'Failed to load node palette', 
                'NODE_PALETTE_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Render node palette items
     * 
     * @param {String} containerId - ID of the container to render items in
     * @param {Array} items - Array of node items to render
     */
    renderNodePaletteItems(containerId, items) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        items.forEach(item => {
            const nodeEl = document.createElement('div');
            nodeEl.className = 'node-item';
            nodeEl.setAttribute('data-node-type', item.id);
            nodeEl.setAttribute('draggable', 'true');
            nodeEl.innerHTML = `
                <div class="node-item-header">${item.label}</div>
                <div class="node-item-description">${item.description}</div>
            `;
            container.appendChild(nodeEl);
        });
    }

    /**
     * Initialize drag-and-drop functionality for nodes
     */
    initializeDragAndDrop() {
        // In a real implementation, this would use React Flow's drag-and-drop API
        // For this demonstration, we'll simulate the functionality
        
        const nodeItems = document.querySelectorAll('.node-item');
        nodeItems.forEach(nodeItem => {
            nodeItem.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('nodeType', nodeItem.getAttribute('data-node-type'));
            });
        });
        
        const reactFlowCanvas = document.getElementById('react-flow-canvas');
        if (reactFlowCanvas) {
            reactFlowCanvas.addEventListener('dragover', (e) => {
                e.preventDefault();
            });
            
            reactFlowCanvas.addEventListener('drop', (e) => {
                e.preventDefault();
                const nodeType = e.dataTransfer.getData('nodeType');
                if (nodeType) {
                    // Calculate position relative to the canvas
                    const reactFlowBounds = reactFlowCanvas.getBoundingClientRect();
                    const position = {
                        x: e.clientX - reactFlowBounds.left,
                        y: e.clientY - reactFlowBounds.top
                    };
                    
                    // Add node to the flow
                    this.addNode(nodeType, position);
                }
            });
        }
    }

    /**
     * Get node types for React Flow
     * 
     * @returns {Object} Node types object for React Flow
     */
    getNodeTypes() {
        // In a real implementation, this would return React components
        // For this demonstration, we'll return a placeholder object
        return {
            dataSource: {},
            transformation: {},
            analysis: {},
            visualization: {}
        };
    }

    /**
     * Get edge types for React Flow
     * 
     * @returns {Object} Edge types object for React Flow
     */
    getEdgeTypes() {
        // In a real implementation, this would return React components
        // For this demonstration, we'll return a placeholder object
        return {
            default: {}
        };
    }

    /**
     * Add a node to the flow
     * 
     * @param {String} nodeType - Type of node to add
     * @param {Object} position - Position of the node
     * @returns {String} ID of the created node
     */
    addNode(nodeType, position) {
        // Save current state for undo
        this.saveStateForUndo();
        
        // Generate node ID
        const nodeId = `node_${Date.now()}`;
        
        // Create node data based on type
        const nodeData = this.createNodeData(nodeType);
        
        // Add node to the flow
        this.flow.nodes.push({
            id: nodeId,
            type: nodeData.type,
            data: nodeData.data,
            position
        });
        
        // Update the UI
        this.updateFlowVisualization();
        
        // Mark flow as modified
        this.isModified = true;
        this.triggerEvent('flowModified');
        
        return nodeId;
    }

    /**
     * Create node data based on node type
     * 
     * @param {String} nodeType - Type of node
     * @returns {Object} Node data
     */
    createNodeData(nodeType) {
        // In a real implementation, this would create node data based on the type
        // For this demonstration, we'll use hardcoded data for different types
        
        switch (nodeType) {
            case 'hana-table':
                return {
                    type: 'dataSource',
                    data: {
                        label: 'HANA Table',
                        tableName: '',
                        schema: '',
                        columns: []
                    }
                };
                
            case 'sql-query':
                return {
                    type: 'dataSource',
                    data: {
                        label: 'SQL Query',
                        query: 'SELECT * FROM'
                    }
                };
                
            case 'filter':
                return {
                    type: 'transformation',
                    data: {
                        label: 'Filter',
                        conditions: []
                    }
                };
                
            case 'join':
                return {
                    type: 'transformation',
                    data: {
                        label: 'Join',
                        joinType: 'INNER',
                        leftField: '',
                        rightField: ''
                    }
                };
                
            case 'time-series':
                return {
                    type: 'analysis',
                    data: {
                        label: 'Time Series',
                        timeColumn: '',
                        valueColumn: '',
                        horizon: 10,
                        method: 'auto'
                    }
                };
                
            case 'bar-chart':
                return {
                    type: 'visualization',
                    data: {
                        label: 'Bar Chart',
                        xAxis: '',
                        yAxis: '',
                        title: 'Bar Chart'
                    }
                };
                
            default:
                return {
                    type: 'default',
                    data: {
                        label: 'Node'
                    }
                };
        }
    }

    /**
     * Add an edge between nodes
     * 
     * @param {String} sourceId - ID of the source node
     * @param {String} targetId - ID of the target node
     * @returns {String} ID of the created edge
     */
    addEdge(sourceId, targetId) {
        // Save current state for undo
        this.saveStateForUndo();
        
        // Generate edge ID
        const edgeId = `edge_${Date.now()}`;
        
        // Add edge to the flow
        this.flow.edges.push({
            id: edgeId,
            source: sourceId,
            target: targetId
        });
        
        // Update the UI
        this.updateFlowVisualization();
        
        // Mark flow as modified
        this.isModified = true;
        this.triggerEvent('flowModified');
        
        return edgeId;
    }

    /**
     * Update flow visualization
     */
    updateFlowVisualization() {
        // In a real implementation, this would update the React Flow component
        // For this demonstration, we'll simulate the update with a status message
        
        const statusMessage = document.querySelector(`#${this.containerId} .status-message`);
        if (statusMessage) {
            statusMessage.textContent = `Flow updated: ${this.flow.nodes.length} nodes, ${this.flow.edges.length} edges`;
        }
    }

    /**
     * Select a node or edge
     * 
     * @param {String} id - ID of the node or edge to select
     * @param {String} type - Type of the selected element ('node' or 'edge')
     */
    selectElement(id, type) {
        this.currentSelection = { id, type };
        
        // Update properties panel based on selection
        this.updatePropertiesPanel();
        
        // Trigger selection event
        this.triggerEvent('selectionChanged', this.currentSelection);
    }

    /**
     * Update properties panel based on current selection
     */
    updatePropertiesPanel() {
        const propertiesContent = document.getElementById('node-properties-content');
        if (!propertiesContent) return;
        
        if (!this.currentSelection) {
            propertiesContent.innerHTML = '<div class="no-selection-message">Select a node to edit its properties</div>';
            return;
        }
        
        const { id, type } = this.currentSelection;
        
        if (type === 'node') {
            // Find the selected node
            const node = this.flow.nodes.find(n => n.id === id);
            if (!node) return;
            
            // Generate properties form based on node type
            let formHtml = `
                <h4>${node.data.label} Properties</h4>
                <form id="node-properties-form">
                    <input type="hidden" name="nodeId" value="${id}">
            `;
            
            // Add properties based on node type
            switch (node.type) {
                case 'dataSource':
                    if (node.data.label === 'HANA Table') {
                        formHtml += `
                            <div class="form-group">
                                <label for="tableName">Table Name</label>
                                <input type="text" id="tableName" name="tableName" value="${node.data.tableName || ''}">
                            </div>
                            <div class="form-group">
                                <label for="schema">Schema</label>
                                <input type="text" id="schema" name="schema" value="${node.data.schema || ''}">
                            </div>
                        `;
                    } else if (node.data.label === 'SQL Query') {
                        formHtml += `
                            <div class="form-group">
                                <label for="query">SQL Query</label>
                                <textarea id="query" name="query" rows="5">${node.data.query || ''}</textarea>
                            </div>
                        `;
                    }
                    break;
                    
                case 'transformation':
                    if (node.data.label === 'Filter') {
                        formHtml += `
                            <div class="form-group">
                                <label for="filterCondition">Filter Condition</label>
                                <input type="text" id="filterCondition" name="filterCondition" placeholder="column = value">
                            </div>
                            <button type="button" class="btn-add-condition">Add Condition</button>
                            <div class="conditions-list">
                                ${(node.data.conditions || []).map((condition, index) => 
                                    `<div class="condition-item">
                                        ${condition}
                                        <button type="button" class="btn-remove-condition" data-index="${index}">×</button>
                                    </div>`
                                ).join('')}
                            </div>
                        `;
                    } else if (node.data.label === 'Join') {
                        formHtml += `
                            <div class="form-group">
                                <label for="joinType">Join Type</label>
                                <select id="joinType" name="joinType">
                                    <option value="INNER" ${node.data.joinType === 'INNER' ? 'selected' : ''}>INNER JOIN</option>
                                    <option value="LEFT" ${node.data.joinType === 'LEFT' ? 'selected' : ''}>LEFT JOIN</option>
                                    <option value="RIGHT" ${node.data.joinType === 'RIGHT' ? 'selected' : ''}>RIGHT JOIN</option>
                                    <option value="FULL" ${node.data.joinType === 'FULL' ? 'selected' : ''}>FULL JOIN</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="leftField">Left Field</label>
                                <input type="text" id="leftField" name="leftField" value="${node.data.leftField || ''}">
                            </div>
                            <div class="form-group">
                                <label for="rightField">Right Field</label>
                                <input type="text" id="rightField" name="rightField" value="${node.data.rightField || ''}">
                            </div>
                        `;
                    }
                    break;
                    
                case 'analysis':
                    if (node.data.label === 'Time Series') {
                        formHtml += `
                            <div class="form-group">
                                <label for="timeColumn">Time Column</label>
                                <input type="text" id="timeColumn" name="timeColumn" value="${node.data.timeColumn || ''}">
                            </div>
                            <div class="form-group">
                                <label for="valueColumn">Value Column</label>
                                <input type="text" id="valueColumn" name="valueColumn" value="${node.data.valueColumn || ''}">
                            </div>
                            <div class="form-group">
                                <label for="horizon">Forecast Horizon</label>
                                <input type="number" id="horizon" name="horizon" value="${node.data.horizon || 10}" min="1">
                            </div>
                            <div class="form-group">
                                <label for="method">Method</label>
                                <select id="method" name="method">
                                    <option value="auto" ${node.data.method === 'auto' ? 'selected' : ''}>Auto</option>
                                    <option value="arima" ${node.data.method === 'arima' ? 'selected' : ''}>ARIMA</option>
                                    <option value="exponential" ${node.data.method === 'exponential' ? 'selected' : ''}>Exponential Smoothing</option>
                                </select>
                            </div>
                        `;
                    }
                    break;
                    
                case 'visualization':
                    if (node.data.label === 'Bar Chart' || node.data.label === 'Line Chart') {
                        formHtml += `
                            <div class="form-group">
                                <label for="title">Chart Title</label>
                                <input type="text" id="title" name="title" value="${node.data.title || ''}">
                            </div>
                            <div class="form-group">
                                <label for="xAxis">X-Axis Column</label>
                                <input type="text" id="xAxis" name="xAxis" value="${node.data.xAxis || ''}">
                            </div>
                            <div class="form-group">
                                <label for="yAxis">Y-Axis Column</label>
                                <input type="text" id="yAxis" name="yAxis" value="${node.data.yAxis || ''}">
                            </div>
                        `;
                    }
                    break;
            }
            
            formHtml += `
                <div class="form-actions">
                    <button type="button" class="btn-apply-properties">Apply</button>
                </div>
            </form>`;
            
            // Set form HTML
            propertiesContent.innerHTML = formHtml;
            
            // Add event listeners to the form
            const applyButton = propertiesContent.querySelector('.btn-apply-properties');
            if (applyButton) {
                applyButton.addEventListener('click', () => this.applyNodeProperties());
            }
            
            // Add event listeners for filter conditions
            const addConditionButton = propertiesContent.querySelector('.btn-add-condition');
            if (addConditionButton) {
                addConditionButton.addEventListener('click', () => this.addFilterCondition());
            }
            
            const removeConditionButtons = propertiesContent.querySelectorAll('.btn-remove-condition');
            removeConditionButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    const index = parseInt(e.target.getAttribute('data-index'));
                    this.removeFilterCondition(index);
                });
            });
        } else if (type === 'edge') {
            // Find the selected edge
            const edge = this.flow.edges.find(e => e.id === id);
            if (!edge) return;
            
            // Generate properties form for edge
            const formHtml = `
                <h4>Connection Properties</h4>
                <form id="edge-properties-form">
                    <input type="hidden" name="edgeId" value="${id}">
                    <div class="form-group">
                        <label for="edgeType">Type</label>
                        <select id="edgeType" name="edgeType">
                            <option value="default" selected>Default</option>
                            <option value="step">Step</option>
                            <option value="smooth">Smooth</option>
                        </select>
                    </div>
                    <div class="form-actions">
                        <button type="button" class="btn-apply-properties">Apply</button>
                    </div>
                </form>
            `;
            
            // Set form HTML
            propertiesContent.innerHTML = formHtml;
            
            // Add event listeners to the form
            const applyButton = propertiesContent.querySelector('.btn-apply-properties');
            if (applyButton) {
                applyButton.addEventListener('click', () => this.applyEdgeProperties());
            }
        }
    }

    /**
     * Apply node properties from form
     */
    applyNodeProperties() {
        // Save current state for undo
        this.saveStateForUndo();
        
        const form = document.getElementById('node-properties-form');
        if (!form) return;
        
        const nodeId = form.querySelector('[name="nodeId"]').value;
        const node = this.flow.nodes.find(n => n.id === nodeId);
        if (!node) return;
        
        // Update node properties based on type
        switch (node.type) {
            case 'dataSource':
                if (node.data.label === 'HANA Table') {
                    node.data.tableName = form.querySelector('[name="tableName"]').value;
                    node.data.schema = form.querySelector('[name="schema"]').value;
                } else if (node.data.label === 'SQL Query') {
                    node.data.query = form.querySelector('[name="query"]').value;
                }
                break;
                
            case 'transformation':
                if (node.data.label === 'Join') {
                    node.data.joinType = form.querySelector('[name="joinType"]').value;
                    node.data.leftField = form.querySelector('[name="leftField"]').value;
                    node.data.rightField = form.querySelector('[name="rightField"]').value;
                }
                break;
                
            case 'analysis':
                if (node.data.label === 'Time Series') {
                    node.data.timeColumn = form.querySelector('[name="timeColumn"]').value;
                    node.data.valueColumn = form.querySelector('[name="valueColumn"]').value;
                    node.data.horizon = parseInt(form.querySelector('[name="horizon"]').value);
                    node.data.method = form.querySelector('[name="method"]').value;
                }
                break;
                
            case 'visualization':
                if (node.data.label === 'Bar Chart' || node.data.label === 'Line Chart') {
                    node.data.title = form.querySelector('[name="title"]').value;
                    node.data.xAxis = form.querySelector('[name="xAxis"]').value;
                    node.data.yAxis = form.querySelector('[name="yAxis"]').value;
                }
                break;
        }
        
        // Update the UI
        this.updateFlowVisualization();
        
        // Mark flow as modified
        this.isModified = true;
        this.triggerEvent('flowModified');
        
        // Show success message
        this.showStatusMessage('Properties updated successfully');
    }

    /**
     * Apply edge properties from form
     */
    applyEdgeProperties() {
        // Save current state for undo
        this.saveStateForUndo();
        
        const form = document.getElementById('edge-properties-form');
        if (!form) return;
        
        const edgeId = form.querySelector('[name="edgeId"]').value;
        const edge = this.flow.edges.find(e => e.id === edgeId);
        if (!edge) return;
        
        // Update edge properties
        edge.type = form.querySelector('[name="edgeType"]').value;
        
        // Update the UI
        this.updateFlowVisualization();
        
        // Mark flow as modified
        this.isModified = true;
        this.triggerEvent('flowModified');
        
        // Show success message
        this.showStatusMessage('Properties updated successfully');
    }

    /**
     * Add filter condition
     */
    addFilterCondition() {
        if (!this.currentSelection || this.currentSelection.type !== 'node') return;
        
        const nodeId = this.currentSelection.id;
        const node = this.flow.nodes.find(n => n.id === nodeId);
        if (!node || node.data.label !== 'Filter') return;
        
        // Get condition value
        const conditionInput = document.getElementById('filterCondition');
        if (!conditionInput || !conditionInput.value.trim()) return;
        
        // Add condition
        if (!node.data.conditions) {
            node.data.conditions = [];
        }
        
        node.data.conditions.push(conditionInput.value.trim());
        
        // Clear input
        conditionInput.value = '';
        
        // Update properties panel
        this.updatePropertiesPanel();
        
        // Mark flow as modified
        this.isModified = true;
        this.triggerEvent('flowModified');
    }

    /**
     * Remove filter condition
     * 
     * @param {Number} index - Index of the condition to remove
     */
    removeFilterCondition(index) {
        if (!this.currentSelection || this.currentSelection.type !== 'node') return;
        
        const nodeId = this.currentSelection.id;
        const node = this.flow.nodes.find(n => n.id === nodeId);
        if (!node || node.data.label !== 'Filter' || !node.data.conditions) return;
        
        // Save current state for undo
        this.saveStateForUndo();
        
        // Remove condition
        node.data.conditions.splice(index, 1);
        
        // Update properties panel
        this.updatePropertiesPanel();
        
        // Mark flow as modified
        this.isModified = true;
        this.triggerEvent('flowModified');
    }

    /**
     * Delete selected node or edge
     */
    deleteSelected() {
        if (!this.currentSelection) return;
        
        // Save current state for undo
        this.saveStateForUndo();
        
        const { id, type } = this.currentSelection;
        
        if (type === 'node') {
            // Remove all connected edges
            this.flow.edges = this.flow.edges.filter(e => e.source !== id && e.target !== id);
            
            // Remove the node
            this.flow.nodes = this.flow.nodes.filter(n => n.id !== id);
        } else if (type === 'edge') {
            // Remove the edge
            this.flow.edges = this.flow.edges.filter(e => e.id !== id);
        }
        
        // Clear selection
        this.currentSelection = null;
        this.updatePropertiesPanel();
        
        // Update the UI
        this.updateFlowVisualization();
        
        // Mark flow as modified
        this.isModified = true;
        this.triggerEvent('flowModified');
        
        // Show success message
        this.showStatusMessage('Deleted successfully');
    }

    /**
     * Save the flow
     */
    async saveFlow() {
        try {
            // Check if flow name is set
            if (!this.flowName.trim()) {
                throw new AppError('Flow name cannot be empty', 'VALIDATION_ERROR');
            }
            
            // Prepare save request
            const saveRequest = {
                name: this.flowName,
                description: this.flowDescription,
                flow: this.flow
            };
            
            // If flowId exists, update existing flow, otherwise create new
            let response;
            if (this.flowId) {
                response = await this.apiClient.put(`/api/v1/developer/flows/${this.flowId}`, saveRequest);
            } else {
                response = await this.apiClient.post('/api/v1/developer/flows', saveRequest);
                this.flowId = response.id;
            }
            
            // Mark flow as saved
            this.isModified = false;
            
            // Show success message
            this.showStatusMessage(`Flow "${this.flowName}" saved successfully`);
            
            // Trigger saved event
            this.triggerEvent('flowSaved', response);
            
            return response;
        } catch (error) {
            this.showStatusMessage(`Error saving flow: ${error.message}`, 'error');
            throw new AppError(
                'Failed to save flow', 
                'SAVE_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Load a flow by ID
     * 
     * @param {String} flowId - ID of the flow to load
     */
    async loadFlow(flowId) {
        try {
            // Fetch flow from API
            const response = await this.apiClient.get(`/api/v1/developer/flows/${flowId}`);
            
            // Update flow properties
            this.flowId = response.id;
            this.flowName = response.name;
            this.flowDescription = response.description || '';
            this.flow = response.flow;
            
            // Update title input
            const titleInput = document.querySelector(`#${this.containerId} .flow-title`);
            if (titleInput) {
                titleInput.value = this.flowName;
            }
            
            // Update the UI
            this.updateFlowVisualization();
            
            // Clear selection
            this.currentSelection = null;
            this.updatePropertiesPanel();
            
            // Clear undo/redo stacks
            this.undoStack = [];
            this.redoStack = [];
            
            // Mark flow as not modified
            this.isModified = false;
            
            // Show success message
            this.showStatusMessage(`Flow "${this.flowName}" loaded successfully`);
            
            // Trigger loaded event
            this.triggerEvent('flowLoaded', response);
            
            return response;
        } catch (error) {
            this.showStatusMessage(`Error loading flow: ${error.message}`, 'error');
            throw new AppError(
                'Failed to load flow', 
                'LOAD_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Validate the flow
     */
    validateFlow() {
        try {
            // Get validation container
            const validationStatus = document.querySelector(`#${this.containerId} .validation-status`);
            if (!validationStatus) return;
            
            // Reset validation status
            validationStatus.innerHTML = '';
            validationStatus.className = 'validation-status';
            
            // Check if flow is empty
            if (this.flow.nodes.length === 0) {
                validationStatus.innerHTML = 'Flow is empty. Add some nodes to get started.';
                validationStatus.classList.add('warning');
                return;
            }
            
            // Initialize validation result
            const validation = {
                isValid: true,
                errors: [],
                warnings: []
            };
            
            // Check for disconnected nodes
            const connectedNodeIds = new Set();
            this.flow.edges.forEach(edge => {
                connectedNodeIds.add(edge.source);
                connectedNodeIds.add(edge.target);
            });
            
            const disconnectedNodes = this.flow.nodes.filter(node => !connectedNodeIds.has(node.id));
            if (disconnectedNodes.length > 0) {
                validation.warnings.push(`${disconnectedNodes.length} disconnected node(s) found`);
            }
            
            // Check for cycles
            if (this.hasCycle()) {
                validation.errors.push('Flow contains cycles, which are not allowed');
                validation.isValid = false;
            }
            
            // Check for missing properties
            this.flow.nodes.forEach(node => {
                switch (node.type) {
                    case 'dataSource':
                        if (node.data.label === 'HANA Table' && (!node.data.tableName || !node.data.tableName.trim())) {
                            validation.errors.push(`Table name is required for "${node.data.label}" node`);
                            validation.isValid = false;
                        } else if (node.data.label === 'SQL Query' && (!node.data.query || !node.data.query.trim())) {
                            validation.errors.push(`SQL query is required for "${node.data.label}" node`);
                            validation.isValid = false;
                        }
                        break;
                        
                    case 'transformation':
                        if (node.data.label === 'Join' && (!node.data.leftField || !node.data.rightField)) {
                            validation.errors.push(`Join fields are required for "${node.data.label}" node`);
                            validation.isValid = false;
                        }
                        break;
                        
                    case 'analysis':
                        if (node.data.label === 'Time Series' && (!node.data.timeColumn || !node.data.valueColumn)) {
                            validation.errors.push(`Time and value columns are required for "${node.data.label}" node`);
                            validation.isValid = false;
                        }
                        break;
                        
                    case 'visualization':
                        if ((node.data.label === 'Bar Chart' || node.data.label === 'Line Chart') && 
                            (!node.data.xAxis || !node.data.yAxis)) {
                            validation.errors.push(`X and Y axes are required for "${node.data.label}" node`);
                            validation.isValid = false;
                        }
                        break;
                }
            });
            
            // Display validation results
            if (validation.isValid) {
                if (validation.warnings.length > 0) {
                    validationStatus.innerHTML = `
                        <div class="validation-title">Flow is valid with warnings:</div>
                        <ul>${validation.warnings.map(w => `<li>${w}</li>`).join('')}</ul>
                    `;
                    validationStatus.classList.add('warning');
                } else {
                    validationStatus.innerHTML = 'Flow is valid';
                    validationStatus.classList.add('success');
                }
            } else {
                validationStatus.innerHTML = `
                    <div class="validation-title">Flow validation failed:</div>
                    <ul>${validation.errors.map(e => `<li>${e}</li>`).join('')}</ul>
                    ${validation.warnings.length > 0 ? `
                        <div class="validation-title">Warnings:</div>
                        <ul>${validation.warnings.map(w => `<li>${w}</li>`).join('')}</ul>
                    ` : ''}
                `;
                validationStatus.classList.add('error');
            }
            
            // Trigger validation event
            this.triggerEvent('flowValidated', validation);
            
            return validation;
        } catch (error) {
            this.showStatusMessage(`Error validating flow: ${error.message}`, 'error');
            throw new AppError(
                'Failed to validate flow', 
                'VALIDATION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Check if the flow contains cycles
     * 
     * @returns {Boolean} True if flow contains cycles, false otherwise
     */
    hasCycle() {
        // Create adjacency list
        const graph = {};
        this.flow.nodes.forEach(node => {
            graph[node.id] = [];
        });
        
        this.flow.edges.forEach(edge => {
            if (graph[edge.source]) {
                graph[edge.source].push(edge.target);
            }
        });
        
        // DFS to detect cycles
        const visited = {};
        const recStack = {};
        
        for (const nodeId in graph) {
            if (this.isCyclicUtil(nodeId, visited, recStack, graph)) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * Utility function for cycle detection
     * 
     * @param {String} nodeId - Current node ID
     * @param {Object} visited - Visited nodes
     * @param {Object} recStack - Recursion stack
     * @param {Object} graph - Adjacency list
     * @returns {Boolean} True if cycle detected, false otherwise
     */
    isCyclicUtil(nodeId, visited, recStack, graph) {
        if (!visited[nodeId]) {
            // Mark current node as visited
            visited[nodeId] = true;
            recStack[nodeId] = true;
            
            // Check all adjacent nodes
            for (const neighborId of graph[nodeId]) {
                if (!visited[neighborId] && this.isCyclicUtil(neighborId, visited, recStack, graph)) {
                    return true;
                } else if (recStack[neighborId]) {
                    return true;
                }
            }
        }
        
        // Remove node from recursion stack
        recStack[nodeId] = false;
        return false;
    }

    /**
     * Generate and execute code from the flow
     */
    async generateAndExecuteCode() {
        try {
            // Validate flow first
            const validation = this.validateFlow();
            if (!validation.isValid) {
                this.showStatusMessage('Cannot execute invalid flow. Please fix errors first.', 'error');
                return;
            }
            
            // Generate code
            const generationResponse = await this.aiIntegration.generateCode(this.flow);
            
            // Show generated code
            this.triggerEvent('codeGenerated', generationResponse);
            
            // Execute code
            const executionResponse = await this.aiIntegration.executeCode(
                generationResponse.code,
                generationResponse.language
            );
            
            // Show execution results
            this.triggerEvent('codeExecuted', executionResponse);
            
            // Show success or error message
            if (executionResponse.success) {
                this.showStatusMessage('Flow executed successfully');
            } else {
                this.showStatusMessage(`Execution failed: ${executionResponse.error}`, 'error');
            }
            
            return executionResponse;
        } catch (error) {
            this.showStatusMessage(`Error executing flow: ${error.message}`, 'error');
            throw new AppError(
                'Failed to execute flow', 
                'EXECUTION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Export the flow to JSON
     */
    exportFlow() {
        try {
            // Create export object
            const exportData = {
                name: this.flowName,
                description: this.flowDescription,
                flow: this.flow,
                exported_at: new Date().toISOString()
            };
            
            // Convert to JSON
            const jsonData = JSON.stringify(exportData, null, 2);
            
            // Create download link
            const blob = new Blob([jsonData], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const downloadLink = document.createElement('a');
            downloadLink.href = url;
            downloadLink.download = `${this.flowName.replace(/\s+/g, '_')}_flow.json`;
            downloadLink.click();
            
            // Clean up
            URL.revokeObjectURL(url);
            
            // Show success message
            this.showStatusMessage('Flow exported successfully');
            
            // Trigger export event
            this.triggerEvent('flowExported', exportData);
        } catch (error) {
            this.showStatusMessage(`Error exporting flow: ${error.message}`, 'error');
            throw new AppError(
                'Failed to export flow', 
                'EXPORT_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Save current state for undo
     */
    saveStateForUndo() {
        // Add current state to undo stack
        this.undoStack.push(JSON.stringify(this.flow));
        
        // Limit undo stack size
        if (this.undoStack.length > this.maxUndoStackSize) {
            this.undoStack.shift();
        }
        
        // Clear redo stack
        this.redoStack = [];
    }

    /**
     * Undo the last action
     */
    undo() {
        if (this.undoStack.length === 0) return;
        
        // Save current state for redo
        this.redoStack.push(JSON.stringify(this.flow));
        
        // Restore previous state
        const previousState = this.undoStack.pop();
        this.flow = JSON.parse(previousState);
        
        // Update the UI
        this.updateFlowVisualization();
        
        // Clear selection
        this.currentSelection = null;
        this.updatePropertiesPanel();
        
        // Mark flow as modified
        this.isModified = true;
        
        // Show status message
        this.showStatusMessage('Undo successful');
        
        // Trigger undo event
        this.triggerEvent('undoPerformed');
    }

    /**
     * Redo the last undone action
     */
    redo() {
        if (this.redoStack.length === 0) return;
        
        // Save current state for undo
        this.undoStack.push(JSON.stringify(this.flow));
        
        // Restore next state
        const nextState = this.redoStack.pop();
        this.flow = JSON.parse(nextState);
        
        // Update the UI
        this.updateFlowVisualization();
        
        // Clear selection
        this.currentSelection = null;
        this.updatePropertiesPanel();
        
        // Mark flow as modified
        this.isModified = true;
        
        // Show status message
        this.showStatusMessage('Redo successful');
        
        // Trigger redo event
        this.triggerEvent('redoPerformed');
    }

    /**
     * Show a status message
     * 
     * @param {String} message - Message to show
     * @param {String} type - Message type ('info', 'success', 'warning', 'error')
     */
    showStatusMessage(message, type = 'success') {
        const statusMessage = document.querySelector(`#${this.containerId} .status-message`);
        if (!statusMessage) return;
        
        // Set message text and class
        statusMessage.textContent = message;
        statusMessage.className = 'status-message';
        statusMessage.classList.add(type);
        
        // Clear message after a delay
        setTimeout(() => {
            statusMessage.textContent = '';
            statusMessage.className = 'status-message';
        }, 5000);
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
window.FlowBuilder = FlowBuilder;
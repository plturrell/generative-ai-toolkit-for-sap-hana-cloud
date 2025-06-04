/**
 * Advanced flow validation system for Developer Studio
 * 
 * This module provides comprehensive validation for flow graphs,
 * including cycle detection, type compatibility checking, and
 * required property validation.
 */

/**
 * Validation result object
 * @typedef {Object} ValidationResult
 * @property {boolean} isValid - Whether the flow is valid
 * @property {Object[]} errors - List of validation errors
 * @property {Object[]} warnings - List of validation warnings
 */

/**
 * Node compatibility definitions
 * Defines which node types can connect to which other node types
 */
const NODE_COMPATIBILITY = {
    // Data source nodes
    'HANA Table': {
        outputType: 'data',
        validOutputs: ['Filter', 'Aggregate', 'Join', 'Sort', 'Visualization', 'Export', 'HANA Write', 'Forecast', 'Classification', 'Regression', 'Clustering']
    },
    'HANA SQL': {
        outputType: 'data',
        validOutputs: ['Filter', 'Aggregate', 'Join', 'Sort', 'Visualization', 'Export', 'HANA Write', 'Forecast', 'Classification', 'Regression', 'Clustering']
    },
    'CSV Input': {
        outputType: 'data',
        validOutputs: ['Filter', 'Aggregate', 'Join', 'Sort', 'Visualization', 'Export', 'HANA Write', 'Forecast', 'Classification', 'Regression', 'Clustering']
    },
    
    // Transformation nodes
    'Filter': {
        inputType: 'data',
        outputType: 'data',
        validOutputs: ['Filter', 'Aggregate', 'Join', 'Sort', 'Visualization', 'Export', 'HANA Write', 'Forecast', 'Classification', 'Regression', 'Clustering']
    },
    'Aggregate': {
        inputType: 'data',
        outputType: 'data',
        validOutputs: ['Filter', 'Sort', 'Visualization', 'Export', 'HANA Write']
    },
    'Join': {
        inputType: 'data',
        outputType: 'data',
        validInputs: ['HANA Table', 'HANA SQL', 'CSV Input', 'Filter', 'Aggregate', 'Sort'],
        validOutputs: ['Filter', 'Aggregate', 'Sort', 'Visualization', 'Export', 'HANA Write'],
        minInputs: 2
    },
    'Sort': {
        inputType: 'data',
        outputType: 'data',
        validOutputs: ['Visualization', 'Export', 'HANA Write']
    },
    
    // AI/ML nodes
    'Forecast': {
        inputType: 'data',
        outputType: 'model',
        validOutputs: ['Visualization', 'Export']
    },
    'Classification': {
        inputType: 'data',
        outputType: 'model',
        validOutputs: ['Visualization', 'Export']
    },
    'Regression': {
        inputType: 'data',
        outputType: 'model',
        validOutputs: ['Visualization', 'Export']
    },
    'Clustering': {
        inputType: 'data',
        outputType: 'model',
        validOutputs: ['Visualization', 'Export']
    },
    
    // Output nodes
    'Visualization': {
        inputType: ['data', 'model'],
        outputType: null
    },
    'Export': {
        inputType: ['data', 'model'],
        outputType: null
    },
    'HANA Write': {
        inputType: 'data',
        outputType: null
    }
};

/**
 * Required properties for each node type
 */
const REQUIRED_PROPERTIES = {
    'HANA Table': ['tableName'],
    'HANA SQL': ['query'],
    'CSV Input': ['filePath'],
    'Filter': ['filterCondition'],
    'Aggregate': ['groupByColumns', 'aggregateFunctions'],
    'Join': ['joinType', 'joinCondition'],
    'Sort': ['sortColumns'],
    'Forecast': ['timeColumn', 'valueColumn', 'horizon'],
    'Classification': ['targetColumn', 'featureColumns'],
    'Regression': ['targetColumn', 'featureColumns'],
    'Clustering': ['featureColumns', 'numClusters'],
    'Visualization': ['chartType'],
    'Export': ['format', 'fileName'],
    'HANA Write': ['targetTable']
};

/**
 * Validate a flow graph
 * 
 * @param {Object[]} nodes - Flow nodes
 * @param {Object[]} edges - Flow edges
 * @returns {ValidationResult} Validation result
 */
function validateFlow(nodes, edges) {
    const result = {
        isValid: true,
        errors: [],
        warnings: []
    };
    
    // Run all validation checks
    checkForCycles(nodes, edges, result);
    checkNodeConnections(nodes, edges, result);
    checkRequiredProperties(nodes, result);
    checkConnectivity(nodes, edges, result);
    
    // Update overall validity
    result.isValid = result.errors.length === 0;
    
    return result;
}

/**
 * Check for cycles in the flow graph
 * 
 * @param {Object[]} nodes - Flow nodes
 * @param {Object[]} edges - Flow edges
 * @param {ValidationResult} result - Validation result to update
 */
function checkForCycles(nodes, edges, result) {
    // Build adjacency list
    const adjacencyList = {};
    
    nodes.forEach(node => {
        adjacencyList[node.id] = [];
    });
    
    edges.forEach(edge => {
        if (adjacencyList[edge.source]) {
            adjacencyList[edge.source].push(edge.target);
        }
    });
    
    // Check for cycles using DFS
    const visited = {};
    const recursionStack = {};
    
    // Function to detect cycle using DFS
    function detectCycle(nodeId, path = []) {
        if (recursionStack[nodeId]) {
            // Found a cycle
            const cycle = [...path, nodeId];
            result.errors.push({
                type: 'cycle',
                message: 'Flow contains a cycle',
                details: {
                    cycle: cycle
                },
                nodeIds: cycle
            });
            return true;
        }
        
        if (visited[nodeId]) {
            return false;
        }
        
        visited[nodeId] = true;
        recursionStack[nodeId] = true;
        
        for (const neighbor of adjacencyList[nodeId]) {
            if (detectCycle(neighbor, [...path, nodeId])) {
                return true;
            }
        }
        
        recursionStack[nodeId] = false;
        return false;
    }
    
    // Check each node
    for (const node of nodes) {
        if (!visited[node.id]) {
            detectCycle(node.id);
        }
    }
}

/**
 * Check node connections for type compatibility
 * 
 * @param {Object[]} nodes - Flow nodes
 * @param {Object[]} edges - Flow edges
 * @param {ValidationResult} result - Validation result to update
 */
function checkNodeConnections(nodes, edges, result) {
    // Create a map of nodes by ID
    const nodeMap = {};
    nodes.forEach(node => {
        nodeMap[node.id] = node;
    });
    
    // Check each edge
    edges.forEach(edge => {
        const sourceNode = nodeMap[edge.source];
        const targetNode = nodeMap[edge.target];
        
        if (!sourceNode || !targetNode) {
            result.errors.push({
                type: 'missing_node',
                message: 'Edge references a missing node',
                details: {
                    edge: edge
                },
                edgeId: edge.id
            });
            return;
        }
        
        const sourceType = sourceNode.data.label;
        const targetType = targetNode.data.label;
        
        // Check if source can connect to target
        const sourceNodeDef = NODE_COMPATIBILITY[sourceType];
        const targetNodeDef = NODE_COMPATIBILITY[targetType];
        
        if (!sourceNodeDef || !targetNodeDef) {
            result.warnings.push({
                type: 'unknown_node_type',
                message: `Unknown node type: ${!sourceNodeDef ? sourceType : targetType}`,
                details: {
                    edge: edge
                },
                edgeId: edge.id
            });
            return;
        }
        
        // Check output compatibility
        if (sourceNodeDef.validOutputs && !sourceNodeDef.validOutputs.includes(targetType)) {
            result.errors.push({
                type: 'incompatible_connection',
                message: `Node of type '${sourceType}' cannot connect to node of type '${targetType}'`,
                details: {
                    edge: edge,
                    sourceType: sourceType,
                    targetType: targetType,
                    validOutputs: sourceNodeDef.validOutputs
                },
                edgeId: edge.id,
                nodeIds: [edge.source, edge.target]
            });
        }
        
        // Check input compatibility
        if (targetNodeDef.validInputs && !targetNodeDef.validInputs.includes(sourceType)) {
            result.errors.push({
                type: 'incompatible_connection',
                message: `Node of type '${targetType}' cannot accept input from node of type '${sourceType}'`,
                details: {
                    edge: edge,
                    sourceType: sourceType,
                    targetType: targetType,
                    validInputs: targetNodeDef.validInputs
                },
                edgeId: edge.id,
                nodeIds: [edge.source, edge.target]
            });
        }
        
        // Check type compatibility
        if (sourceNodeDef.outputType && targetNodeDef.inputType) {
            // Handle array of valid input types
            const validInputTypes = Array.isArray(targetNodeDef.inputType) 
                ? targetNodeDef.inputType 
                : [targetNodeDef.inputType];
                
            if (!validInputTypes.includes(sourceNodeDef.outputType)) {
                result.errors.push({
                    type: 'type_mismatch',
                    message: `Type mismatch: '${sourceType}' outputs '${sourceNodeDef.outputType}' but '${targetType}' expects ${validInputTypes.join(' or ')}`,
                    details: {
                        edge: edge,
                        sourceType: sourceType,
                        targetType: targetType,
                        outputType: sourceNodeDef.outputType,
                        inputType: targetNodeDef.inputType
                    },
                    edgeId: edge.id,
                    nodeIds: [edge.source, edge.target]
                });
            }
        }
    });
    
    // Check for nodes with multiple inputs when not allowed
    const inputCounts = {};
    edges.forEach(edge => {
        if (!inputCounts[edge.target]) {
            inputCounts[edge.target] = 0;
        }
        inputCounts[edge.target]++;
    });
    
    // Check each node with inputs
    Object.keys(inputCounts).forEach(nodeId => {
        const node = nodeMap[nodeId];
        if (!node) return;
        
        const nodeType = node.data.label;
        const nodeDef = NODE_COMPATIBILITY[nodeType];
        
        if (!nodeDef) return;
        
        // Check if node can have multiple inputs
        if (inputCounts[nodeId] > 1 && !nodeDef.minInputs) {
            result.errors.push({
                type: 'too_many_inputs',
                message: `Node of type '${nodeType}' cannot have multiple inputs`,
                details: {
                    nodeId: nodeId,
                    nodeType: nodeType,
                    inputCount: inputCounts[nodeId]
                },
                nodeIds: [nodeId]
            });
        }
        
        // Check if node has minimum required inputs
        if (nodeDef.minInputs && inputCounts[nodeId] < nodeDef.minInputs) {
            result.errors.push({
                type: 'too_few_inputs',
                message: `Node of type '${nodeType}' requires at least ${nodeDef.minInputs} inputs`,
                details: {
                    nodeId: nodeId,
                    nodeType: nodeType,
                    inputCount: inputCounts[nodeId],
                    minInputs: nodeDef.minInputs
                },
                nodeIds: [nodeId]
            });
        }
    });
}

/**
 * Check if nodes have all required properties
 * 
 * @param {Object[]} nodes - Flow nodes
 * @param {ValidationResult} result - Validation result to update
 */
function checkRequiredProperties(nodes, result) {
    nodes.forEach(node => {
        const nodeType = node.data.label;
        const requiredProps = REQUIRED_PROPERTIES[nodeType];
        
        if (!requiredProps) return;
        
        const missingProps = [];
        
        // Check each required property
        requiredProps.forEach(prop => {
            if (!node.data[prop]) {
                missingProps.push(prop);
            }
        });
        
        if (missingProps.length > 0) {
            result.errors.push({
                type: 'missing_properties',
                message: `Node of type '${nodeType}' is missing required properties: ${missingProps.join(', ')}`,
                details: {
                    nodeId: node.id,
                    nodeType: nodeType,
                    missingProperties: missingProps
                },
                nodeIds: [node.id]
            });
        }
    });
}

/**
 * Check flow connectivity (all nodes are connected)
 * 
 * @param {Object[]} nodes - Flow nodes
 * @param {Object[]} edges - Flow edges
 * @param {ValidationResult} result - Validation result to update
 */
function checkConnectivity(nodes, edges, result) {
    if (nodes.length <= 1) return;
    
    // Build adjacency list (bidirectional)
    const adjacencyList = {};
    
    nodes.forEach(node => {
        adjacencyList[node.id] = [];
    });
    
    edges.forEach(edge => {
        if (adjacencyList[edge.source]) {
            adjacencyList[edge.source].push(edge.target);
        }
        if (adjacencyList[edge.target]) {
            adjacencyList[edge.target].push(edge.source);
        }
    });
    
    // Run BFS to check connectivity
    const visited = {};
    const queue = [nodes[0].id];
    visited[nodes[0].id] = true;
    
    while (queue.length > 0) {
        const nodeId = queue.shift();
        
        adjacencyList[nodeId].forEach(neighbor => {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        });
    }
    
    // Check if all nodes were visited
    const disconnectedNodes = nodes.filter(node => !visited[node.id]);
    
    if (disconnectedNodes.length > 0) {
        result.warnings.push({
            type: 'disconnected_nodes',
            message: 'Flow contains disconnected nodes',
            details: {
                disconnectedNodes: disconnectedNodes.map(node => ({
                    id: node.id,
                    type: node.data.label
                }))
            },
            nodeIds: disconnectedNodes.map(node => node.id)
        });
    }
}

/**
 * Validate required properties for a specific node
 * 
 * @param {Object} node - Node to validate
 * @returns {Object} Validation result for the node
 */
function validateNodeProperties(node) {
    const result = {
        isValid: true,
        missingProperties: []
    };
    
    const nodeType = node.data.label;
    const requiredProps = REQUIRED_PROPERTIES[nodeType];
    
    if (!requiredProps) return result;
    
    // Check each required property
    requiredProps.forEach(prop => {
        if (!node.data[prop]) {
            result.missingProperties.push(prop);
            result.isValid = false;
        }
    });
    
    return result;
}

/**
 * Apply visual indicators to flow nodes based on validation results
 * 
 * @param {Object[]} nodes - Flow nodes
 * @param {ValidationResult} validationResult - Validation result
 * @returns {Object[]} Updated nodes with validation classes
 */
function applyValidationStyles(nodes, validationResult) {
    // Create a map of error and warning node IDs
    const errorNodeIds = new Set();
    const warningNodeIds = new Set();
    
    validationResult.errors.forEach(error => {
        if (error.nodeIds) {
            error.nodeIds.forEach(id => errorNodeIds.add(id));
        }
    });
    
    validationResult.warnings.forEach(warning => {
        if (warning.nodeIds) {
            warning.nodeIds.forEach(id => warningNodeIds.add(id));
        }
    });
    
    // Apply styles to nodes
    return nodes.map(node => {
        const newNode = { ...node };
        
        // Remove any existing validation classes
        if (newNode.className) {
            newNode.className = newNode.className
                .replace(/\bnode-error\b/, '')
                .replace(/\bnode-warning\b/, '')
                .trim();
        }
        
        // Add error class if node has errors
        if (errorNodeIds.has(node.id)) {
            newNode.className = `${newNode.className || ''} node-error`.trim();
        }
        // Add warning class if node has warnings (and no errors)
        else if (warningNodeIds.has(node.id)) {
            newNode.className = `${newNode.className || ''} node-warning`.trim();
        }
        
        return newNode;
    });
}

/**
 * Get detailed validation error messages for a specific node
 * 
 * @param {string} nodeId - ID of the node
 * @param {ValidationResult} validationResult - Validation result
 * @returns {Object[]} Error and warning messages for the node
 */
function getNodeValidationMessages(nodeId, validationResult) {
    const messages = {
        errors: [],
        warnings: []
    };
    
    // Get errors for this node
    validationResult.errors.forEach(error => {
        if (error.nodeIds && error.nodeIds.includes(nodeId)) {
            messages.errors.push(error.message);
        }
    });
    
    // Get warnings for this node
    validationResult.warnings.forEach(warning => {
        if (warning.nodeIds && warning.nodeIds.includes(nodeId)) {
            messages.warnings.push(warning.message);
        }
    });
    
    return messages;
}

// Export all utilities
window.FlowValidator = {
    validateFlow,
    validateNodeProperties,
    applyValidationStyles,
    getNodeValidationMessages,
    NODE_COMPATIBILITY,
    REQUIRED_PROPERTIES
};
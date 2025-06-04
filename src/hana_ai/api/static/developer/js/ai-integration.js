/**
 * AI Integration Module
 * 
 * Provides integration with SAP AI Core for code generation, 
 * query generation, and other AI-assisted features.
 */

class AIIntegration {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.isGenerating = false;
    }

    /**
     * Generate code from a flow definition
     * 
     * @param {Object} flow - The flow definition with nodes and edges
     * @param {String} language - The target programming language
     * @param {Boolean} includeComments - Whether to include comments in the generated code
     * @returns {Promise<Object>} - The generated code and metadata
     */
    async generateCode(flow, language = 'python', includeComments = true) {
        if (this.isGenerating) {
            throw new AppError(
                'Code generation already in progress', 
                'GENERATION_IN_PROGRESS'
            );
        }

        try {
            this.isGenerating = true;
            
            // Call the API to generate code
            const response = await this.apiClient.post('/api/v1/developer/generate-code', {
                flow,
                language,
                include_comments: includeComments
            });
            
            return response;
        } catch (error) {
            throw new AppError(
                'Failed to generate code', 
                'GENERATION_FAILED',
                { originalError: error }
            );
        } finally {
            this.isGenerating = false;
        }
    }

    /**
     * Generate SQL query from schema information and requirements
     * 
     * @param {Object} queryParams - The query generation parameters
     * @param {Array} queryParams.tables - List of tables to query
     * @param {Array} queryParams.columns - List of columns to select
     * @param {Array} queryParams.filters - Filter conditions
     * @param {Array} queryParams.groupBy - Group by columns
     * @param {Array} queryParams.orderBy - Order by columns and direction
     * @param {String} queryParams.requirements - Natural language requirements
     * @returns {Promise<Object>} - The generated query and metadata
     */
    async generateQuery(queryParams) {
        if (this.isGenerating) {
            throw new AppError(
                'Query generation already in progress', 
                'GENERATION_IN_PROGRESS'
            );
        }

        try {
            this.isGenerating = true;
            
            // Call the API to generate query
            const response = await this.apiClient.post('/api/v1/developer/generate-query', {
                tables: queryParams.tables || [],
                columns: queryParams.columns || [],
                filters: queryParams.filters || null,
                group_by: queryParams.groupBy || null,
                order_by: queryParams.orderBy || null,
                requirements: queryParams.requirements || null
            });
            
            return response;
        } catch (error) {
            throw new AppError(
                'Failed to generate query', 
                'GENERATION_FAILED',
                { originalError: error }
            );
        } finally {
            this.isGenerating = false;
        }
    }

    /**
     * Execute code in a sandboxed environment
     * 
     * @param {String} code - The code to execute
     * @param {String} language - The programming language of the code
     * @param {Number} timeout - Execution timeout in seconds
     * @returns {Promise<Object>} - The execution output and metadata
     */
    async executeCode(code, language = 'python', timeout = 30) {
        try {
            // Call the API to execute code
            const response = await this.apiClient.post('/api/v1/developer/execute-code', {
                code,
                language,
                timeout
            });
            
            return response;
        } catch (error) {
            throw new AppError(
                'Failed to execute code', 
                'EXECUTION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Execute SQL query against the HANA database
     * 
     * @param {String} query - The SQL query to execute
     * @param {Number} maxRows - Maximum number of rows to return
     * @returns {Promise<Object>} - The query results and metadata
     */
    async executeQuery(query, maxRows = 1000) {
        try {
            // Call the API to execute query
            const response = await this.apiClient.post('/api/v1/developer/execute-query', {
                query,
                max_rows: maxRows
            });
            
            return response;
        } catch (error) {
            throw new AppError(
                'Failed to execute query', 
                'QUERY_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Get AI Core status and information
     * 
     * @returns {Promise<Object>} - AI Core status information
     */
    async getAICoreStatus() {
        try {
            const response = await this.apiClient.get('/api/v1/config/test/aicore');
            return response;
        } catch (error) {
            throw new AppError(
                'Failed to get AI Core status', 
                'SERVICE_STATUS_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Get usage metrics for AI services
     * 
     * @returns {Promise<Object>} - Usage metrics for the AI services
     */
    async getUsageMetrics() {
        try {
            // This would be implemented in a real system to track API usage
            // For now, return sample data
            return {
                codeGeneration: {
                    count: 42,
                    tokensUsed: 15780
                },
                queryGeneration: {
                    count: 23,
                    tokensUsed: 8450
                },
                executionTime: {
                    total: 356.8,
                    average: 5.4
                }
            };
        } catch (error) {
            throw new AppError(
                'Failed to get usage metrics', 
                'METRICS_FAILED',
                { originalError: error }
            );
        }
    }
}

// Export the class
window.AIIntegration = AIIntegration;
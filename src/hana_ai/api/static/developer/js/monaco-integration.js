/**
 * Monaco Editor Integration Module
 * 
 * Provides integration with Monaco Editor for code editing capabilities
 * in the Developer Studio.
 */

class MonacoIntegration {
    constructor(containerId) {
        this.containerId = containerId;
        this.editor = null;
        this.language = 'python';
        this.theme = 'vs-dark';
        this.code = '';
        this.isInitialized = false;
        this.eventListeners = {};
    }

    /**
     * Initialize the Monaco editor
     * 
     * @param {Object} options - Editor options
     * @returns {Promise<monaco.editor.IStandaloneCodeEditor>} - The Monaco editor instance
     */
    async initialize(options = {}) {
        try {
            // Wait for Monaco to be loaded
            if (typeof monaco === 'undefined') {
                await this.loadMonacoEditor();
            }
            
            // Get container element
            const container = document.getElementById(this.containerId);
            if (!container) {
                throw new Error(`Container with ID '${this.containerId}' not found`);
            }
            
            // Set language
            this.language = options.language || this.language;
            
            // Set theme
            this.theme = options.theme || this.theme;
            
            // Create editor
            this.editor = monaco.editor.create(container, {
                value: this.code,
                language: this.language,
                theme: this.theme,
                automaticLayout: true,
                minimap: {
                    enabled: options.minimap !== undefined ? options.minimap : true
                },
                scrollBeyondLastLine: false,
                fontSize: options.fontSize || 14,
                lineNumbers: options.lineNumbers !== undefined ? options.lineNumbers : 'on',
                wordWrap: options.wordWrap || 'off',
                renderWhitespace: options.renderWhitespace || 'selection',
                contextmenu: true,
                rulers: options.rulers || [],
                ...options
            });
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Mark as initialized
            this.isInitialized = true;
            
            // Trigger initialized event
            this.triggerEvent('initialized', this.editor);
            
            return this.editor;
        } catch (error) {
            console.error('Failed to initialize Monaco editor:', error);
            throw new AppError(
                'Failed to initialize Monaco editor', 
                'INITIALIZATION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Load Monaco editor dynamically
     * 
     * @returns {Promise<void>}
     */
    loadMonacoEditor() {
        return new Promise((resolve, reject) => {
            // Check if Monaco is already loaded
            if (typeof monaco !== 'undefined') {
                resolve();
                return;
            }
            
            // Create script element for Monaco loader
            const loaderScript = document.createElement('script');
            loaderScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.39.0/min/vs/loader.min.js';
            loaderScript.async = true;
            
            // Handle script load
            loaderScript.onload = () => {
                // Configure RequireJS
                window.require.config({
                    paths: {
                        vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.39.0/min/vs'
                    }
                });
                
                // Load Monaco
                window.require(['vs/editor/editor.main'], () => {
                    // Configure language support
                    this.configureLanguages();
                    
                    resolve();
                });
            };
            
            // Handle load error
            loaderScript.onerror = () => {
                reject(new Error('Failed to load Monaco editor script'));
            };
            
            // Add script to document
            document.head.appendChild(loaderScript);
        });
    }

    /**
     * Configure Monaco languages
     */
    configureLanguages() {
        // Register SQL syntax highlighting
        monaco.languages.registerCompletionItemProvider('sql', {
            provideCompletionItems: (model, position) => {
                const suggestions = [
                    {
                        label: 'SELECT',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'SELECT',
                        detail: 'SQL SELECT statement'
                    },
                    {
                        label: 'FROM',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'FROM',
                        detail: 'SQL FROM clause'
                    },
                    {
                        label: 'WHERE',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'WHERE',
                        detail: 'SQL WHERE clause'
                    },
                    {
                        label: 'GROUP BY',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'GROUP BY',
                        detail: 'SQL GROUP BY clause'
                    },
                    {
                        label: 'ORDER BY',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'ORDER BY',
                        detail: 'SQL ORDER BY clause'
                    },
                    {
                        label: 'JOIN',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'JOIN',
                        detail: 'SQL JOIN clause'
                    },
                    {
                        label: 'LEFT JOIN',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'LEFT JOIN',
                        detail: 'SQL LEFT JOIN clause'
                    },
                    {
                        label: 'INNER JOIN',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'INNER JOIN',
                        detail: 'SQL INNER JOIN clause'
                    },
                    {
                        label: 'HAVING',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'HAVING',
                        detail: 'SQL HAVING clause'
                    },
                    {
                        label: 'COUNT',
                        kind: monaco.languages.CompletionItemKind.Function,
                        insertText: 'COUNT(${1:*})',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'SQL COUNT function'
                    },
                    {
                        label: 'SUM',
                        kind: monaco.languages.CompletionItemKind.Function,
                        insertText: 'SUM(${1:column})',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'SQL SUM function'
                    },
                    {
                        label: 'AVG',
                        kind: monaco.languages.CompletionItemKind.Function,
                        insertText: 'AVG(${1:column})',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'SQL AVG function'
                    }
                ];
                
                return {
                    suggestions
                };
            }
        });
        
        // Register Python syntax highlighting
        monaco.languages.registerCompletionItemProvider('python', {
            provideCompletionItems: (model, position) => {
                const suggestions = [
                    {
                        label: 'def',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'def ${1:function_name}(${2:parameters}):\n\t${3:pass}',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'Define a function'
                    },
                    {
                        label: 'class',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'class ${1:ClassName}:\n\tdef __init__(self, ${2:parameters}):\n\t\t${3:pass}',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'Define a class'
                    },
                    {
                        label: 'if',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'if ${1:condition}:\n\t${2:pass}',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'If statement'
                    },
                    {
                        label: 'for',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'for ${1:item} in ${2:iterable}:\n\t${3:pass}',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'For loop'
                    },
                    {
                        label: 'while',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'while ${1:condition}:\n\t${2:pass}',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'While loop'
                    },
                    {
                        label: 'try',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'try:\n\t${1:pass}\nexcept ${2:Exception} as ${3:e}:\n\t${4:pass}',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'Try-except block'
                    },
                    {
                        label: 'import',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'import ${1:module}',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'Import statement'
                    },
                    {
                        label: 'from',
                        kind: monaco.languages.CompletionItemKind.Keyword,
                        insertText: 'from ${1:module} import ${2:submodule}',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'From import statement'
                    },
                    {
                        label: 'print',
                        kind: monaco.languages.CompletionItemKind.Function,
                        insertText: 'print(${1:object})',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'Print function'
                    }
                ];
                
                // Add HANA ML specific suggestions
                const hanamlSuggestions = [
                    {
                        label: 'ConnectionContext',
                        kind: monaco.languages.CompletionItemKind.Class,
                        insertText: 'ConnectionContext(address="${1:host}", port=${2:port}, user="${3:user}", password="${4:password}")',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'HANA ML ConnectionContext'
                    },
                    {
                        label: 'sql',
                        kind: monaco.languages.CompletionItemKind.Method,
                        insertText: 'sql("${1:query}")',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'Execute SQL query'
                    },
                    {
                        label: 'collect',
                        kind: monaco.languages.CompletionItemKind.Method,
                        insertText: 'collect()',
                        detail: 'Collect results from query'
                    },
                    {
                        label: 'head',
                        kind: monaco.languages.CompletionItemKind.Method,
                        insertText: 'head(${1:n})',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'Get first n rows'
                    },
                    {
                        label: 'create_dataframe_from_pandas',
                        kind: monaco.languages.CompletionItemKind.Function,
                        insertText: 'create_dataframe_from_pandas(connection_context=${1:connection}, pandas_df=${2:df}, table_name="${3:table_name}")',
                        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                        detail: 'Create HANA DataFrame from pandas DataFrame'
                    }
                ];
                
                return {
                    suggestions: [...suggestions, ...hanamlSuggestions]
                };
            }
        });
    }

    /**
     * Set up event listeners for the editor
     */
    setupEventListeners() {
        if (!this.editor) return;
        
        // Content change event
        this.editor.onDidChangeModelContent(() => {
            this.code = this.editor.getValue();
            this.triggerEvent('contentChanged', this.code);
        });
        
        // Cursor position change event
        this.editor.onDidChangeCursorPosition((e) => {
            this.triggerEvent('cursorPositionChanged', e.position);
        });
        
        // Focus event
        this.editor.onDidFocusEditorText(() => {
            this.triggerEvent('focused');
        });
        
        // Blur event
        this.editor.onDidBlurEditorText(() => {
            this.triggerEvent('blurred');
        });
    }

    /**
     * Set the editor's content
     * 
     * @param {String} code - The code to set
     * @param {Boolean} formatted - Whether to format the code after setting
     */
    setCode(code, formatted = false) {
        if (!this.editor) return;
        
        this.code = code;
        this.editor.setValue(code);
        
        if (formatted) {
            this.formatCode();
        }
        
        // Trigger event
        this.triggerEvent('codeSet', code);
    }

    /**
     * Get the editor's content
     * 
     * @returns {String} The current code
     */
    getCode() {
        if (!this.editor) return this.code;
        
        return this.editor.getValue();
    }

    /**
     * Set the editor's language
     * 
     * @param {String} language - The language to set
     */
    setLanguage(language) {
        if (!this.editor) return;
        
        this.language = language;
        monaco.editor.setModelLanguage(this.editor.getModel(), language);
        
        // Trigger event
        this.triggerEvent('languageChanged', language);
    }

    /**
     * Set the editor's theme
     * 
     * @param {String} theme - The theme to set
     */
    setTheme(theme) {
        if (!this.editor) return;
        
        this.theme = theme;
        monaco.editor.setTheme(theme);
        
        // Trigger event
        this.triggerEvent('themeChanged', theme);
    }

    /**
     * Format the code in the editor
     */
    formatCode() {
        if (!this.editor) return;
        
        this.editor.getAction('editor.action.formatDocument').run();
        
        // Trigger event
        this.triggerEvent('codeFormatted');
    }

    /**
     * Insert text at the current cursor position
     * 
     * @param {String} text - The text to insert
     */
    insertText(text) {
        if (!this.editor) return;
        
        const selection = this.editor.getSelection();
        const id = { major: 1, minor: 1 };
        const op = { identifier: id, range: selection, text, forceMoveMarkers: true };
        this.editor.executeEdits('insertText', [op]);
        
        // Trigger event
        this.triggerEvent('textInserted', text);
    }

    /**
     * Set a marker on the editor
     * 
     * @param {Object} marker - The marker to set
     */
    setMarker(marker) {
        if (!this.editor) return;
        
        const model = this.editor.getModel();
        monaco.editor.setModelMarkers(model, 'owner', [marker]);
        
        // Trigger event
        this.triggerEvent('markerSet', marker);
    }

    /**
     * Clear all markers from the editor
     */
    clearMarkers() {
        if (!this.editor) return;
        
        const model = this.editor.getModel();
        monaco.editor.setModelMarkers(model, 'owner', []);
        
        // Trigger event
        this.triggerEvent('markersCleared');
    }

    /**
     * Dispose the editor
     */
    dispose() {
        if (this.editor) {
            this.editor.dispose();
            this.editor = null;
            this.isInitialized = false;
            
            // Trigger event
            this.triggerEvent('disposed');
        }
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
window.MonacoIntegration = MonacoIntegration;
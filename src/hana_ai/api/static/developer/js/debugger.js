/**
 * Integrated Debugger Module
 * 
 * Provides advanced debugging capabilities integrated with the Monaco editor.
 */

class IntegratedDebugger {
    constructor(monacoIntegration, apiClient) {
        this.monacoIntegration = monacoIntegration;
        this.apiClient = apiClient;
        this.debugSession = null;
        this.breakpoints = new Map();
        this.currentStack = [];
        this.variables = {};
        this.isDebugging = false;
        this.isInitialized = false;
        this.debugDecorationsIds = [];
        this.language = 'python';
        this.eventListeners = {};
    }

    /**
     * Initialize the debugger
     * 
     * @returns {Promise<boolean>} - True if initialized successfully
     */
    async initialize() {
        try {
            if (!this.monacoIntegration || !this.monacoIntegration.editor) {
                throw new Error('Monaco editor not initialized');
            }
            
            // Add breakpoint marker provider
            this.registerBreakpointProvider();
            
            // Add gutter for breakpoints
            this.addBreakpointGutter();
            
            // Set up keyboard shortcuts
            this.setupKeyboardShortcuts();
            
            // Create debugging interface
            this.createDebugInterface();
            
            this.isInitialized = true;
            
            // Trigger initialized event
            this.triggerEvent('initialized');
            
            return true;
        } catch (error) {
            console.error('Failed to initialize debugger:', error);
            throw new AppError(
                'Failed to initialize debugger', 
                'DEBUGGER_INITIALIZATION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Register breakpoint provider for Monaco editor
     */
    registerBreakpointProvider() {
        const editor = this.monacoIntegration.editor;
        
        // Listen for gutter clicks to set/unset breakpoints
        editor.onMouseDown((e) => {
            // Check if the click is in the gutter area
            if (e.target.type === monaco.editor.MouseTargetType.GUTTER_GLYPH_MARGIN) {
                const lineNumber = e.target.position.lineNumber;
                this.toggleBreakpoint(lineNumber);
            }
        });
        
        // Update breakpoints when editor content changes
        editor.onDidChangeModelContent(() => {
            this.updateBreakpointDecorations();
        });
    }

    /**
     * Add breakpoint gutter to Monaco editor
     */
    addBreakpointGutter() {
        const editor = this.monacoIntegration.editor;
        
        // Set glyph margin
        editor.updateOptions({
            glyphMargin: true,
            lineNumbersMinChars: 3
        });
    }

    /**
     * Set up keyboard shortcuts for debugging
     */
    setupKeyboardShortcuts() {
        const editor = this.monacoIntegration.editor;
        
        // Add command for starting debugging (F5)
        editor.addCommand(monaco.KeyCode.F5, () => {
            if (this.isDebugging) {
                this.resume();
            } else {
                this.startDebugging();
            }
        });
        
        // Add command for stopping debugging (Shift+F5)
        editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.F5, () => {
            this.stopDebugging();
        });
        
        // Add command for step over (F10)
        editor.addCommand(monaco.KeyCode.F10, () => {
            this.stepOver();
        });
        
        // Add command for step into (F11)
        editor.addCommand(monaco.KeyCode.F11, () => {
            this.stepInto();
        });
        
        // Add command for step out (Shift+F11)
        editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.F11, () => {
            this.stepOut();
        });
    }

    /**
     * Create debugging interface
     */
    createDebugInterface() {
        // Create debug panel if it doesn't exist
        if (!document.getElementById('debug-panel')) {
            // Create debug panel container
            const debugPanel = document.createElement('div');
            debugPanel.id = 'debug-panel';
            debugPanel.className = 'debug-panel';
            
            // Create debug panel header
            const debugHeader = document.createElement('div');
            debugHeader.className = 'debug-panel-header';
            debugHeader.innerHTML = `
                <div class="debug-panel-title">Debugger</div>
                <div class="debug-panel-actions">
                    <button class="debug-action start" title="Start Debugging (F5)">
                        <i class="fas fa-play"></i>
                    </button>
                    <button class="debug-action stop" title="Stop Debugging (Shift+F5)" disabled>
                        <i class="fas fa-stop"></i>
                    </button>
                    <button class="debug-action step-over" title="Step Over (F10)" disabled>
                        <i class="fas fa-arrow-right"></i>
                    </button>
                    <button class="debug-action step-into" title="Step Into (F11)" disabled>
                        <i class="fas fa-arrow-down"></i>
                    </button>
                    <button class="debug-action step-out" title="Step Out (Shift+F11)" disabled>
                        <i class="fas fa-arrow-up"></i>
                    </button>
                    <button class="debug-action close" title="Close Debug Panel">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            
            // Create debug panel content with tabs
            const debugContent = document.createElement('div');
            debugContent.className = 'debug-panel-content';
            debugContent.innerHTML = `
                <div class="debug-tabs">
                    <div class="debug-tab active" data-tab="variables">Variables</div>
                    <div class="debug-tab" data-tab="call-stack">Call Stack</div>
                    <div class="debug-tab" data-tab="breakpoints">Breakpoints</div>
                    <div class="debug-tab" data-tab="console">Console</div>
                </div>
                <div class="debug-content">
                    <div class="debug-tab-content variables active">
                        <div class="debug-variables"></div>
                    </div>
                    <div class="debug-tab-content call-stack">
                        <div class="debug-call-stack"></div>
                    </div>
                    <div class="debug-tab-content breakpoints">
                        <div class="debug-breakpoints"></div>
                    </div>
                    <div class="debug-tab-content console">
                        <div class="debug-console"></div>
                        <div class="debug-console-input-container">
                            <div class="debug-console-prompt">&gt;</div>
                            <input type="text" class="debug-console-input" placeholder="Evaluate expression">
                        </div>
                    </div>
                </div>
            `;
            
            // Append header and content to panel
            debugPanel.appendChild(debugHeader);
            debugPanel.appendChild(debugContent);
            
            // Add panel to editor container
            const editorContainer = document.getElementById('code-editor-container');
            
            // Add after results panel
            const resultsPanel = editorContainer.querySelector('.results-panel');
            if (resultsPanel) {
                editorContainer.insertBefore(debugPanel, resultsPanel.nextSibling);
            } else {
                editorContainer.appendChild(debugPanel);
            }
            
            // Add event listeners
            this.addDebugPanelEventListeners();
        }
    }

    /**
     * Add event listeners to debug panel
     */
    addDebugPanelEventListeners() {
        // Debug actions
        const startButton = document.querySelector('.debug-action.start');
        const stopButton = document.querySelector('.debug-action.stop');
        const stepOverButton = document.querySelector('.debug-action.step-over');
        const stepIntoButton = document.querySelector('.debug-action.step-into');
        const stepOutButton = document.querySelector('.debug-action.step-out');
        const closeButton = document.querySelector('.debug-action.close');
        
        if (startButton) {
            startButton.addEventListener('click', () => {
                if (this.isDebugging) {
                    this.resume();
                } else {
                    this.startDebugging();
                }
            });
        }
        
        if (stopButton) {
            stopButton.addEventListener('click', () => {
                this.stopDebugging();
            });
        }
        
        if (stepOverButton) {
            stepOverButton.addEventListener('click', () => {
                this.stepOver();
            });
        }
        
        if (stepIntoButton) {
            stepIntoButton.addEventListener('click', () => {
                this.stepInto();
            });
        }
        
        if (stepOutButton) {
            stepOutButton.addEventListener('click', () => {
                this.stepOut();
            });
        }
        
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                this.hideDebugPanel();
            });
        }
        
        // Debug tabs
        const debugTabs = document.querySelectorAll('.debug-tab');
        const tabContents = document.querySelectorAll('.debug-tab-content');
        
        debugTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                debugTabs.forEach(t => t.classList.remove('active'));
                
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
        
        // Console input
        const consoleInput = document.querySelector('.debug-console-input');
        if (consoleInput) {
            consoleInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    this.evaluateExpression(consoleInput.value);
                    consoleInput.value = '';
                }
            });
        }
    }

    /**
     * Show debug panel
     */
    showDebugPanel() {
        const debugPanel = document.getElementById('debug-panel');
        if (debugPanel) {
            debugPanel.style.display = 'flex';
            this.monacoIntegration.editor.layout();
        }
    }

    /**
     * Hide debug panel
     */
    hideDebugPanel() {
        const debugPanel = document.getElementById('debug-panel');
        if (debugPanel) {
            debugPanel.style.display = 'none';
            this.monacoIntegration.editor.layout();
        }
    }

    /**
     * Toggle breakpoint at the specified line
     * 
     * @param {number} lineNumber - Line number for the breakpoint
     */
    toggleBreakpoint(lineNumber) {
        const editor = this.monacoIntegration.editor;
        const model = editor.getModel();
        
        // Check if line has content
        const lineContent = model.getLineContent(lineNumber).trim();
        if (!lineContent) {
            return;
        }
        
        // Check if breakpoint already exists
        if (this.breakpoints.has(lineNumber)) {
            // Remove breakpoint
            this.breakpoints.delete(lineNumber);
        } else {
            // Add breakpoint
            this.breakpoints.set(lineNumber, {
                id: `bp_${Date.now()}_${lineNumber}`,
                line: lineNumber,
                enabled: true,
                condition: null
            });
        }
        
        // Update breakpoint decorations
        this.updateBreakpointDecorations();
        
        // Update breakpoints list
        this.updateBreakpointsList();
        
        // Trigger breakpoint toggled event
        this.triggerEvent('breakpointToggled', {
            lineNumber,
            active: this.breakpoints.has(lineNumber)
        });
    }

    /**
     * Update breakpoint decorations in Monaco editor
     */
    updateBreakpointDecorations() {
        const editor = this.monacoIntegration.editor;
        
        // Remove existing decorations
        this.debugDecorationsIds = editor.deltaDecorations(this.debugDecorationsIds, []);
        
        // Add decorations for active breakpoints
        const breakpointDecorations = [];
        this.breakpoints.forEach((bp, lineNumber) => {
            if (bp.enabled) {
                breakpointDecorations.push({
                    range: new monaco.Range(lineNumber, 1, lineNumber, 1),
                    options: {
                        isWholeLine: false,
                        glyphMarginClassName: 'debug-breakpoint-glyph'
                    }
                });
            } else {
                breakpointDecorations.push({
                    range: new monaco.Range(lineNumber, 1, lineNumber, 1),
                    options: {
                        isWholeLine: false,
                        glyphMarginClassName: 'debug-breakpoint-glyph-disabled'
                    }
                });
            }
        });
        
        // Add current execution position if debugging
        if (this.isDebugging && this.currentStack.length > 0) {
            const currentFrame = this.currentStack[0];
            breakpointDecorations.push({
                range: new monaco.Range(currentFrame.line, 1, currentFrame.line, 1),
                options: {
                    isWholeLine: true,
                    className: 'debug-current-line',
                    glyphMarginClassName: 'debug-current-position-glyph'
                }
            });
        }
        
        // Apply decorations
        this.debugDecorationsIds = editor.deltaDecorations([], breakpointDecorations);
    }

    /**
     * Update breakpoints list in debug panel
     */
    updateBreakpointsList() {
        const breakpointsContainer = document.querySelector('.debug-breakpoints');
        if (!breakpointsContainer) return;
        
        // Clear container
        breakpointsContainer.innerHTML = '';
        
        // Add breakpoints
        if (this.breakpoints.size === 0) {
            breakpointsContainer.innerHTML = '<div class="debug-empty-message">No breakpoints set</div>';
            return;
        }
        
        const breakpointsList = document.createElement('div');
        breakpointsList.className = 'debug-list';
        
        this.breakpoints.forEach((bp, line) => {
            const breakpointItem = document.createElement('div');
            breakpointItem.className = 'debug-list-item';
            breakpointItem.innerHTML = `
                <div class="debug-list-item-checkbox">
                    <input type="checkbox" ${bp.enabled ? 'checked' : ''} data-line="${line}">
                </div>
                <div class="debug-list-item-text">
                    Line ${line}
                </div>
                <div class="debug-list-item-actions">
                    <button class="debug-list-item-action delete" data-line="${line}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
            breakpointsList.appendChild(breakpointItem);
        });
        
        breakpointsContainer.appendChild(breakpointsList);
        
        // Add event listeners
        const checkboxes = breakpointsContainer.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const line = parseInt(e.target.getAttribute('data-line'));
                const bp = this.breakpoints.get(line);
                if (bp) {
                    bp.enabled = e.target.checked;
                    this.updateBreakpointDecorations();
                }
            });
        });
        
        const deleteButtons = breakpointsContainer.querySelectorAll('.debug-list-item-action.delete');
        deleteButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const line = parseInt(e.target.closest('button').getAttribute('data-line'));
                this.breakpoints.delete(line);
                this.updateBreakpointDecorations();
                this.updateBreakpointsList();
            });
        });
    }

    /**
     * Start debugging session
     */
    async startDebugging() {
        try {
            if (this.isDebugging) {
                return;
            }
            
            // Show debug panel
            this.showDebugPanel();
            
            // Get code from editor
            const code = this.monacoIntegration.getCode();
            
            // Get language
            this.language = this.monacoIntegration.language;
            
            // Prepare breakpoints
            const breakpointsArray = Array.from(this.breakpoints.entries()).map(([line, bp]) => ({
                line,
                enabled: bp.enabled,
                condition: bp.condition
            }));
            
            // Start debug session
            const response = await this.apiClient.post('/api/v1/developer/debug/start', {
                code,
                language: this.language,
                breakpoints: breakpointsArray
            });
            
            // Store debug session ID
            this.debugSession = response.session_id;
            
            // Set debugging state
            this.isDebugging = true;
            
            // Update UI
            this.updateDebugButtonStates();
            
            // Log to console
            this.logToConsole('Debugging session started');
            
            // Check if stopped at a breakpoint
            if (response.status === 'paused') {
                // Update current stack
                this.currentStack = response.stack;
                
                // Update variables
                this.variables = response.variables;
                
                // Update UI
                this.updateStackTrace();
                this.updateVariables();
                this.updateBreakpointDecorations();
                
                // Scroll to current position
                this.scrollToCurrentPosition();
                
                // Log to console
                this.logToConsole(`Paused at line ${this.currentStack[0].line}`);
            }
            
            // Trigger debug started event
            this.triggerEvent('debugStarted', response);
            
            return response;
        } catch (error) {
            console.error('Failed to start debugging:', error);
            this.logToConsole(`Error: ${error.message}`, 'error');
            
            // Reset debugging state
            this.isDebugging = false;
            this.debugSession = null;
            
            // Update UI
            this.updateDebugButtonStates();
            
            throw new AppError(
                'Failed to start debugging', 
                'DEBUG_START_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Stop debugging session
     */
    async stopDebugging() {
        try {
            if (!this.isDebugging) {
                return;
            }
            
            // Stop debug session
            await this.apiClient.post(`/api/v1/developer/debug/${this.debugSession}/stop`);
            
            // Reset debugging state
            this.isDebugging = false;
            this.debugSession = null;
            this.currentStack = [];
            this.variables = {};
            
            // Update UI
            this.updateDebugButtonStates();
            this.updateBreakpointDecorations();
            this.updateStackTrace();
            this.updateVariables();
            
            // Log to console
            this.logToConsole('Debugging session stopped');
            
            // Trigger debug stopped event
            this.triggerEvent('debugStopped');
        } catch (error) {
            console.error('Failed to stop debugging:', error);
            this.logToConsole(`Error: ${error.message}`, 'error');
            
            // Force reset debugging state
            this.isDebugging = false;
            this.debugSession = null;
            this.currentStack = [];
            this.variables = {};
            
            // Update UI
            this.updateDebugButtonStates();
            this.updateBreakpointDecorations();
            
            throw new AppError(
                'Failed to stop debugging', 
                'DEBUG_STOP_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Resume execution
     */
    async resume() {
        try {
            if (!this.isDebugging || !this.debugSession) {
                return;
            }
            
            // Resume debug session
            const response = await this.apiClient.post(`/api/v1/developer/debug/${this.debugSession}/resume`);
            
            // Log to console
            this.logToConsole('Execution resumed');
            
            // Check if stopped at a breakpoint
            if (response.status === 'paused') {
                // Update current stack
                this.currentStack = response.stack;
                
                // Update variables
                this.variables = response.variables;
                
                // Update UI
                this.updateStackTrace();
                this.updateVariables();
                this.updateBreakpointDecorations();
                
                // Scroll to current position
                this.scrollToCurrentPosition();
                
                // Log to console
                this.logToConsole(`Paused at line ${this.currentStack[0].line}`);
            } else if (response.status === 'terminated') {
                // Debugging finished
                this.isDebugging = false;
                this.debugSession = null;
                this.currentStack = [];
                this.variables = {};
                
                // Update UI
                this.updateDebugButtonStates();
                this.updateBreakpointDecorations();
                this.updateStackTrace();
                this.updateVariables();
                
                // Log to console
                this.logToConsole('Debugging session finished');
                
                // Show output
                if (response.output) {
                    const resultPanel = document.querySelector('.results-panel-content');
                    if (resultPanel) {
                        resultPanel.textContent = response.output;
                    }
                }
                
                // Trigger debug stopped event
                this.triggerEvent('debugStopped');
            }
            
            // Trigger debug resumed event
            this.triggerEvent('debugResumed', response);
            
            return response;
        } catch (error) {
            console.error('Failed to resume debugging:', error);
            this.logToConsole(`Error: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to resume debugging', 
                'DEBUG_RESUME_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Step over current line
     */
    async stepOver() {
        try {
            if (!this.isDebugging || !this.debugSession) {
                return;
            }
            
            // Step over
            const response = await this.apiClient.post(`/api/v1/developer/debug/${this.debugSession}/step-over`);
            
            // Log to console
            this.logToConsole('Step over');
            
            // Check if stopped at a breakpoint
            if (response.status === 'paused') {
                // Update current stack
                this.currentStack = response.stack;
                
                // Update variables
                this.variables = response.variables;
                
                // Update UI
                this.updateStackTrace();
                this.updateVariables();
                this.updateBreakpointDecorations();
                
                // Scroll to current position
                this.scrollToCurrentPosition();
                
                // Log to console
                this.logToConsole(`Paused at line ${this.currentStack[0].line}`);
            } else if (response.status === 'terminated') {
                // Debugging finished
                this.isDebugging = false;
                this.debugSession = null;
                this.currentStack = [];
                this.variables = {};
                
                // Update UI
                this.updateDebugButtonStates();
                this.updateBreakpointDecorations();
                this.updateStackTrace();
                this.updateVariables();
                
                // Log to console
                this.logToConsole('Debugging session finished');
                
                // Show output
                if (response.output) {
                    const resultPanel = document.querySelector('.results-panel-content');
                    if (resultPanel) {
                        resultPanel.textContent = response.output;
                    }
                }
                
                // Trigger debug stopped event
                this.triggerEvent('debugStopped');
            }
            
            // Trigger debug step over event
            this.triggerEvent('debugStepOver', response);
            
            return response;
        } catch (error) {
            console.error('Failed to step over:', error);
            this.logToConsole(`Error: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to step over', 
                'DEBUG_STEP_OVER_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Step into function
     */
    async stepInto() {
        try {
            if (!this.isDebugging || !this.debugSession) {
                return;
            }
            
            // Step into
            const response = await this.apiClient.post(`/api/v1/developer/debug/${this.debugSession}/step-into`);
            
            // Log to console
            this.logToConsole('Step into');
            
            // Check if stopped at a breakpoint
            if (response.status === 'paused') {
                // Update current stack
                this.currentStack = response.stack;
                
                // Update variables
                this.variables = response.variables;
                
                // Update UI
                this.updateStackTrace();
                this.updateVariables();
                this.updateBreakpointDecorations();
                
                // Scroll to current position
                this.scrollToCurrentPosition();
                
                // Log to console
                this.logToConsole(`Paused at line ${this.currentStack[0].line}`);
            } else if (response.status === 'terminated') {
                // Debugging finished
                this.isDebugging = false;
                this.debugSession = null;
                this.currentStack = [];
                this.variables = {};
                
                // Update UI
                this.updateDebugButtonStates();
                this.updateBreakpointDecorations();
                this.updateStackTrace();
                this.updateVariables();
                
                // Log to console
                this.logToConsole('Debugging session finished');
                
                // Show output
                if (response.output) {
                    const resultPanel = document.querySelector('.results-panel-content');
                    if (resultPanel) {
                        resultPanel.textContent = response.output;
                    }
                }
                
                // Trigger debug stopped event
                this.triggerEvent('debugStopped');
            }
            
            // Trigger debug step into event
            this.triggerEvent('debugStepInto', response);
            
            return response;
        } catch (error) {
            console.error('Failed to step into:', error);
            this.logToConsole(`Error: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to step into', 
                'DEBUG_STEP_INTO_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Step out of function
     */
    async stepOut() {
        try {
            if (!this.isDebugging || !this.debugSession) {
                return;
            }
            
            // Step out
            const response = await this.apiClient.post(`/api/v1/developer/debug/${this.debugSession}/step-out`);
            
            // Log to console
            this.logToConsole('Step out');
            
            // Check if stopped at a breakpoint
            if (response.status === 'paused') {
                // Update current stack
                this.currentStack = response.stack;
                
                // Update variables
                this.variables = response.variables;
                
                // Update UI
                this.updateStackTrace();
                this.updateVariables();
                this.updateBreakpointDecorations();
                
                // Scroll to current position
                this.scrollToCurrentPosition();
                
                // Log to console
                this.logToConsole(`Paused at line ${this.currentStack[0].line}`);
            } else if (response.status === 'terminated') {
                // Debugging finished
                this.isDebugging = false;
                this.debugSession = null;
                this.currentStack = [];
                this.variables = {};
                
                // Update UI
                this.updateDebugButtonStates();
                this.updateBreakpointDecorations();
                this.updateStackTrace();
                this.updateVariables();
                
                // Log to console
                this.logToConsole('Debugging session finished');
                
                // Show output
                if (response.output) {
                    const resultPanel = document.querySelector('.results-panel-content');
                    if (resultPanel) {
                        resultPanel.textContent = response.output;
                    }
                }
                
                // Trigger debug stopped event
                this.triggerEvent('debugStopped');
            }
            
            // Trigger debug step out event
            this.triggerEvent('debugStepOut', response);
            
            return response;
        } catch (error) {
            console.error('Failed to step out:', error);
            this.logToConsole(`Error: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to step out', 
                'DEBUG_STEP_OUT_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Evaluate expression in current context
     * 
     * @param {string} expression - Expression to evaluate
     */
    async evaluateExpression(expression) {
        try {
            if (!this.isDebugging || !this.debugSession) {
                this.logToConsole('No active debugging session', 'error');
                return;
            }
            
            // Log expression to console
            this.logToConsole(`> ${expression}`, 'input');
            
            // Evaluate expression
            const response = await this.apiClient.post(`/api/v1/developer/debug/${this.debugSession}/evaluate`, {
                expression
            });
            
            // Log result to console
            this.logToConsole(response.result);
            
            // Trigger expression evaluated event
            this.triggerEvent('expressionEvaluated', {
                expression,
                result: response.result
            });
            
            return response;
        } catch (error) {
            console.error('Failed to evaluate expression:', error);
            this.logToConsole(`Error: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to evaluate expression', 
                'DEBUG_EVALUATE_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Update debug button states
     */
    updateDebugButtonStates() {
        const startButton = document.querySelector('.debug-action.start');
        const stopButton = document.querySelector('.debug-action.stop');
        const stepOverButton = document.querySelector('.debug-action.step-over');
        const stepIntoButton = document.querySelector('.debug-action.step-into');
        const stepOutButton = document.querySelector('.debug-action.step-out');
        
        if (this.isDebugging) {
            // Update start button to show pause/resume
            if (startButton) {
                startButton.innerHTML = '<i class="fas fa-play"></i>';
                startButton.title = 'Resume (F5)';
            }
            
            // Enable stop button
            if (stopButton) {
                stopButton.disabled = false;
            }
            
            // Enable step buttons
            if (stepOverButton) {
                stepOverButton.disabled = false;
            }
            
            if (stepIntoButton) {
                stepIntoButton.disabled = false;
            }
            
            if (stepOutButton) {
                stepOutButton.disabled = false;
            }
        } else {
            // Update start button to show start
            if (startButton) {
                startButton.innerHTML = '<i class="fas fa-play"></i>';
                startButton.title = 'Start Debugging (F5)';
            }
            
            // Disable stop button
            if (stopButton) {
                stopButton.disabled = true;
            }
            
            // Disable step buttons
            if (stepOverButton) {
                stepOverButton.disabled = true;
            }
            
            if (stepIntoButton) {
                stepIntoButton.disabled = true;
            }
            
            if (stepOutButton) {
                stepOutButton.disabled = true;
            }
        }
    }

    /**
     * Update stack trace display
     */
    updateStackTrace() {
        const stackContainer = document.querySelector('.debug-call-stack');
        if (!stackContainer) return;
        
        // Clear container
        stackContainer.innerHTML = '';
        
        // Add stack frames
        if (!this.currentStack || this.currentStack.length === 0) {
            stackContainer.innerHTML = '<div class="debug-empty-message">No stack frames</div>';
            return;
        }
        
        const stackList = document.createElement('div');
        stackList.className = 'debug-list';
        
        this.currentStack.forEach((frame, index) => {
            const stackItem = document.createElement('div');
            stackItem.className = 'debug-list-item';
            stackItem.innerHTML = `
                <div class="debug-list-item-text">
                    <div class="debug-frame-name">${frame.name || '<anonymous>'}</div>
                    <div class="debug-frame-location">Line ${frame.line}</div>
                </div>
            `;
            
            // Mark active frame
            if (index === 0) {
                stackItem.classList.add('active');
            }
            
            // Add click handler to navigate to frame
            stackItem.addEventListener('click', () => {
                this.navigateToLine(frame.line);
            });
            
            stackList.appendChild(stackItem);
        });
        
        stackContainer.appendChild(stackList);
    }

    /**
     * Update variables display
     */
    updateVariables() {
        const variablesContainer = document.querySelector('.debug-variables');
        if (!variablesContainer) return;
        
        // Clear container
        variablesContainer.innerHTML = '';
        
        // Add variables
        if (!this.variables || Object.keys(this.variables).length === 0) {
            variablesContainer.innerHTML = '<div class="debug-empty-message">No variables</div>';
            return;
        }
        
        const variablesList = document.createElement('div');
        variablesList.className = 'debug-variables-list';
        
        // Sort variables alphabetically
        const sortedVars = Object.entries(this.variables).sort((a, b) => a[0].localeCompare(b[0]));
        
        sortedVars.forEach(([name, value]) => {
            const varItem = document.createElement('div');
            varItem.className = 'debug-variable-item';
            
            // Format value based on type
            let formattedValue = String(value);
            let valueType = typeof value;
            
            if (value === null) {
                formattedValue = 'null';
                valueType = 'null';
            } else if (Array.isArray(value)) {
                formattedValue = `Array(${value.length})`;
                valueType = 'array';
            } else if (valueType === 'object') {
                formattedValue = 'Object';
                valueType = 'object';
            } else if (valueType === 'string') {
                formattedValue = `"${value}"`;
            }
            
            varItem.innerHTML = `
                <div class="debug-variable-name">${name}</div>
                <div class="debug-variable-value" title="${formattedValue}">${formattedValue}</div>
                <div class="debug-variable-type">${valueType}</div>
            `;
            
            variablesList.appendChild(varItem);
        });
        
        variablesContainer.appendChild(variablesList);
    }

    /**
     * Navigate to specific line in editor
     * 
     * @param {number} line - Line number
     */
    navigateToLine(line) {
        const editor = this.monacoIntegration.editor;
        
        // Set position
        editor.revealLineInCenter(line);
        
        // Set cursor position
        editor.setPosition({ lineNumber: line, column: 1 });
    }

    /**
     * Scroll to current position in editor
     */
    scrollToCurrentPosition() {
        if (this.currentStack && this.currentStack.length > 0) {
            this.navigateToLine(this.currentStack[0].line);
        }
    }

    /**
     * Log message to debug console
     * 
     * @param {string} message - Message to log
     * @param {string} type - Message type (log, error, input)
     */
    logToConsole(message, type = 'log') {
        const consoleContainer = document.querySelector('.debug-console');
        if (!consoleContainer) return;
        
        const logItem = document.createElement('div');
        logItem.className = `debug-console-log debug-console-${type}`;
        logItem.textContent = message;
        
        consoleContainer.appendChild(logItem);
        
        // Scroll to bottom
        consoleContainer.scrollTop = consoleContainer.scrollHeight;
    }

    /**
     * Clear debug console
     */
    clearConsole() {
        const consoleContainer = document.querySelector('.debug-console');
        if (consoleContainer) {
            consoleContainer.innerHTML = '';
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
window.IntegratedDebugger = IntegratedDebugger;
/**
 * Collaboration Module
 * 
 * Provides real-time collaborative editing capabilities for code and flows.
 */

class Collaboration {
    constructor(apiClient, monacoIntegration, flowBuilder) {
        this.apiClient = apiClient;
        this.monacoIntegration = monacoIntegration;
        this.flowBuilder = flowBuilder;
        this.socket = null;
        this.roomId = null;
        this.userId = this.generateUserId();
        this.username = 'User-' + this.userId.substring(0, 5);
        this.collaborators = [];
        this.cursorPositions = {};
        this.localCursors = {};
        this.editingMode = 'code'; // 'code' or 'flow'
        this.isConnected = false;
        this.isInitialized = false;
        this.eventListeners = {};
        this.pendingOperations = [];
        this.lastSentOperation = null;
        this.awaitingAck = false;
        this.lastProcessedOperationId = null;
        this.clientColors = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 2000;
    }

    /**
     * Initialize the collaboration module
     * 
     * @returns {Promise<boolean>} - True if initialized successfully
     */
    async initialize() {
        try {
            // Ask for username
            this.username = prompt('Enter your display name for collaboration:', this.username) || this.username;
            
            // Create collaboration UI
            this.createCollaborationUI();
            
            // Initialize event listeners
            this.setupEventListeners();
            
            // Set initialized flag
            this.isInitialized = true;
            
            // Trigger initialized event
            this.triggerEvent('initialized');
            
            return true;
        } catch (error) {
            console.error('Failed to initialize collaboration:', error);
            throw new AppError(
                'Failed to initialize collaboration', 
                'COLLABORATION_INITIALIZATION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Generate a unique user ID
     * 
     * @returns {string} - Generated user ID
     */
    generateUserId() {
        return 'user_' + Math.random().toString(36).substring(2, 15);
    }

    /**
     * Create collaboration UI
     */
    createCollaborationUI() {
        // Create collaboration panel
        const collaborationPanel = document.createElement('div');
        collaborationPanel.id = 'collaboration-panel';
        collaborationPanel.className = 'collaboration-panel';
        
        // Create collaboration panel header
        const collaborationHeader = document.createElement('div');
        collaborationHeader.className = 'collaboration-panel-header';
        collaborationHeader.innerHTML = `
            <div class="collaboration-panel-title">
                <i class="fas fa-users"></i>
                Collaboration
            </div>
            <div class="collaboration-status">
                <span class="collaboration-status-indicator disconnected"></span>
                <span class="collaboration-status-text">Disconnected</span>
            </div>
            <div class="collaboration-panel-actions">
                <button class="collaboration-action share" title="Share Session">
                    <i class="fas fa-share-alt"></i>
                </button>
                <button class="collaboration-action settings" title="Collaboration Settings">
                    <i class="fas fa-cog"></i>
                </button>
                <button class="collaboration-action close" title="Close Collaboration Panel">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Create collaboration panel content
        const collaborationContent = document.createElement('div');
        collaborationContent.className = 'collaboration-panel-content';
        collaborationContent.innerHTML = `
            <div class="collaboration-section">
                <div class="collaboration-section-header">
                    <div class="collaboration-section-title">Session</div>
                    <div class="collaboration-section-actions">
                        <button class="collaboration-section-action new-session" title="Create New Session">
                            <i class="fas fa-plus"></i>
                        </button>
                        <button class="collaboration-section-action join-session" title="Join Session">
                            <i class="fas fa-sign-in-alt"></i>
                        </button>
                        <button class="collaboration-section-action leave-session" title="Leave Session" disabled>
                            <i class="fas fa-sign-out-alt"></i>
                        </button>
                    </div>
                </div>
                <div class="collaboration-session">
                    <div class="collaboration-session-info">
                        <div class="collaboration-session-id">No active session</div>
                        <div class="collaboration-session-mode"></div>
                    </div>
                </div>
            </div>
            
            <div class="collaboration-section">
                <div class="collaboration-section-header">
                    <div class="collaboration-section-title">Collaborators</div>
                </div>
                <div class="collaboration-collaborators">
                    <div class="collaboration-empty-message">No active collaborators</div>
                </div>
            </div>
            
            <div class="collaboration-section">
                <div class="collaboration-section-header">
                    <div class="collaboration-section-title">Chat</div>
                </div>
                <div class="collaboration-chat">
                    <div class="collaboration-chat-messages">
                        <div class="collaboration-system-message">Start a session to chat with collaborators</div>
                    </div>
                    <div class="collaboration-chat-input-container">
                        <input type="text" class="collaboration-chat-input" placeholder="Type a message..." disabled>
                        <button class="collaboration-chat-send" disabled>
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Add header and content to panel
        collaborationPanel.appendChild(collaborationHeader);
        collaborationPanel.appendChild(collaborationContent);
        
        // Add panel to the workspace
        const workspace = document.querySelector('.developer-workspace');
        if (workspace) {
            workspace.appendChild(collaborationPanel);
        }
        
        // Add event listeners
        this.addCollaborationPanelEventListeners();
    }

    /**
     * Add event listeners to collaboration panel
     */
    addCollaborationPanelEventListeners() {
        // Panel header actions
        const shareButton = document.querySelector('.collaboration-action.share');
        const settingsButton = document.querySelector('.collaboration-action.settings');
        const closeButton = document.querySelector('.collaboration-action.close');
        
        if (shareButton) {
            shareButton.addEventListener('click', () => {
                this.shareSession();
            });
        }
        
        if (settingsButton) {
            settingsButton.addEventListener('click', () => {
                this.showSettings();
            });
        }
        
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                this.hideCollaborationPanel();
            });
        }
        
        // Session actions
        const newSessionButton = document.querySelector('.collaboration-section-action.new-session');
        const joinSessionButton = document.querySelector('.collaboration-section-action.join-session');
        const leaveSessionButton = document.querySelector('.collaboration-section-action.leave-session');
        
        if (newSessionButton) {
            newSessionButton.addEventListener('click', () => {
                this.createNewSession();
            });
        }
        
        if (joinSessionButton) {
            joinSessionButton.addEventListener('click', () => {
                this.showJoinSessionDialog();
            });
        }
        
        if (leaveSessionButton) {
            leaveSessionButton.addEventListener('click', () => {
                this.leaveSession();
            });
        }
        
        // Chat input
        const chatInput = document.querySelector('.collaboration-chat-input');
        const chatSendButton = document.querySelector('.collaboration-chat-send');
        
        if (chatInput) {
            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    this.sendChatMessage(chatInput.value);
                    chatInput.value = '';
                }
            });
        }
        
        if (chatSendButton) {
            chatSendButton.addEventListener('click', () => {
                const chatInput = document.querySelector('.collaboration-chat-input');
                this.sendChatMessage(chatInput.value);
                chatInput.value = '';
            });
        }
    }

    /**
     * Set up event listeners for editor and flow builder
     */
    setupEventListeners() {
        // Editor event listeners
        if (this.monacoIntegration && this.monacoIntegration.editor) {
            // Listen for cursor position changes
            this.monacoIntegration.editor.onDidChangeCursorPosition((e) => {
                if (this.isConnected && this.editingMode === 'code') {
                    this.sendCursorPosition(e.position);
                }
            });
            
            // Listen for content changes
            this.monacoIntegration.editor.onDidChangeModelContent((e) => {
                if (this.isConnected && this.editingMode === 'code') {
                    // Convert Monaco change event to operation
                    const operations = this.monacoChangesToOperations(e.changes);
                    this.sendOperations(operations);
                }
            });
        }
        
        // Flow builder event listeners
        if (this.flowBuilder) {
            // Listen for flow changes
            this.flowBuilder.addEventListener('flowModified', () => {
                if (this.isConnected && this.editingMode === 'flow') {
                    this.sendFlowUpdate(this.flowBuilder.flow);
                }
            });
        }
    }

    /**
     * Show collaboration panel
     */
    showCollaborationPanel() {
        const collaborationPanel = document.getElementById('collaboration-panel');
        if (collaborationPanel) {
            collaborationPanel.style.display = 'flex';
        }
    }

    /**
     * Hide collaboration panel
     */
    hideCollaborationPanel() {
        const collaborationPanel = document.getElementById('collaboration-panel');
        if (collaborationPanel) {
            collaborationPanel.style.display = 'none';
        }
    }

    /**
     * Toggle collaboration panel
     */
    toggleCollaborationPanel() {
        const collaborationPanel = document.getElementById('collaboration-panel');
        if (collaborationPanel) {
            if (collaborationPanel.style.display === 'none') {
                this.showCollaborationPanel();
            } else {
                this.hideCollaborationPanel();
            }
        } else {
            this.createCollaborationUI();
            this.showCollaborationPanel();
        }
    }

    /**
     * Create a new collaboration session
     */
    async createNewSession() {
        try {
            // Ask for editing mode
            const mode = await this.showEditingModeDialog();
            
            if (!mode) {
                return; // User cancelled
            }
            
            this.editingMode = mode;
            
            // Create session on server
            const response = await this.apiClient.post('/api/v1/developer/collaboration/session', {
                mode,
                username: this.username
            });
            
            // Store room ID
            this.roomId = response.roomId;
            
            // Update UI
            this.updateSessionInfo();
            
            // Connect to WebSocket
            await this.connectWebSocket();
            
            // Show success notification
            showToast(`Created ${mode} collaboration session`, 'success');
            
            // Trigger session created event
            this.triggerEvent('sessionCreated', {
                roomId: this.roomId,
                mode
            });
        } catch (error) {
            console.error('Failed to create collaboration session:', error);
            
            // Show error notification
            showToast(`Failed to create session: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to create collaboration session', 
                'COLLABORATION_CREATE_SESSION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Show dialog to select editing mode
     * 
     * @returns {Promise<string|null>} - Selected editing mode or null if cancelled
     */
    showEditingModeDialog() {
        return new Promise((resolve) => {
            // Create modal container
            const modal = document.createElement('div');
            modal.className = 'modal-overlay';
            modal.innerHTML = `
                <div class="modal-dialog collaboration-mode-dialog">
                    <div class="modal-header">
                        <div class="modal-title">Select Editing Mode</div>
                        <button class="modal-close">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="modal-content">
                        <div class="collaboration-mode-options">
                            <div class="collaboration-mode-option" data-mode="code">
                                <div class="collaboration-mode-icon">
                                    <i class="fas fa-code"></i>
                                </div>
                                <div class="collaboration-mode-info">
                                    <div class="collaboration-mode-name">Code Editor</div>
                                    <div class="collaboration-mode-description">Collaborate on code in real-time</div>
                                </div>
                            </div>
                            <div class="collaboration-mode-option" data-mode="flow">
                                <div class="collaboration-mode-icon">
                                    <i class="fas fa-project-diagram"></i>
                                </div>
                                <div class="collaboration-mode-info">
                                    <div class="collaboration-mode-name">Flow Builder</div>
                                    <div class="collaboration-mode-description">Collaborate on visual flows in real-time</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="modal-action close">Cancel</button>
                    </div>
                </div>
            `;
            
            // Add modal to body
            document.body.appendChild(modal);
            
            // Add event listeners
            const closeButtons = modal.querySelectorAll('.modal-close, .modal-action.close');
            closeButtons.forEach(button => {
                button.addEventListener('click', () => {
                    document.body.removeChild(modal);
                    resolve(null);
                });
            });
            
            // Add mode selection event listeners
            const modeOptions = modal.querySelectorAll('.collaboration-mode-option');
            modeOptions.forEach(option => {
                option.addEventListener('click', () => {
                    const mode = option.getAttribute('data-mode');
                    document.body.removeChild(modal);
                    resolve(mode);
                });
            });
        });
    }

    /**
     * Show dialog to join a session
     */
    showJoinSessionDialog() {
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-dialog collaboration-join-dialog">
                <div class="modal-header">
                    <div class="modal-title">Join Collaboration Session</div>
                    <button class="modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-content">
                    <div class="form-group">
                        <label for="session-id">Session ID</label>
                        <input type="text" id="session-id" class="form-control" placeholder="Enter session ID">
                    </div>
                    <div class="form-group">
                        <label for="username">Your Display Name</label>
                        <input type="text" id="username" class="form-control" value="${this.username}">
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="modal-action join">Join Session</button>
                    <button class="modal-action close">Cancel</button>
                </div>
            </div>
        `;
        
        // Add modal to body
        document.body.appendChild(modal);
        
        // Add event listeners
        const closeButtons = modal.querySelectorAll('.modal-close, .modal-action.close');
        closeButtons.forEach(button => {
            button.addEventListener('click', () => {
                document.body.removeChild(modal);
            });
        });
        
        // Add join session event listener
        const joinButton = modal.querySelector('.modal-action.join');
        if (joinButton) {
            joinButton.addEventListener('click', async () => {
                const sessionId = document.getElementById('session-id').value.trim();
                const username = document.getElementById('username').value.trim();
                
                if (!sessionId) {
                    showToast('Session ID cannot be empty', 'error');
                    return;
                }
                
                if (!username) {
                    showToast('Display name cannot be empty', 'error');
                    return;
                }
                
                // Update username
                this.username = username;
                
                // Close modal
                document.body.removeChild(modal);
                
                // Join session
                await this.joinSession(sessionId);
            });
        }
    }

    /**
     * Join a collaboration session
     * 
     * @param {string} roomId - Room ID to join
     */
    async joinSession(roomId) {
        try {
            // Check session info
            const response = await this.apiClient.get(`/api/v1/developer/collaboration/session/${roomId}`);
            
            // Store room ID and editing mode
            this.roomId = roomId;
            this.editingMode = response.mode;
            
            // Update UI
            this.updateSessionInfo();
            
            // Connect to WebSocket
            await this.connectWebSocket();
            
            // Show success notification
            showToast(`Joined ${this.editingMode} collaboration session`, 'success');
            
            // Trigger session joined event
            this.triggerEvent('sessionJoined', {
                roomId: this.roomId,
                mode: this.editingMode
            });
        } catch (error) {
            console.error(`Failed to join collaboration session ${roomId}:`, error);
            
            // Show error notification
            showToast(`Failed to join session: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to join collaboration session', 
                'COLLABORATION_JOIN_SESSION_FAILED',
                { originalError: error, roomId }
            );
        }
    }

    /**
     * Leave the current collaboration session
     */
    async leaveSession() {
        try {
            // Check if connected
            if (!this.isConnected || !this.roomId) {
                return;
            }
            
            // Disconnect WebSocket
            this.disconnectWebSocket();
            
            // Notify server
            await this.apiClient.post(`/api/v1/developer/collaboration/session/${this.roomId}/leave`, {
                username: this.username
            });
            
            // Clear room ID
            this.roomId = null;
            
            // Clear collaborators
            this.collaborators = [];
            
            // Clear cursor positions
            this.cursorPositions = {};
            
            // Clear local cursors
            this.removeCursorDecorations();
            
            // Update UI
            this.updateSessionInfo();
            this.updateCollaboratorsUI();
            
            // Show success notification
            showToast('Left collaboration session', 'success');
            
            // Trigger session left event
            this.triggerEvent('sessionLeft');
        } catch (error) {
            console.error('Failed to leave collaboration session:', error);
            
            // Show error notification
            showToast(`Failed to leave session: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to leave collaboration session', 
                'COLLABORATION_LEAVE_SESSION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Share the current session
     */
    shareSession() {
        // Check if in a session
        if (!this.roomId) {
            showToast('No active session to share', 'error');
            return;
        }
        
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-dialog collaboration-share-dialog">
                <div class="modal-header">
                    <div class="modal-title">Share Collaboration Session</div>
                    <button class="modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-content">
                    <div class="collaboration-share-info">
                        <p>Share this session ID with your collaborators:</p>
                        <div class="collaboration-share-id">
                            <input type="text" readonly value="${this.roomId}">
                            <button class="collaboration-copy-id" title="Copy to Clipboard">
                                <i class="fas fa-copy"></i>
                            </button>
                        </div>
                        <p class="collaboration-share-note">Collaborators can join by clicking "Join Session" and entering this ID.</p>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="modal-action close">Close</button>
                </div>
            </div>
        `;
        
        // Add modal to body
        document.body.appendChild(modal);
        
        // Add event listeners
        const closeButtons = modal.querySelectorAll('.modal-close, .modal-action.close');
        closeButtons.forEach(button => {
            button.addEventListener('click', () => {
                document.body.removeChild(modal);
            });
        });
        
        // Add copy button event listener
        const copyButton = modal.querySelector('.collaboration-copy-id');
        if (copyButton) {
            copyButton.addEventListener('click', () => {
                const input = modal.querySelector('.collaboration-share-id input');
                input.select();
                document.execCommand('copy');
                
                // Show success notification
                showToast('Session ID copied to clipboard', 'success');
            });
        }
    }

    /**
     * Show collaboration settings
     */
    showSettings() {
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-dialog collaboration-settings-dialog">
                <div class="modal-header">
                    <div class="modal-title">Collaboration Settings</div>
                    <button class="modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-content">
                    <div class="form-group">
                        <label for="display-name">Your Display Name</label>
                        <input type="text" id="display-name" class="form-control" value="${this.username}">
                    </div>
                    <div class="form-group">
                        <label class="checkbox-label">
                            <input type="checkbox" id="show-cursors" checked>
                            Show collaborator cursors
                        </label>
                    </div>
                    <div class="form-group">
                        <label class="checkbox-label">
                            <input type="checkbox" id="play-sound" checked>
                            Play sound when receiving messages
                        </label>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="modal-action save">Save Settings</button>
                    <button class="modal-action close">Cancel</button>
                </div>
            </div>
        `;
        
        // Add modal to body
        document.body.appendChild(modal);
        
        // Add event listeners
        const closeButtons = modal.querySelectorAll('.modal-close, .modal-action.close');
        closeButtons.forEach(button => {
            button.addEventListener('click', () => {
                document.body.removeChild(modal);
            });
        });
        
        // Add save settings event listener
        const saveButton = modal.querySelector('.modal-action.save');
        if (saveButton) {
            saveButton.addEventListener('click', () => {
                const displayName = document.getElementById('display-name').value.trim();
                const showCursors = document.getElementById('show-cursors').checked;
                const playSound = document.getElementById('play-sound').checked;
                
                if (!displayName) {
                    showToast('Display name cannot be empty', 'error');
                    return;
                }
                
                // Update settings
                const oldUsername = this.username;
                this.username = displayName;
                
                // Save settings in local storage
                localStorage.setItem('collaboration.username', this.username);
                localStorage.setItem('collaboration.showCursors', showCursors);
                localStorage.setItem('collaboration.playSound', playSound);
                
                // Send username change if connected
                if (this.isConnected && oldUsername !== this.username) {
                    this.sendUsernameChange(oldUsername, this.username);
                }
                
                // Close modal
                document.body.removeChild(modal);
                
                // Show success notification
                showToast('Settings saved successfully', 'success');
            });
        }
    }

    /**
     * Connect to WebSocket server
     */
    async connectWebSocket() {
        try {
            // Check if already connected
            if (this.isConnected) {
                return;
            }
            
            // Get WebSocket URL from API
            const response = await this.apiClient.get('/api/v1/developer/collaboration/socket-url');
            const wsUrl = response.url;
            
            // Create WebSocket connection
            this.socket = new WebSocket(wsUrl);
            
            // Set up event listeners
            this.socket.onopen = this.handleSocketOpen.bind(this);
            this.socket.onmessage = this.handleSocketMessage.bind(this);
            this.socket.onclose = this.handleSocketClose.bind(this);
            this.socket.onerror = this.handleSocketError.bind(this);
            
            // Return a promise that resolves when the socket is connected
            return new Promise((resolve, reject) => {
                // Set timeout for connection
                const timeout = setTimeout(() => {
                    reject(new Error('WebSocket connection timeout'));
                }, 5000);
                
                // Resolve when socket is open
                this.socket.addEventListener('open', () => {
                    clearTimeout(timeout);
                    resolve();
                });
                
                // Reject on error
                this.socket.addEventListener('error', (error) => {
                    clearTimeout(timeout);
                    reject(error);
                });
            });
        } catch (error) {
            console.error('Failed to connect to WebSocket server:', error);
            throw new AppError(
                'Failed to connect to WebSocket server', 
                'COLLABORATION_WEBSOCKET_CONNECT_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Disconnect from WebSocket server
     */
    disconnectWebSocket() {
        if (this.socket) {
            // Send leave message
            if (this.isConnected && this.roomId) {
                this.sendToSocket({
                    type: 'leave',
                    roomId: this.roomId,
                    userId: this.userId,
                    username: this.username
                });
            }
            
            // Close socket
            this.socket.close();
            this.socket = null;
            
            // Update connection status
            this.isConnected = false;
            this.updateConnectionStatus();
            
            // Enable/disable buttons
            this.updateButtonStates();
            
            // Trigger disconnected event
            this.triggerEvent('disconnected');
        }
    }

    /**
     * Handle WebSocket open event
     */
    handleSocketOpen() {
        // Update connection status
        this.isConnected = true;
        this.updateConnectionStatus();
        
        // Reset reconnect attempts
        this.reconnectAttempts = 0;
        
        // Join room
        this.sendToSocket({
            type: 'join',
            roomId: this.roomId,
            userId: this.userId,
            username: this.username,
            mode: this.editingMode
        });
        
        // Enable/disable buttons
        this.updateButtonStates();
        
        // Add system message
        this.addSystemMessage('Connected to collaboration server');
        
        // Trigger connected event
        this.triggerEvent('connected');
    }

    /**
     * Handle WebSocket message event
     * 
     * @param {MessageEvent} event - WebSocket message event
     */
    handleSocketMessage(event) {
        try {
            // Parse message
            const message = JSON.parse(event.data);
            
            // Handle message based on type
            switch (message.type) {
                case 'joined':
                    this.handleJoinedMessage(message);
                    break;
                case 'left':
                    this.handleLeftMessage(message);
                    break;
                case 'collaborators':
                    this.handleCollaboratorsMessage(message);
                    break;
                case 'cursor':
                    this.handleCursorMessage(message);
                    break;
                case 'operation':
                    this.handleOperationMessage(message);
                    break;
                case 'flow':
                    this.handleFlowMessage(message);
                    break;
                case 'chat':
                    this.handleChatMessage(message);
                    break;
                case 'usernameChanged':
                    this.handleUsernameChangedMessage(message);
                    break;
                case 'error':
                    this.handleErrorMessage(message);
                    break;
                default:
                    console.warn('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('Error handling WebSocket message:', error);
        }
    }

    /**
     * Handle WebSocket close event
     * 
     * @param {CloseEvent} event - WebSocket close event
     */
    handleSocketClose(event) {
        // Update connection status
        this.isConnected = false;
        this.updateConnectionStatus();
        
        // Enable/disable buttons
        this.updateButtonStates();
        
        // Add system message
        this.addSystemMessage(`Disconnected from collaboration server: ${event.reason || 'Connection closed'}`);
        
        // Trigger disconnected event
        this.triggerEvent('disconnected');
        
        // Attempt to reconnect if not a normal closure
        if (event.code !== 1000 && event.code !== 1001) {
            this.attemptReconnect();
        }
    }

    /**
     * Handle WebSocket error event
     * 
     * @param {Event} event - WebSocket error event
     */
    handleSocketError(event) {
        console.error('WebSocket error:', event);
        
        // Add system message
        this.addSystemMessage('Error connecting to collaboration server');
        
        // Trigger error event
        this.triggerEvent('error', {
            message: 'WebSocket error',
            event
        });
    }

    /**
     * Attempt to reconnect to WebSocket server
     */
    attemptReconnect() {
        // Check if max reconnect attempts reached
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.addSystemMessage('Max reconnect attempts reached. Please try again later.');
            return;
        }
        
        // Increment reconnect attempts
        this.reconnectAttempts++;
        
        // Add system message
        this.addSystemMessage(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        
        // Set timeout for reconnect
        setTimeout(async () => {
            try {
                // Reconnect WebSocket
                await this.connectWebSocket();
            } catch (error) {
                console.error('Failed to reconnect:', error);
                
                // Try again
                this.attemptReconnect();
            }
        }, this.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1));
    }

    /**
     * Send message to WebSocket server
     * 
     * @param {Object} message - Message to send
     */
    sendToSocket(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, cannot send message:', message);
        }
    }

    /**
     * Handle 'joined' message
     * 
     * @param {Object} message - Message data
     */
    handleJoinedMessage(message) {
        // Add user to collaborators if not already present
        if (!this.collaborators.some(c => c.userId === message.userId)) {
            this.collaborators.push({
                userId: message.userId,
                username: message.username,
                color: this.getColorForUser(message.userId)
            });
            
            // Update collaborators UI
            this.updateCollaboratorsUI();
        }
        
        // Add system message
        this.addSystemMessage(`${message.username} joined the session`);
        
        // Trigger collaborator joined event
        this.triggerEvent('collaboratorJoined', {
            userId: message.userId,
            username: message.username
        });
    }

    /**
     * Handle 'left' message
     * 
     * @param {Object} message - Message data
     */
    handleLeftMessage(message) {
        // Remove user from collaborators
        this.collaborators = this.collaborators.filter(c => c.userId !== message.userId);
        
        // Remove cursor position
        delete this.cursorPositions[message.userId];
        
        // Remove cursor decoration
        this.removeCursorDecoration(message.userId);
        
        // Update collaborators UI
        this.updateCollaboratorsUI();
        
        // Add system message
        this.addSystemMessage(`${message.username} left the session`);
        
        // Trigger collaborator left event
        this.triggerEvent('collaboratorLeft', {
            userId: message.userId,
            username: message.username
        });
    }

    /**
     * Handle 'collaborators' message
     * 
     * @param {Object} message - Message data
     */
    handleCollaboratorsMessage(message) {
        // Update collaborators list
        this.collaborators = message.collaborators.map(c => ({
            userId: c.userId,
            username: c.username,
            color: this.getColorForUser(c.userId)
        }));
        
        // Update collaborators UI
        this.updateCollaboratorsUI();
        
        // Trigger collaborators updated event
        this.triggerEvent('collaboratorsUpdated', this.collaborators);
    }

    /**
     * Handle 'cursor' message
     * 
     * @param {Object} message - Message data
     */
    handleCursorMessage(message) {
        // Store cursor position
        this.cursorPositions[message.userId] = message.position;
        
        // Update cursor decoration
        this.updateCursorDecoration(message.userId, message.position);
    }

    /**
     * Handle 'operation' message
     * 
     * @param {Object} message - Message data
     */
    handleOperationMessage(message) {
        // Skip own operations
        if (message.userId === this.userId) {
            return;
        }
        
        // Apply operations to editor
        if (this.editingMode === 'code' && this.monacoIntegration && this.monacoIntegration.editor) {
            // Get editor model
            const model = this.monacoIntegration.editor.getModel();
            
            if (!model) {
                return;
            }
            
            // Apply operations
            message.operations.forEach(op => {
                this.applyOperation(op, model);
            });
        }
    }

    /**
     * Handle 'flow' message
     * 
     * @param {Object} message - Message data
     */
    handleFlowMessage(message) {
        // Skip own flow updates
        if (message.userId === this.userId) {
            return;
        }
        
        // Apply flow update to flow builder
        if (this.editingMode === 'flow' && this.flowBuilder) {
            // Update flow
            this.flowBuilder.flow = message.flow;
            
            // Update flow visualization
            this.flowBuilder.updateFlowVisualization();
        }
    }

    /**
     * Handle 'chat' message
     * 
     * @param {Object} message - Message data
     */
    handleChatMessage(message) {
        // Add chat message to UI
        this.addChatMessage(message.userId, message.username, message.text);
        
        // Play sound if not own message and setting enabled
        if (message.userId !== this.userId && localStorage.getItem('collaboration.playSound') !== 'false') {
            this.playMessageSound();
        }
        
        // Trigger chat message received event
        this.triggerEvent('chatMessageReceived', {
            userId: message.userId,
            username: message.username,
            text: message.text
        });
    }

    /**
     * Handle 'usernameChanged' message
     * 
     * @param {Object} message - Message data
     */
    handleUsernameChangedMessage(message) {
        // Update username in collaborators list
        const collaborator = this.collaborators.find(c => c.userId === message.userId);
        if (collaborator) {
            collaborator.username = message.newUsername;
            
            // Update collaborators UI
            this.updateCollaboratorsUI();
            
            // Add system message
            this.addSystemMessage(`${message.oldUsername} changed their name to ${message.newUsername}`);
            
            // Trigger username changed event
            this.triggerEvent('usernameChanged', {
                userId: message.userId,
                oldUsername: message.oldUsername,
                newUsername: message.newUsername
            });
        }
    }

    /**
     * Handle 'error' message
     * 
     * @param {Object} message - Message data
     */
    handleErrorMessage(message) {
        // Add system message
        this.addSystemMessage(`Error: ${message.error}`);
        
        // Show error notification
        showToast(`Collaboration error: ${message.error}`, 'error');
        
        // Trigger error event
        this.triggerEvent('error', {
            message: message.error
        });
    }

    /**
     * Update connection status in UI
     */
    updateConnectionStatus() {
        const statusIndicator = document.querySelector('.collaboration-status-indicator');
        const statusText = document.querySelector('.collaboration-status-text');
        
        if (statusIndicator && statusText) {
            if (this.isConnected) {
                statusIndicator.classList.remove('disconnected');
                statusIndicator.classList.add('connected');
                statusText.textContent = 'Connected';
            } else {
                statusIndicator.classList.remove('connected');
                statusIndicator.classList.add('disconnected');
                statusText.textContent = 'Disconnected';
            }
        }
    }

    /**
     * Update session info in UI
     */
    updateSessionInfo() {
        const sessionId = document.querySelector('.collaboration-session-id');
        const sessionMode = document.querySelector('.collaboration-session-mode');
        
        if (sessionId && sessionMode) {
            if (this.roomId) {
                sessionId.textContent = this.roomId;
                sessionMode.textContent = this.editingMode === 'code' ? 'Code Editor' : 'Flow Builder';
            } else {
                sessionId.textContent = 'No active session';
                sessionMode.textContent = '';
            }
        }
    }

    /**
     * Update collaborators in UI
     */
    updateCollaboratorsUI() {
        const collaboratorsContainer = document.querySelector('.collaboration-collaborators');
        if (!collaboratorsContainer) return;
        
        // Clear container
        collaboratorsContainer.innerHTML = '';
        
        // Check if there are collaborators
        if (this.collaborators.length === 0) {
            collaboratorsContainer.innerHTML = '<div class="collaboration-empty-message">No active collaborators</div>';
            return;
        }
        
        // Create collaborators list
        const collaboratorsList = document.createElement('div');
        collaboratorsList.className = 'collaboration-collaborators-list';
        
        // Add own user first
        const ownUser = this.collaborators.find(c => c.userId === this.userId);
        if (ownUser) {
            const collaboratorItem = document.createElement('div');
            collaboratorItem.className = 'collaboration-collaborator-item';
            collaboratorItem.innerHTML = `
                <div class="collaboration-collaborator-color" style="background-color: ${ownUser.color}"></div>
                <div class="collaboration-collaborator-name">${ownUser.username} (You)</div>
            `;
            
            collaboratorsList.appendChild(collaboratorItem);
        }
        
        // Add other collaborators
        this.collaborators
            .filter(c => c.userId !== this.userId)
            .forEach(collaborator => {
                const collaboratorItem = document.createElement('div');
                collaboratorItem.className = 'collaboration-collaborator-item';
                collaboratorItem.innerHTML = `
                    <div class="collaboration-collaborator-color" style="background-color: ${collaborator.color}"></div>
                    <div class="collaboration-collaborator-name">${collaborator.username}</div>
                `;
                
                collaboratorsList.appendChild(collaboratorItem);
            });
        
        collaboratorsContainer.appendChild(collaboratorsList);
    }

    /**
     * Update button states based on connection status
     */
    updateButtonStates() {
        const leaveSessionButton = document.querySelector('.collaboration-section-action.leave-session');
        const chatInput = document.querySelector('.collaboration-chat-input');
        const chatSendButton = document.querySelector('.collaboration-chat-send');
        
        if (leaveSessionButton) {
            leaveSessionButton.disabled = !this.isConnected;
        }
        
        if (chatInput) {
            chatInput.disabled = !this.isConnected;
        }
        
        if (chatSendButton) {
            chatSendButton.disabled = !this.isConnected;
        }
    }

    /**
     * Add system message to chat
     * 
     * @param {string} message - System message
     */
    addSystemMessage(message) {
        const chatMessages = document.querySelector('.collaboration-chat-messages');
        if (!chatMessages) return;
        
        const messageElement = document.createElement('div');
        messageElement.className = 'collaboration-system-message';
        messageElement.textContent = message;
        
        chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    /**
     * Add chat message to UI
     * 
     * @param {string} userId - User ID
     * @param {string} username - Username
     * @param {string} text - Message text
     */
    addChatMessage(userId, username, text) {
        const chatMessages = document.querySelector('.collaboration-chat-messages');
        if (!chatMessages) return;
        
        const messageElement = document.createElement('div');
        messageElement.className = 'collaboration-chat-message';
        
        // Add own class if message is from current user
        if (userId === this.userId) {
            messageElement.classList.add('own');
        }
        
        // Get user color
        const color = this.getColorForUser(userId);
        
        messageElement.innerHTML = `
            <div class="collaboration-chat-message-header">
                <div class="collaboration-chat-message-username" style="color: ${color}">
                    ${username}${userId === this.userId ? ' (You)' : ''}
                </div>
                <div class="collaboration-chat-message-time">
                    ${new Date().toLocaleTimeString()}
                </div>
            </div>
            <div class="collaboration-chat-message-text">
                ${this.escapeHtml(text)}
            </div>
        `;
        
        chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    /**
     * Send chat message
     * 
     * @param {string} text - Message text
     */
    sendChatMessage(text) {
        // Check if connected
        if (!this.isConnected || !this.roomId) {
            return;
        }
        
        // Check if message is empty
        text = text.trim();
        if (!text) {
            return;
        }
        
        // Send message to server
        this.sendToSocket({
            type: 'chat',
            roomId: this.roomId,
            userId: this.userId,
            username: this.username,
            text
        });
    }

    /**
     * Send cursor position
     * 
     * @param {Object} position - Cursor position
     */
    sendCursorPosition(position) {
        // Check if connected
        if (!this.isConnected || !this.roomId) {
            return;
        }
        
        // Send cursor position to server
        this.sendToSocket({
            type: 'cursor',
            roomId: this.roomId,
            userId: this.userId,
            position
        });
    }

    /**
     * Send operations
     * 
     * @param {Array} operations - Operations to send
     */
    sendOperations(operations) {
        // Check if connected
        if (!this.isConnected || !this.roomId || operations.length === 0) {
            return;
        }
        
        // Send operations to server
        this.sendToSocket({
            type: 'operation',
            roomId: this.roomId,
            userId: this.userId,
            operations
        });
    }

    /**
     * Send flow update
     * 
     * @param {Object} flow - Flow data
     */
    sendFlowUpdate(flow) {
        // Check if connected
        if (!this.isConnected || !this.roomId) {
            return;
        }
        
        // Send flow update to server
        this.sendToSocket({
            type: 'flow',
            roomId: this.roomId,
            userId: this.userId,
            flow
        });
    }

    /**
     * Send username change
     * 
     * @param {string} oldUsername - Old username
     * @param {string} newUsername - New username
     */
    sendUsernameChange(oldUsername, newUsername) {
        // Check if connected
        if (!this.isConnected || !this.roomId) {
            return;
        }
        
        // Send username change to server
        this.sendToSocket({
            type: 'usernameChanged',
            roomId: this.roomId,
            userId: this.userId,
            oldUsername,
            newUsername
        });
    }

    /**
     * Convert Monaco change events to operations
     * 
     * @param {Array} changes - Monaco change events
     * @returns {Array} - Operations
     */
    monacoChangesToOperations(changes) {
        return changes.map(change => {
            return {
                type: change.text === '' ? 'delete' : (change.rangeLength === 0 ? 'insert' : 'replace'),
                range: {
                    startLineNumber: change.range.startLineNumber,
                    startColumn: change.range.startColumn,
                    endLineNumber: change.range.endLineNumber,
                    endColumn: change.range.endColumn
                },
                text: change.text,
                rangeLength: change.rangeLength
            };
        });
    }

    /**
     * Apply operation to Monaco editor
     * 
     * @param {Object} operation - Operation to apply
     * @param {monaco.editor.ITextModel} model - Monaco editor model
     */
    applyOperation(operation, model) {
        // Create edit operation
        const range = new monaco.Range(
            operation.range.startLineNumber,
            operation.range.startColumn,
            operation.range.endLineNumber,
            operation.range.endColumn
        );
        
        // Execute edit
        model.pushEditOperations(
            [],
            [
                {
                    range,
                    text: operation.text,
                    forceMoveMarkers: true
                }
            ],
            () => null
        );
    }

    /**
     * Update cursor decoration for a user
     * 
     * @param {string} userId - User ID
     * @param {Object} position - Cursor position
     */
    updateCursorDecoration(userId, position) {
        // Check if cursors should be shown
        if (localStorage.getItem('collaboration.showCursors') === 'false') {
            return;
        }
        
        // Check if in code editing mode
        if (this.editingMode !== 'code' || !this.monacoIntegration || !this.monacoIntegration.editor) {
            return;
        }
        
        // Skip own cursor
        if (userId === this.userId) {
            return;
        }
        
        // Get user data
        const collaborator = this.collaborators.find(c => c.userId === userId);
        if (!collaborator) {
            return;
        }
        
        // Remove existing decoration
        this.removeCursorDecoration(userId);
        
        // Create cursor decoration
        const decorations = [
            {
                range: new monaco.Range(
                    position.lineNumber,
                    position.column,
                    position.lineNumber,
                    position.column
                ),
                options: {
                    className: `collaboration-cursor-${userId}`,
                    zIndex: 100,
                    hoverMessage: {
                        value: collaborator.username
                    }
                }
            }
        ];
        
        // Add cursor decoration
        this.localCursors[userId] = this.monacoIntegration.editor.deltaDecorations([], decorations);
        
        // Create cursor style if it doesn't exist
        if (!document.getElementById(`cursor-style-${userId}`)) {
            const style = document.createElement('style');
            style.id = `cursor-style-${userId}`;
            style.textContent = `
                .collaboration-cursor-${userId} {
                    background: ${collaborator.color};
                    width: 2px !important;
                    margin-left: -1px;
                    height: 18px !important;
                    position: absolute;
                    animation: cursor-blink 1s ease infinite;
                }
                
                @keyframes cursor-blink {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
            `;
            
            document.head.appendChild(style);
        }
    }

    /**
     * Remove cursor decoration for a user
     * 
     * @param {string} userId - User ID
     */
    removeCursorDecoration(userId) {
        // Check if in code editing mode
        if (this.editingMode !== 'code' || !this.monacoIntegration || !this.monacoIntegration.editor) {
            return;
        }
        
        // Remove decoration if it exists
        if (this.localCursors[userId]) {
            this.monacoIntegration.editor.deltaDecorations(this.localCursors[userId], []);
            delete this.localCursors[userId];
        }
        
        // Remove cursor style
        const style = document.getElementById(`cursor-style-${userId}`);
        if (style) {
            document.head.removeChild(style);
        }
    }

    /**
     * Remove all cursor decorations
     */
    removeCursorDecorations() {
        // Check if in code editing mode
        if (this.editingMode !== 'code' || !this.monacoIntegration || !this.monacoIntegration.editor) {
            return;
        }
        
        // Remove all decorations
        for (const userId in this.localCursors) {
            this.removeCursorDecoration(userId);
        }
        
        // Clear local cursors
        this.localCursors = {};
    }

    /**
     * Get color for a user
     * 
     * @param {string} userId - User ID
     * @returns {string} - User color
     */
    getColorForUser(userId) {
        // Return color if already assigned
        if (this.clientColors[userId]) {
            return this.clientColors[userId];
        }
        
        // Colors array
        const colors = [
            '#FF5733', // Red
            '#33A8FF', // Blue
            '#33FF57', // Green
            '#FF33A8', // Pink
            '#A833FF', // Purple
            '#FF8C33', // Orange
            '#33FFC1', // Teal
            '#8CFF33', // Lime
            '#FF33FF', // Magenta
            '#33FFFF'  // Cyan
        ];
        
        // Assign color
        const colorIndex = Object.keys(this.clientColors).length % colors.length;
        this.clientColors[userId] = colors[colorIndex];
        
        return this.clientColors[userId];
    }

    /**
     * Play message notification sound
     */
    playMessageSound() {
        try {
            // Create audio element
            const audio = new Audio('data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjI5LjEwMAAAAAAAAAAAAAAA//tQwAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAABAAADQgD///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////8AAAA5TEFNRTMuMTAwBK8AAAAAAAAAABUgJAMGQQABmgAAA0LPZ3yYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//tAwAAADoSVIUAYYdIrCik3PBAWjEgAgwMADBkYrAIBgYPhm8+D4fH8+MQPBDwfD//L4Ph8ZKTgMDx/JJw+GAMYYAwwxswAygEjAFGAYMAcYCYwCpgHDAVmA0MBGdQ6F7Nty8+Yc/Tr4z+dlk8caj51pXqF5sSt5l5zj+Yxj/5kOdtn6n//zxfUZ7///5Z/q2/Vms///MiZprLUmdGXQ3csLHT/7atcaGfPMrVQanV+pHUvLXA46Hpms0Lm8yrepeWh52/XcbOamVWJemczt+01V9Q7ZybeaOZzMbnuVbs5/TLpQ7fUuHcttfUunRlPrAGx5ZR+AAAANzI4AEIADQApkrKAA');
            
            // Play sound
            audio.play();
        } catch (error) {
            console.error('Failed to play message sound:', error);
        }
    }

    /**
     * Escape HTML to prevent XSS
     * 
     * @param {string} html - HTML to escape
     * @returns {string} - Escaped HTML
     */
    escapeHtml(html) {
        const div = document.createElement('div');
        div.textContent = html;
        return div.innerHTML;
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
window.Collaboration = Collaboration;
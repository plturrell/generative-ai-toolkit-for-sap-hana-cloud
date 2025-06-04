/**
 * Git Integration Module
 * 
 * Provides Git integration capabilities for the Developer Studio.
 */

class GitIntegration {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.currentRepository = null;
        this.currentBranch = null;
        this.branches = [];
        this.commits = [];
        this.status = {};
        this.isInitialized = false;
        this.eventListeners = {};
    }

    /**
     * Initialize Git integration
     * 
     * @returns {Promise<boolean>} - True if initialized successfully
     */
    async initialize() {
        try {
            // Check if Git is available
            const gitStatus = await this.apiClient.get('/api/v1/developer/git/status');
            
            // Store repository info
            this.currentRepository = gitStatus.repository;
            this.currentBranch = gitStatus.branch;
            this.status = gitStatus.status;
            
            // Get branches
            await this.fetchBranches();
            
            // Get commits
            await this.fetchCommits();
            
            // Create Git UI
            this.createGitUI();
            
            // Set initialized flag
            this.isInitialized = true;
            
            // Trigger initialized event
            this.triggerEvent('initialized', {
                repository: this.currentRepository,
                branch: this.currentBranch,
                status: this.status
            });
            
            return true;
        } catch (error) {
            console.error('Failed to initialize Git integration:', error);
            throw new AppError(
                'Failed to initialize Git integration', 
                'GIT_INITIALIZATION_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Create Git UI
     */
    createGitUI() {
        // Create Git panel
        const gitPanel = document.createElement('div');
        gitPanel.id = 'git-panel';
        gitPanel.className = 'git-panel';
        
        // Create Git panel header
        const gitHeader = document.createElement('div');
        gitHeader.className = 'git-panel-header';
        gitHeader.innerHTML = `
            <div class="git-panel-title">
                <i class="fas fa-code-branch"></i>
                Git
            </div>
            <div class="git-branch-info">
                ${this.currentBranch || 'No branch'}
            </div>
            <div class="git-panel-actions">
                <button class="git-action refresh" title="Refresh">
                    <i class="fas fa-sync-alt"></i>
                </button>
                <button class="git-action close" title="Close Git Panel">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Create Git panel content with tabs
        const gitContent = document.createElement('div');
        gitContent.className = 'git-panel-content';
        gitContent.innerHTML = `
            <div class="git-tabs">
                <div class="git-tab active" data-tab="changes">Changes</div>
                <div class="git-tab" data-tab="history">History</div>
                <div class="git-tab" data-tab="branches">Branches</div>
            </div>
            <div class="git-content">
                <div class="git-tab-content changes active">
                    <div class="git-changes-header">
                        <div class="git-changes-title">Changes</div>
                        <div class="git-changes-actions">
                            <button class="git-changes-action stage-all" title="Stage All Changes">
                                <i class="fas fa-plus"></i> Stage All
                            </button>
                            <button class="git-changes-action refresh" title="Refresh">
                                <i class="fas fa-sync-alt"></i>
                            </button>
                        </div>
                    </div>
                    <div class="git-changes">
                        <div class="git-staged-changes">
                            <div class="git-section-header">
                                <div class="git-section-title">Staged Changes</div>
                                <div class="git-section-actions">
                                    <button class="git-section-action unstage-all" title="Unstage All">
                                        <i class="fas fa-minus"></i>
                                    </button>
                                </div>
                            </div>
                            <div class="git-staged-files">
                                <div class="git-empty-message">No staged changes</div>
                            </div>
                        </div>
                        <div class="git-unstaged-changes">
                            <div class="git-section-header">
                                <div class="git-section-title">Changes</div>
                            </div>
                            <div class="git-unstaged-files">
                                <div class="git-empty-message">No changes</div>
                            </div>
                        </div>
                    </div>
                    <div class="git-commit-area">
                        <textarea class="git-commit-message" placeholder="Commit message"></textarea>
                        <div class="git-commit-actions">
                            <button class="git-commit-action commit" disabled>
                                <i class="fas fa-check"></i> Commit
                            </button>
                            <button class="git-commit-action commit-push">
                                <i class="fas fa-upload"></i> Commit & Push
                            </button>
                        </div>
                    </div>
                </div>
                <div class="git-tab-content history">
                    <div class="git-history-header">
                        <div class="git-history-title">Commit History</div>
                        <div class="git-history-actions">
                            <button class="git-history-action refresh" title="Refresh">
                                <i class="fas fa-sync-alt"></i>
                            </button>
                        </div>
                    </div>
                    <div class="git-commits">
                        <div class="git-empty-message">No commits</div>
                    </div>
                </div>
                <div class="git-tab-content branches">
                    <div class="git-branches-header">
                        <div class="git-branches-title">Branches</div>
                        <div class="git-branches-actions">
                            <button class="git-branches-action create" title="Create Branch">
                                <i class="fas fa-plus"></i> New Branch
                            </button>
                            <button class="git-branches-action refresh" title="Refresh">
                                <i class="fas fa-sync-alt"></i>
                            </button>
                        </div>
                    </div>
                    <div class="git-branches">
                        <div class="git-empty-message">No branches</div>
                    </div>
                </div>
            </div>
        `;
        
        // Add header and content to panel
        gitPanel.appendChild(gitHeader);
        gitPanel.appendChild(gitContent);
        
        // Add panel to the workspace
        const workspace = document.querySelector('.developer-workspace');
        if (workspace) {
            workspace.appendChild(gitPanel);
        }
        
        // Add event listeners
        this.addGitPanelEventListeners();
        
        // Update Git UI with initial data
        this.updateGitUI();
    }

    /**
     * Add event listeners to Git panel
     */
    addGitPanelEventListeners() {
        // Panel header actions
        const refreshButton = document.querySelector('.git-panel-header .git-action.refresh');
        const closeButton = document.querySelector('.git-panel-header .git-action.close');
        
        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                this.refreshGitStatus();
            });
        }
        
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                this.hideGitPanel();
            });
        }
        
        // Tab switching
        const gitTabs = document.querySelectorAll('.git-tab');
        const tabContents = document.querySelectorAll('.git-tab-content');
        
        gitTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                gitTabs.forEach(t => t.classList.remove('active'));
                
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
        
        // Changes tab actions
        const stageAllButton = document.querySelector('.git-changes-action.stage-all');
        const refreshChangesButton = document.querySelector('.git-changes-action.refresh');
        const unstageAllButton = document.querySelector('.git-section-action.unstage-all');
        
        if (stageAllButton) {
            stageAllButton.addEventListener('click', () => {
                this.stageAllChanges();
            });
        }
        
        if (refreshChangesButton) {
            refreshChangesButton.addEventListener('click', () => {
                this.refreshGitStatus();
            });
        }
        
        if (unstageAllButton) {
            unstageAllButton.addEventListener('click', () => {
                this.unstageAllChanges();
            });
        }
        
        // Commit actions
        const commitButton = document.querySelector('.git-commit-action.commit');
        const commitPushButton = document.querySelector('.git-commit-action.commit-push');
        const commitMessage = document.querySelector('.git-commit-message');
        
        if (commitMessage) {
            commitMessage.addEventListener('input', () => {
                // Enable commit button if message is not empty
                if (commitButton) {
                    commitButton.disabled = !commitMessage.value.trim();
                }
            });
        }
        
        if (commitButton) {
            commitButton.addEventListener('click', () => {
                const message = commitMessage.value.trim();
                if (message) {
                    this.commit(message);
                }
            });
        }
        
        if (commitPushButton) {
            commitPushButton.addEventListener('click', () => {
                const message = commitMessage.value.trim();
                if (message) {
                    this.commitAndPush(message);
                }
            });
        }
        
        // History tab actions
        const refreshHistoryButton = document.querySelector('.git-history-action.refresh');
        
        if (refreshHistoryButton) {
            refreshHistoryButton.addEventListener('click', () => {
                this.fetchCommits();
            });
        }
        
        // Branches tab actions
        const createBranchButton = document.querySelector('.git-branches-action.create');
        const refreshBranchesButton = document.querySelector('.git-branches-action.refresh');
        
        if (createBranchButton) {
            createBranchButton.addEventListener('click', () => {
                this.showCreateBranchDialog();
            });
        }
        
        if (refreshBranchesButton) {
            refreshBranchesButton.addEventListener('click', () => {
                this.fetchBranches();
            });
        }
    }

    /**
     * Show Git panel
     */
    showGitPanel() {
        const gitPanel = document.getElementById('git-panel');
        if (gitPanel) {
            gitPanel.style.display = 'flex';
        }
    }

    /**
     * Hide Git panel
     */
    hideGitPanel() {
        const gitPanel = document.getElementById('git-panel');
        if (gitPanel) {
            gitPanel.style.display = 'none';
        }
    }

    /**
     * Toggle Git panel
     */
    toggleGitPanel() {
        const gitPanel = document.getElementById('git-panel');
        if (gitPanel) {
            if (gitPanel.style.display === 'none') {
                this.showGitPanel();
            } else {
                this.hideGitPanel();
            }
        } else {
            this.createGitUI();
            this.showGitPanel();
        }
    }

    /**
     * Update Git UI with current data
     */
    updateGitUI() {
        // Update branch info
        const branchInfo = document.querySelector('.git-branch-info');
        if (branchInfo) {
            branchInfo.textContent = this.currentBranch || 'No branch';
        }
        
        // Update changes
        this.updateChangesUI();
        
        // Update history
        this.updateHistoryUI();
        
        // Update branches
        this.updateBranchesUI();
    }

    /**
     * Update Changes tab UI
     */
    updateChangesUI() {
        const stagedFiles = document.querySelector('.git-staged-files');
        const unstagedFiles = document.querySelector('.git-unstaged-files');
        
        if (!stagedFiles || !unstagedFiles) return;
        
        // Clear containers
        stagedFiles.innerHTML = '';
        unstagedFiles.innerHTML = '';
        
        // Handle staged files
        if (this.status.staged && this.status.staged.length > 0) {
            const filesList = document.createElement('div');
            filesList.className = 'git-files-list';
            
            this.status.staged.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'git-file-item';
                
                // Set icon based on status
                let statusIcon;
                let statusClass;
                
                switch (file.status) {
                    case 'added':
                        statusIcon = 'fas fa-plus';
                        statusClass = 'added';
                        break;
                    case 'modified':
                        statusIcon = 'fas fa-pencil-alt';
                        statusClass = 'modified';
                        break;
                    case 'deleted':
                        statusIcon = 'fas fa-trash';
                        statusClass = 'deleted';
                        break;
                    case 'renamed':
                        statusIcon = 'fas fa-file-signature';
                        statusClass = 'renamed';
                        break;
                    default:
                        statusIcon = 'fas fa-file';
                        statusClass = '';
                }
                
                fileItem.innerHTML = `
                    <div class="git-file-status ${statusClass}">
                        <i class="${statusIcon}"></i>
                    </div>
                    <div class="git-file-name" title="${file.path}">${file.path}</div>
                    <div class="git-file-actions">
                        <button class="git-file-action unstage" data-file="${file.path}" title="Unstage File">
                            <i class="fas fa-minus"></i>
                        </button>
                    </div>
                `;
                
                filesList.appendChild(fileItem);
            });
            
            stagedFiles.appendChild(filesList);
            
            // Add event listeners for unstage buttons
            const unstageButtons = stagedFiles.querySelectorAll('.git-file-action.unstage');
            unstageButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    const filePath = e.currentTarget.getAttribute('data-file');
                    this.unstageFile(filePath);
                });
            });
        } else {
            stagedFiles.innerHTML = '<div class="git-empty-message">No staged changes</div>';
        }
        
        // Handle unstaged files
        if (this.status.unstaged && this.status.unstaged.length > 0) {
            const filesList = document.createElement('div');
            filesList.className = 'git-files-list';
            
            this.status.unstaged.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'git-file-item';
                
                // Set icon based on status
                let statusIcon;
                let statusClass;
                
                switch (file.status) {
                    case 'added':
                        statusIcon = 'fas fa-plus';
                        statusClass = 'added';
                        break;
                    case 'modified':
                        statusIcon = 'fas fa-pencil-alt';
                        statusClass = 'modified';
                        break;
                    case 'deleted':
                        statusIcon = 'fas fa-trash';
                        statusClass = 'deleted';
                        break;
                    case 'untracked':
                        statusIcon = 'fas fa-question';
                        statusClass = 'untracked';
                        break;
                    default:
                        statusIcon = 'fas fa-file';
                        statusClass = '';
                }
                
                fileItem.innerHTML = `
                    <div class="git-file-status ${statusClass}">
                        <i class="${statusIcon}"></i>
                    </div>
                    <div class="git-file-name" title="${file.path}">${file.path}</div>
                    <div class="git-file-actions">
                        <button class="git-file-action stage" data-file="${file.path}" title="Stage File">
                            <i class="fas fa-plus"></i>
                        </button>
                        <button class="git-file-action view-diff" data-file="${file.path}" title="View Changes">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                `;
                
                filesList.appendChild(fileItem);
            });
            
            unstagedFiles.appendChild(filesList);
            
            // Add event listeners for stage buttons
            const stageButtons = unstagedFiles.querySelectorAll('.git-file-action.stage');
            stageButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    const filePath = e.currentTarget.getAttribute('data-file');
                    this.stageFile(filePath);
                });
            });
            
            // Add event listeners for view diff buttons
            const viewDiffButtons = unstagedFiles.querySelectorAll('.git-file-action.view-diff');
            viewDiffButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    const filePath = e.currentTarget.getAttribute('data-file');
                    this.viewDiff(filePath);
                });
            });
        } else {
            unstagedFiles.innerHTML = '<div class="git-empty-message">No unstaged changes</div>';
        }
        
        // Update commit button state
        const commitButton = document.querySelector('.git-commit-action.commit');
        const commitMessage = document.querySelector('.git-commit-message');
        
        if (commitButton && commitMessage) {
            const hasStagedChanges = this.status.staged && this.status.staged.length > 0;
            const hasCommitMessage = commitMessage.value.trim().length > 0;
            
            commitButton.disabled = !hasStagedChanges || !hasCommitMessage;
        }
    }

    /**
     * Update History tab UI
     */
    updateHistoryUI() {
        const commitsContainer = document.querySelector('.git-commits');
        if (!commitsContainer) return;
        
        // Clear container
        commitsContainer.innerHTML = '';
        
        // Handle commits
        if (this.commits && this.commits.length > 0) {
            const commitsList = document.createElement('div');
            commitsList.className = 'git-commits-list';
            
            this.commits.forEach(commit => {
                const commitItem = document.createElement('div');
                commitItem.className = 'git-commit-item';
                commitItem.innerHTML = `
                    <div class="git-commit-header">
                        <div class="git-commit-hash" title="${commit.hash}">${commit.hash.substring(0, 7)}</div>
                        <div class="git-commit-date">${formatDate(commit.date)}</div>
                    </div>
                    <div class="git-commit-message" title="${commit.message}">${commit.message}</div>
                    <div class="git-commit-author">${commit.author}</div>
                `;
                
                // Add click handler to show commit details
                commitItem.addEventListener('click', () => {
                    this.showCommitDetails(commit.hash);
                });
                
                commitsList.appendChild(commitItem);
            });
            
            commitsContainer.appendChild(commitsList);
        } else {
            commitsContainer.innerHTML = '<div class="git-empty-message">No commits</div>';
        }
        
        // Helper function to format date
        function formatDate(dateString) {
            const date = new Date(dateString);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        }
    }

    /**
     * Update Branches tab UI
     */
    updateBranchesUI() {
        const branchesContainer = document.querySelector('.git-branches');
        if (!branchesContainer) return;
        
        // Clear container
        branchesContainer.innerHTML = '';
        
        // Handle branches
        if (this.branches && this.branches.length > 0) {
            const branchesList = document.createElement('div');
            branchesList.className = 'git-branches-list';
            
            this.branches.forEach(branch => {
                const branchItem = document.createElement('div');
                branchItem.className = 'git-branch-item';
                
                // Mark current branch
                if (branch.name === this.currentBranch) {
                    branchItem.classList.add('active');
                }
                
                branchItem.innerHTML = `
                    <div class="git-branch-name">
                        <i class="fas fa-code-branch"></i>
                        ${branch.name}
                        ${branch.name === this.currentBranch ? '<span class="git-current-branch">current</span>' : ''}
                    </div>
                    <div class="git-branch-actions">
                        ${branch.name !== this.currentBranch ? `
                            <button class="git-branch-action checkout" data-branch="${branch.name}" title="Checkout Branch">
                                <i class="fas fa-check"></i>
                            </button>
                        ` : ''}
                        <button class="git-branch-action merge" data-branch="${branch.name}" title="Merge Branch">
                            <i class="fas fa-code-merge"></i>
                        </button>
                    </div>
                `;
                
                branchesList.appendChild(branchItem);
            });
            
            branchesContainer.appendChild(branchesList);
            
            // Add event listeners for checkout buttons
            const checkoutButtons = branchesContainer.querySelectorAll('.git-branch-action.checkout');
            checkoutButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    const branchName = e.currentTarget.getAttribute('data-branch');
                    this.checkoutBranch(branchName);
                });
            });
            
            // Add event listeners for merge buttons
            const mergeButtons = branchesContainer.querySelectorAll('.git-branch-action.merge');
            mergeButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    const branchName = e.currentTarget.getAttribute('data-branch');
                    this.mergeBranch(branchName);
                });
            });
        } else {
            branchesContainer.innerHTML = '<div class="git-empty-message">No branches</div>';
        }
    }

    /**
     * Refresh Git status
     */
    async refreshGitStatus() {
        try {
            // Get Git status
            const gitStatus = await this.apiClient.get('/api/v1/developer/git/status');
            
            // Update repository info
            this.currentRepository = gitStatus.repository;
            this.currentBranch = gitStatus.branch;
            this.status = gitStatus.status;
            
            // Update UI
            this.updateGitUI();
            
            // Trigger status updated event
            this.triggerEvent('statusUpdated', {
                repository: this.currentRepository,
                branch: this.currentBranch,
                status: this.status
            });
            
            return gitStatus;
        } catch (error) {
            console.error('Failed to refresh Git status:', error);
            throw new AppError(
                'Failed to refresh Git status', 
                'GIT_STATUS_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Fetch Git branches
     */
    async fetchBranches() {
        try {
            // Get branches
            const response = await this.apiClient.get('/api/v1/developer/git/branches');
            
            // Update branches
            this.branches = response.branches;
            
            // Update UI
            this.updateBranchesUI();
            
            // Trigger branches updated event
            this.triggerEvent('branchesUpdated', this.branches);
            
            return this.branches;
        } catch (error) {
            console.error('Failed to fetch Git branches:', error);
            throw new AppError(
                'Failed to fetch Git branches', 
                'GIT_BRANCHES_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Fetch Git commits
     */
    async fetchCommits() {
        try {
            // Get commits
            const response = await this.apiClient.get('/api/v1/developer/git/commits');
            
            // Update commits
            this.commits = response.commits;
            
            // Update UI
            this.updateHistoryUI();
            
            // Trigger commits updated event
            this.triggerEvent('commitsUpdated', this.commits);
            
            return this.commits;
        } catch (error) {
            console.error('Failed to fetch Git commits:', error);
            throw new AppError(
                'Failed to fetch Git commits', 
                'GIT_COMMITS_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Stage a file
     * 
     * @param {string} filePath - Path of the file to stage
     */
    async stageFile(filePath) {
        try {
            // Stage file
            await this.apiClient.post('/api/v1/developer/git/stage', {
                files: [filePath]
            });
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Trigger file staged event
            this.triggerEvent('fileStaged', filePath);
        } catch (error) {
            console.error(`Failed to stage file ${filePath}:`, error);
            throw new AppError(
                'Failed to stage file', 
                'GIT_STAGE_FAILED',
                { originalError: error, filePath }
            );
        }
    }

    /**
     * Unstage a file
     * 
     * @param {string} filePath - Path of the file to unstage
     */
    async unstageFile(filePath) {
        try {
            // Unstage file
            await this.apiClient.post('/api/v1/developer/git/unstage', {
                files: [filePath]
            });
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Trigger file unstaged event
            this.triggerEvent('fileUnstaged', filePath);
        } catch (error) {
            console.error(`Failed to unstage file ${filePath}:`, error);
            throw new AppError(
                'Failed to unstage file', 
                'GIT_UNSTAGE_FAILED',
                { originalError: error, filePath }
            );
        }
    }

    /**
     * Stage all changes
     */
    async stageAllChanges() {
        try {
            // Stage all changes
            await this.apiClient.post('/api/v1/developer/git/stage-all');
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Trigger all files staged event
            this.triggerEvent('allFilesStaged');
        } catch (error) {
            console.error('Failed to stage all changes:', error);
            throw new AppError(
                'Failed to stage all changes', 
                'GIT_STAGE_ALL_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Unstage all changes
     */
    async unstageAllChanges() {
        try {
            // Unstage all changes
            await this.apiClient.post('/api/v1/developer/git/unstage-all');
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Trigger all files unstaged event
            this.triggerEvent('allFilesUnstaged');
        } catch (error) {
            console.error('Failed to unstage all changes:', error);
            throw new AppError(
                'Failed to unstage all changes', 
                'GIT_UNSTAGE_ALL_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * View diff for a file
     * 
     * @param {string} filePath - Path of the file to view diff
     */
    async viewDiff(filePath) {
        try {
            // Get diff
            const response = await this.apiClient.get(`/api/v1/developer/git/diff?file=${encodeURIComponent(filePath)}`);
            
            // Show diff in modal
            this.showDiffModal(filePath, response.diff);
            
            // Trigger diff viewed event
            this.triggerEvent('diffViewed', {
                filePath,
                diff: response.diff
            });
            
            return response.diff;
        } catch (error) {
            console.error(`Failed to view diff for file ${filePath}:`, error);
            throw new AppError(
                'Failed to view diff', 
                'GIT_DIFF_FAILED',
                { originalError: error, filePath }
            );
        }
    }

    /**
     * Show diff modal
     * 
     * @param {string} filePath - Path of the file
     * @param {string} diff - Git diff content
     */
    showDiffModal(filePath, diff) {
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-dialog git-diff-modal">
                <div class="modal-header">
                    <div class="modal-title">Diff: ${filePath}</div>
                    <button class="modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-content">
                    <pre class="git-diff-content">${this.formatDiff(diff)}</pre>
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
    }

    /**
     * Format Git diff for display
     * 
     * @param {string} diff - Git diff content
     * @returns {string} - Formatted diff with HTML for syntax highlighting
     */
    formatDiff(diff) {
        // Escape HTML
        let htmlDiff = diff.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        
        // Add color syntax
        htmlDiff = htmlDiff.replace(/^(\+.*?)$/gm, '<span class="diff-added">$1</span>');
        htmlDiff = htmlDiff.replace(/^(-.*?)$/gm, '<span class="diff-removed">$1</span>');
        htmlDiff = htmlDiff.replace(/^(@@ .*? @@)(.*)$/gm, '<span class="diff-hunk">$1</span>$2');
        htmlDiff = htmlDiff.replace(/^(diff --git .*)$/gm, '<span class="diff-header">$1</span>');
        htmlDiff = htmlDiff.replace(/^(index .*?)$/gm, '<span class="diff-index">$1</span>');
        htmlDiff = htmlDiff.replace(/^(--- .*)$/gm, '<span class="diff-file">$1</span>');
        htmlDiff = htmlDiff.replace(/^(\+\+\+ .*)$/gm, '<span class="diff-file">$1</span>');
        
        return htmlDiff;
    }

    /**
     * Commit changes
     * 
     * @param {string} message - Commit message
     */
    async commit(message) {
        try {
            // Validate commit message
            if (!message || !message.trim()) {
                throw new Error('Commit message cannot be empty');
            }
            
            // Make commit
            await this.apiClient.post('/api/v1/developer/git/commit', {
                message
            });
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Refresh commits
            await this.fetchCommits();
            
            // Clear commit message
            const commitMessage = document.querySelector('.git-commit-message');
            if (commitMessage) {
                commitMessage.value = '';
                
                // Disable commit button
                const commitButton = document.querySelector('.git-commit-action.commit');
                if (commitButton) {
                    commitButton.disabled = true;
                }
            }
            
            // Trigger commit created event
            this.triggerEvent('commitCreated', {
                message
            });
            
            // Show success notification
            showToast('Changes committed successfully', 'success');
        } catch (error) {
            console.error(`Failed to commit changes:`, error);
            
            // Show error notification
            showToast(`Failed to commit changes: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to commit changes', 
                'GIT_COMMIT_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Commit and push changes
     * 
     * @param {string} message - Commit message
     */
    async commitAndPush(message) {
        try {
            // Validate commit message
            if (!message || !message.trim()) {
                throw new Error('Commit message cannot be empty');
            }
            
            // Make commit
            await this.apiClient.post('/api/v1/developer/git/commit', {
                message
            });
            
            // Push changes
            await this.apiClient.post('/api/v1/developer/git/push');
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Refresh commits
            await this.fetchCommits();
            
            // Clear commit message
            const commitMessage = document.querySelector('.git-commit-message');
            if (commitMessage) {
                commitMessage.value = '';
                
                // Disable commit button
                const commitButton = document.querySelector('.git-commit-action.commit');
                if (commitButton) {
                    commitButton.disabled = true;
                }
            }
            
            // Trigger commit and push event
            this.triggerEvent('commitAndPush', {
                message
            });
            
            // Show success notification
            showToast('Changes committed and pushed successfully', 'success');
        } catch (error) {
            console.error(`Failed to commit and push changes:`, error);
            
            // Show error notification
            showToast(`Failed to commit and push changes: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to commit and push changes', 
                'GIT_COMMIT_PUSH_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Show commit details
     * 
     * @param {string} commitHash - Commit hash to show details for
     */
    async showCommitDetails(commitHash) {
        try {
            // Get commit details
            const response = await this.apiClient.get(`/api/v1/developer/git/commit/${commitHash}`);
            
            // Show commit details in modal
            this.showCommitModal(response.commit);
            
            // Trigger commit details viewed event
            this.triggerEvent('commitDetailsViewed', response.commit);
            
            return response.commit;
        } catch (error) {
            console.error(`Failed to show commit details for ${commitHash}:`, error);
            throw new AppError(
                'Failed to show commit details', 
                'GIT_COMMIT_DETAILS_FAILED',
                { originalError: error, commitHash }
            );
        }
    }

    /**
     * Show commit details modal
     * 
     * @param {Object} commit - Commit details
     */
    showCommitModal(commit) {
        // Format date
        const date = new Date(commit.date);
        const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-dialog git-commit-modal">
                <div class="modal-header">
                    <div class="modal-title">Commit: ${commit.hash.substring(0, 7)}</div>
                    <button class="modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-content">
                    <div class="git-commit-details">
                        <div class="git-commit-info">
                            <div class="git-commit-hash">
                                <span class="git-detail-label">Hash:</span>
                                <span class="git-detail-value">${commit.hash}</span>
                            </div>
                            <div class="git-commit-author">
                                <span class="git-detail-label">Author:</span>
                                <span class="git-detail-value">${commit.author}</span>
                            </div>
                            <div class="git-commit-date">
                                <span class="git-detail-label">Date:</span>
                                <span class="git-detail-value">${formattedDate}</span>
                            </div>
                            <div class="git-commit-message">
                                <span class="git-detail-label">Message:</span>
                                <span class="git-detail-value">${commit.message}</span>
                            </div>
                        </div>
                        <div class="git-commit-files">
                            <div class="git-detail-label">Changed Files:</div>
                            <div class="git-files-list">
                                ${commit.files.map(file => `
                                    <div class="git-file-item">
                                        <div class="git-file-status ${getStatusClass(file.status)}">
                                            <i class="${getStatusIcon(file.status)}"></i>
                                        </div>
                                        <div class="git-file-name" title="${file.path}">${file.path}</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                    ${commit.diff ? `
                        <div class="git-commit-diff">
                            <div class="git-detail-label">Diff:</div>
                            <pre class="git-diff-content">${this.formatDiff(commit.diff)}</pre>
                        </div>
                    ` : ''}
                </div>
                <div class="modal-footer">
                    <button class="modal-action revert" data-hash="${commit.hash}">Revert Commit</button>
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
        
        // Add revert commit event listener
        const revertButton = modal.querySelector('.modal-action.revert');
        if (revertButton) {
            revertButton.addEventListener('click', () => {
                const hash = revertButton.getAttribute('data-hash');
                this.revertCommit(hash);
                document.body.removeChild(modal);
            });
        }
        
        // Helper functions to get status icon and class
        function getStatusIcon(status) {
            switch (status) {
                case 'added':
                    return 'fas fa-plus';
                case 'modified':
                    return 'fas fa-pencil-alt';
                case 'deleted':
                    return 'fas fa-trash';
                case 'renamed':
                    return 'fas fa-file-signature';
                default:
                    return 'fas fa-file';
            }
        }
        
        function getStatusClass(status) {
            switch (status) {
                case 'added':
                    return 'added';
                case 'modified':
                    return 'modified';
                case 'deleted':
                    return 'deleted';
                case 'renamed':
                    return 'renamed';
                default:
                    return '';
            }
        }
    }

    /**
     * Revert a commit
     * 
     * @param {string} commitHash - Commit hash to revert
     */
    async revertCommit(commitHash) {
        try {
            // Revert commit
            await this.apiClient.post('/api/v1/developer/git/revert', {
                commit: commitHash
            });
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Refresh commits
            await this.fetchCommits();
            
            // Trigger commit reverted event
            this.triggerEvent('commitReverted', commitHash);
            
            // Show success notification
            showToast('Commit reverted successfully', 'success');
        } catch (error) {
            console.error(`Failed to revert commit ${commitHash}:`, error);
            
            // Show error notification
            showToast(`Failed to revert commit: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to revert commit', 
                'GIT_REVERT_FAILED',
                { originalError: error, commitHash }
            );
        }
    }

    /**
     * Show create branch dialog
     */
    showCreateBranchDialog() {
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-dialog git-branch-modal">
                <div class="modal-header">
                    <div class="modal-title">Create New Branch</div>
                    <button class="modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-content">
                    <div class="form-group">
                        <label for="branch-name">Branch Name</label>
                        <input type="text" id="branch-name" class="form-control" placeholder="feature/new-feature">
                    </div>
                    <div class="form-group">
                        <label for="branch-source">Source Branch</label>
                        <select id="branch-source" class="form-control">
                            <option value="${this.currentBranch}">${this.currentBranch} (current)</option>
                            ${this.branches.filter(b => b.name !== this.currentBranch).map(b => `
                                <option value="${b.name}">${b.name}</option>
                            `).join('')}
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="checkbox-label">
                            <input type="checkbox" id="checkout-branch">
                            Checkout new branch after creation
                        </label>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="modal-action create">Create Branch</button>
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
        
        // Add create branch event listener
        const createButton = modal.querySelector('.modal-action.create');
        if (createButton) {
            createButton.addEventListener('click', async () => {
                const branchName = document.getElementById('branch-name').value.trim();
                const sourceBranch = document.getElementById('branch-source').value;
                const checkout = document.getElementById('checkout-branch').checked;
                
                if (!branchName) {
                    // Show error notification
                    showToast('Branch name cannot be empty', 'error');
                    return;
                }
                
                try {
                    // Create branch
                    await this.createBranch(branchName, sourceBranch, checkout);
                    
                    // Close modal
                    document.body.removeChild(modal);
                } catch (error) {
                    // Error will be handled by createBranch method
                }
            });
        }
    }

    /**
     * Create a new branch
     * 
     * @param {string} branchName - Name of the new branch
     * @param {string} sourceBranch - Source branch to create from
     * @param {boolean} checkout - Whether to checkout the new branch
     */
    async createBranch(branchName, sourceBranch, checkout = false) {
        try {
            // Create branch
            await this.apiClient.post('/api/v1/developer/git/branch', {
                name: branchName,
                source: sourceBranch
            });
            
            // Checkout branch if requested
            if (checkout) {
                await this.checkoutBranch(branchName);
            }
            
            // Refresh branches
            await this.fetchBranches();
            
            // Trigger branch created event
            this.triggerEvent('branchCreated', {
                name: branchName,
                source: sourceBranch,
                checkout
            });
            
            // Show success notification
            showToast(`Branch '${branchName}' created successfully${checkout ? ' and checked out' : ''}`, 'success');
        } catch (error) {
            console.error(`Failed to create branch ${branchName}:`, error);
            
            // Show error notification
            showToast(`Failed to create branch: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to create branch', 
                'GIT_CREATE_BRANCH_FAILED',
                { originalError: error, branchName, sourceBranch }
            );
        }
    }

    /**
     * Checkout a branch
     * 
     * @param {string} branchName - Name of the branch to checkout
     */
    async checkoutBranch(branchName) {
        try {
            // Checkout branch
            await this.apiClient.post('/api/v1/developer/git/checkout', {
                branch: branchName
            });
            
            // Update current branch
            this.currentBranch = branchName;
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Refresh branches
            await this.fetchBranches();
            
            // Trigger branch checked out event
            this.triggerEvent('branchCheckedOut', branchName);
            
            // Show success notification
            showToast(`Branch '${branchName}' checked out successfully`, 'success');
        } catch (error) {
            console.error(`Failed to checkout branch ${branchName}:`, error);
            
            // Show error notification
            showToast(`Failed to checkout branch: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to checkout branch', 
                'GIT_CHECKOUT_FAILED',
                { originalError: error, branchName }
            );
        }
    }

    /**
     * Merge a branch
     * 
     * @param {string} branchName - Name of the branch to merge
     */
    async mergeBranch(branchName) {
        try {
            // Show confirmation dialog
            if (!confirm(`Are you sure you want to merge branch '${branchName}' into '${this.currentBranch}'?`)) {
                return;
            }
            
            // Merge branch
            await this.apiClient.post('/api/v1/developer/git/merge', {
                branch: branchName
            });
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Refresh commits
            await this.fetchCommits();
            
            // Trigger branch merged event
            this.triggerEvent('branchMerged', {
                source: branchName,
                target: this.currentBranch
            });
            
            // Show success notification
            showToast(`Branch '${branchName}' merged into '${this.currentBranch}' successfully`, 'success');
        } catch (error) {
            console.error(`Failed to merge branch ${branchName}:`, error);
            
            // Show error notification
            showToast(`Failed to merge branch: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to merge branch', 
                'GIT_MERGE_FAILED',
                { originalError: error, branchName }
            );
        }
    }

    /**
     * Pull changes from remote
     */
    async pull() {
        try {
            // Pull changes
            await this.apiClient.post('/api/v1/developer/git/pull');
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Refresh commits
            await this.fetchCommits();
            
            // Trigger pull completed event
            this.triggerEvent('pullCompleted');
            
            // Show success notification
            showToast('Pulled changes successfully', 'success');
        } catch (error) {
            console.error('Failed to pull changes:', error);
            
            // Show error notification
            showToast(`Failed to pull changes: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to pull changes', 
                'GIT_PULL_FAILED',
                { originalError: error }
            );
        }
    }

    /**
     * Push changes to remote
     */
    async push() {
        try {
            // Push changes
            await this.apiClient.post('/api/v1/developer/git/push');
            
            // Refresh Git status
            await this.refreshGitStatus();
            
            // Trigger push completed event
            this.triggerEvent('pushCompleted');
            
            // Show success notification
            showToast('Pushed changes successfully', 'success');
        } catch (error) {
            console.error('Failed to push changes:', error);
            
            // Show error notification
            showToast(`Failed to push changes: ${error.message}`, 'error');
            
            throw new AppError(
                'Failed to push changes', 
                'GIT_PUSH_FAILED',
                { originalError: error }
            );
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
window.GitIntegration = GitIntegration;
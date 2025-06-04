/**
 * Component Showcase
 * 
 * This module demonstrates how to use the React components.
 */

// Component Showcase container
class ComponentShowcase {
    constructor() {
        this.currentTab = 0;
        this.isModalOpen = false;
        this.toasts = [];
    }

    initialize() {
        // Create a container for the showcase
        const showcaseContainer = document.createElement('div');
        showcaseContainer.id = 'component-showcase-container';
        showcaseContainer.style.display = 'none';
        document.body.appendChild(showcaseContainer);

        // Add a button to the developer header to toggle the showcase
        const developerHeader = document.querySelector('.developer-header');
        if (developerHeader) {
            const showcaseButton = document.createElement('div');
            showcaseButton.className = 'developer-nav-item';
            showcaseButton.innerHTML = `
                <i class="fas fa-palette"></i>
                <span>Components</span>
            `;
            
            showcaseButton.addEventListener('click', () => {
                this.toggleShowcase();
            });
            
            developerHeader.querySelector('.developer-nav').appendChild(showcaseButton);
        }

        return this;
    }

    toggleShowcase() {
        const container = document.getElementById('component-showcase-container');
        
        if (container.style.display === 'none') {
            container.style.display = 'block';
            this.renderShowcase();
            
            // Hide the main content
            const workspaceContent = document.querySelector('.workspace-content');
            if (workspaceContent) {
                workspaceContent.style.display = 'none';
            }
        } else {
            container.style.display = 'none';
            
            // Show the main content
            const workspaceContent = document.querySelector('.workspace-content');
            if (workspaceContent) {
                workspaceContent.style.display = 'flex';
            }
        }
    }

    renderShowcase() {
        const { Button, Tabs, Card, Input, Select, Checkbox, RadioGroup, Alert, Toast, Modal, Badge, Progress, Dropdown, DataTable, DragDropContainer } = window.ReactComponents;
        
        // Component categories
        const componentTabs = [
            'Basic Inputs',
            'Layout',
            'Feedback',
            'Data Display',
            'Navigation',
        ];

        // Define the showcase element
        const showcase = (
            <div className="p-6 bg-gray-100 min-h-screen">
                <div className="max-w-6xl mx-auto">
                    <div className="mb-6 bg-white rounded-lg shadow p-4">
                        <h1 className="text-2xl font-bold mb-2">React Component Library</h1>
                        <p className="text-gray-600">
                            A collection of reusable React components for building the Developer Studio UI.
                        </p>
                    </div>
                    
                    <Tabs
                        tabs={componentTabs}
                        activeTab={this.currentTab}
                        onChange={(index) => {
                            this.currentTab = index;
                            this.renderShowcase();
                        }}
                        className="mb-6 bg-white rounded-lg shadow p-4"
                    />
                    
                    {this.currentTab === 0 && (
                        <div className="space-y-6">
                            {/* Buttons */}
                            <Card
                                title="Buttons"
                                subtitle="Various button styles and variants"
                                className="shadow"
                            >
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Button Variants</h3>
                                        <div className="flex flex-wrap gap-2">
                                            <Button variant="primary">Primary</Button>
                                            <Button variant="secondary">Secondary</Button>
                                            <Button variant="success">Success</Button>
                                            <Button variant="danger">Danger</Button>
                                            <Button variant="outline">Outline</Button>
                                            <Button variant="ghost">Ghost</Button>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Button Sizes</h3>
                                        <div className="flex flex-wrap items-center gap-2">
                                            <Button size="small" variant="primary">Small</Button>
                                            <Button size="medium" variant="primary">Medium</Button>
                                            <Button size="large" variant="primary">Large</Button>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Button with Icons</h3>
                                        <div className="flex flex-wrap gap-2">
                                            <Button variant="primary" icon={<i className="fas fa-save"></i>}>Save</Button>
                                            <Button variant="secondary" icon={<i className="fas fa-trash"></i>}>Delete</Button>
                                            <Button variant="success" icon={<i className="fas fa-check"></i>}>Approve</Button>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Button States</h3>
                                        <div className="flex flex-wrap gap-2">
                                            <Button variant="primary">Normal</Button>
                                            <Button variant="primary" disabled>Disabled</Button>
                                            <Button variant="primary" fullWidth>Full Width</Button>
                                        </div>
                                    </div>
                                </div>
                            </Card>
                            
                            {/* Form Inputs */}
                            <Card
                                title="Form Inputs"
                                subtitle="Text fields, selects, checkboxes, and radio buttons"
                                className="shadow"
                            >
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Text Inputs</h3>
                                        <div className="space-y-4">
                                            <Input
                                                label="Standard Input"
                                                placeholder="Enter text here"
                                            />
                                            
                                            <Input
                                                label="Required Input"
                                                placeholder="Required field"
                                                required
                                            />
                                            
                                            <Input
                                                label="Disabled Input"
                                                placeholder="This field is disabled"
                                                disabled
                                                value="Disabled value"
                                            />
                                            
                                            <Input
                                                label="Input with Error"
                                                placeholder="Error field"
                                                error
                                                helperText="This field has an error"
                                            />
                                            
                                            <Input
                                                label="Input with Helper Text"
                                                placeholder="With helper text"
                                                helperText="This is some helpful information"
                                            />
                                            
                                            <Input
                                                label="Input with Icons"
                                                placeholder="Search..."
                                                startIcon={<i className="fas fa-search"></i>}
                                                endIcon={<i className="fas fa-times"></i>}
                                            />
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Select Input</h3>
                                        <Select
                                            label="Select an option"
                                            options={[
                                                { value: '', label: 'Select an option', disabled: true },
                                                { value: 'option1', label: 'Option 1' },
                                                { value: 'option2', label: 'Option 2' },
                                                { value: 'option3', label: 'Option 3' },
                                            ]}
                                        />
                                        
                                        <h3 className="text-lg font-medium mt-6 mb-2">Checkbox</h3>
                                        <div className="space-y-2">
                                            <Checkbox
                                                label="Default Checkbox"
                                                id="checkbox1"
                                            />
                                            
                                            <Checkbox
                                                label="Checked Checkbox"
                                                id="checkbox2"
                                                checked
                                            />
                                            
                                            <Checkbox
                                                label="Disabled Checkbox"
                                                id="checkbox3"
                                                disabled
                                            />
                                        </div>
                                        
                                        <h3 className="text-lg font-medium mt-6 mb-2">Radio Group</h3>
                                        <RadioGroup
                                            label="Select an option"
                                            name="radio-group"
                                            options={[
                                                { value: 'option1', label: 'Option 1' },
                                                { value: 'option2', label: 'Option 2' },
                                                { value: 'option3', label: 'Option 3', disabled: true },
                                            ]}
                                            value="option1"
                                        />
                                    </div>
                                </div>
                            </Card>
                        </div>
                    )}
                    
                    {this.currentTab === 1 && (
                        <div className="space-y-6">
                            {/* Cards */}
                            <Card
                                title="Card Variants"
                                subtitle="Different card styles for content organization"
                                className="shadow"
                            >
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <Card
                                        title="Default Card"
                                        subtitle="The standard card design"
                                        variant="default"
                                    >
                                        <p className="text-gray-600">
                                            This is the content of a default card. It can contain any elements including text, buttons, and other components.
                                        </p>
                                    </Card>
                                    
                                    <Card
                                        title="Outlined Card"
                                        subtitle="Card with a border outline"
                                        variant="outlined"
                                    >
                                        <p className="text-gray-600">
                                            This is the content of an outlined card. It has a border around it to provide visual separation.
                                        </p>
                                    </Card>
                                    
                                    <Card
                                        title="Elevated Card"
                                        subtitle="Card with a shadow effect"
                                        variant="elevated"
                                    >
                                        <p className="text-gray-600">
                                            This is the content of an elevated card. It has a shadow to create a raised effect.
                                        </p>
                                    </Card>
                                    
                                    <Card
                                        title="Card with Actions"
                                        subtitle="Card with action buttons"
                                        variant="default"
                                        actions={
                                            <>
                                                <Button variant="ghost" size="small">Cancel</Button>
                                                <Button variant="primary" size="small">Save</Button>
                                            </>
                                        }
                                    >
                                        <p className="text-gray-600">
                                            This card has action buttons in the footer. They can be used for form submissions or other actions.
                                        </p>
                                    </Card>
                                </div>
                            </Card>
                            
                            {/* Drag and Drop */}
                            <Card
                                title="Drag and Drop"
                                subtitle="File upload with drag and drop support"
                                className="shadow"
                            >
                                <DragDropContainer
                                    onDrop={(files) => {
                                        console.log('Files dropped:', files);
                                        this.addToast({
                                            message: `Dropped ${files.length} file${files.length === 1 ? '' : 's'}: ${files.map(f => f.name).join(', ')}`,
                                            severity: 'success'
                                        });
                                    }}
                                    acceptTypes={['image/*', 'application/pdf']}
                                    className="text-center py-10"
                                >
                                    <div className="space-y-2">
                                        <i className="fas fa-upload text-4xl text-blue-500"></i>
                                        <h3 className="text-lg font-medium">Drop files here</h3>
                                        <p className="text-sm text-gray-500">Drag and drop files here, or click to select files</p>
                                        <p className="text-xs text-gray-400">Supports: Images, PDFs</p>
                                    </div>
                                </DragDropContainer>
                            </Card>
                        </div>
                    )}
                    
                    {this.currentTab === 2 && (
                        <div className="space-y-6">
                            {/* Alerts */}
                            <Card
                                title="Alerts"
                                subtitle="Informational messages for users"
                                className="shadow"
                            >
                                <div className="space-y-4">
                                    <Alert severity="info" title="Information">
                                        This is an informational message to notify the user.
                                    </Alert>
                                    
                                    <Alert severity="success" title="Success">
                                        Operation completed successfully. Your changes have been saved.
                                    </Alert>
                                    
                                    <Alert severity="warning" title="Warning">
                                        Please review your input before proceeding. Some fields may require attention.
                                    </Alert>
                                    
                                    <Alert severity="error" title="Error">
                                        An error occurred while processing your request. Please try again.
                                    </Alert>
                                    
                                    <Alert 
                                        severity="info" 
                                        title="Dismissible Alert"
                                        onClose={() => console.log('Alert closed')}
                                    >
                                        This alert can be dismissed by clicking the close button.
                                    </Alert>
                                </div>
                            </Card>
                            
                            {/* Toasts */}
                            <Card
                                title="Toasts"
                                subtitle="Temporary notifications for user feedback"
                                className="shadow"
                            >
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Toast Types</h3>
                                        <div className="space-y-2">
                                            <Button 
                                                variant="outline" 
                                                fullWidth
                                                onClick={() => this.addToast({
                                                    message: 'This is an information message',
                                                    severity: 'info'
                                                })}
                                            >
                                                Show Info Toast
                                            </Button>
                                            
                                            <Button 
                                                variant="outline" 
                                                fullWidth
                                                onClick={() => this.addToast({
                                                    message: 'Operation completed successfully',
                                                    severity: 'success'
                                                })}
                                            >
                                                Show Success Toast
                                            </Button>
                                            
                                            <Button 
                                                variant="outline" 
                                                fullWidth
                                                onClick={() => this.addToast({
                                                    message: 'Please check your input',
                                                    severity: 'warning'
                                                })}
                                            >
                                                Show Warning Toast
                                            </Button>
                                            
                                            <Button 
                                                variant="outline" 
                                                fullWidth
                                                onClick={() => this.addToast({
                                                    message: 'An error occurred',
                                                    severity: 'error'
                                                })}
                                            >
                                                Show Error Toast
                                            </Button>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Toast Positions</h3>
                                        <div className="space-y-2">
                                            <Button 
                                                variant="outline" 
                                                fullWidth
                                                onClick={() => this.addToast({
                                                    message: 'Top-left position',
                                                    severity: 'info',
                                                    position: 'top-left'
                                                })}
                                            >
                                                Top Left
                                            </Button>
                                            
                                            <Button 
                                                variant="outline" 
                                                fullWidth
                                                onClick={() => this.addToast({
                                                    message: 'Top-right position',
                                                    severity: 'info',
                                                    position: 'top-right'
                                                })}
                                            >
                                                Top Right
                                            </Button>
                                            
                                            <Button 
                                                variant="outline" 
                                                fullWidth
                                                onClick={() => this.addToast({
                                                    message: 'Bottom-left position',
                                                    severity: 'info',
                                                    position: 'bottom-left'
                                                })}
                                            >
                                                Bottom Left
                                            </Button>
                                            
                                            <Button 
                                                variant="outline" 
                                                fullWidth
                                                onClick={() => this.addToast({
                                                    message: 'Bottom-right position',
                                                    severity: 'info',
                                                    position: 'bottom-right'
                                                })}
                                            >
                                                Bottom Right
                                            </Button>
                                        </div>
                                    </div>
                                </div>
                            </Card>
                            
                            {/* Modal */}
                            <Card
                                title="Modal"
                                subtitle="Dialog windows for focused interactions"
                                className="shadow"
                            >
                                <div className="space-y-4">
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        <Button 
                                            variant="primary"
                                            onClick={() => {
                                                this.isModalOpen = true;
                                                this.modalSize = 'small';
                                                this.renderShowcase();
                                            }}
                                        >
                                            Open Small Modal
                                        </Button>
                                        
                                        <Button 
                                            variant="primary"
                                            onClick={() => {
                                                this.isModalOpen = true;
                                                this.modalSize = 'medium';
                                                this.renderShowcase();
                                            }}
                                        >
                                            Open Medium Modal
                                        </Button>
                                        
                                        <Button 
                                            variant="primary"
                                            onClick={() => {
                                                this.isModalOpen = true;
                                                this.modalSize = 'large';
                                                this.renderShowcase();
                                            }}
                                        >
                                            Open Large Modal
                                        </Button>
                                    </div>
                                    
                                    {this.isModalOpen && (
                                        <Modal
                                            isOpen={this.isModalOpen}
                                            onClose={() => {
                                                this.isModalOpen = false;
                                                this.renderShowcase();
                                            }}
                                            title="Modal Dialog"
                                            size={this.modalSize}
                                            footer={
                                                <div className="flex justify-end space-x-2">
                                                    <Button 
                                                        variant="secondary"
                                                        onClick={() => {
                                                            this.isModalOpen = false;
                                                            this.renderShowcase();
                                                        }}
                                                    >
                                                        Cancel
                                                    </Button>
                                                    <Button 
                                                        variant="primary"
                                                        onClick={() => {
                                                            this.isModalOpen = false;
                                                            this.addToast({
                                                                message: 'Modal action confirmed',
                                                                severity: 'success'
                                                            });
                                                            this.renderShowcase();
                                                        }}
                                                    >
                                                        Confirm
                                                    </Button>
                                                </div>
                                            }
                                        >
                                            <div className="space-y-4">
                                                <p className="text-gray-600">
                                                    This is a modal dialog window. It's useful for focused interactions that require user attention.
                                                </p>
                                                
                                                <Input
                                                    label="Sample Input"
                                                    placeholder="Enter text here"
                                                />
                                                
                                                <Select
                                                    label="Sample Select"
                                                    options={[
                                                        { value: '', label: 'Select an option', disabled: true },
                                                        { value: 'option1', label: 'Option 1' },
                                                        { value: 'option2', label: 'Option 2' },
                                                        { value: 'option3', label: 'Option 3' },
                                                    ]}
                                                />
                                            </div>
                                        </Modal>
                                    )}
                                </div>
                            </Card>
                            
                            {/* Progress */}
                            <Card
                                title="Progress Indicators"
                                subtitle="Visual feedback for loading and progress states"
                                className="shadow"
                            >
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Determinate Progress</h3>
                                        <div className="space-y-4">
                                            <Progress value={25} label />
                                            <Progress value={50} color="success" label />
                                            <Progress value={75} color="warning" label />
                                            <Progress value={100} color="error" label />
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Indeterminate Progress</h3>
                                        <div className="space-y-4">
                                            <Progress variant="indeterminate" />
                                            <Progress variant="indeterminate" color="success" />
                                            <Progress variant="indeterminate" color="warning" />
                                            <Progress variant="indeterminate" color="error" />
                                        </div>
                                        
                                        <h3 className="text-lg font-medium mt-6 mb-2">Progress Thickness</h3>
                                        <div className="space-y-4">
                                            <Progress value={50} thickness="thin" />
                                            <Progress value={50} thickness="medium" />
                                            <Progress value={50} thickness="thick" />
                                        </div>
                                    </div>
                                </div>
                            </Card>
                        </div>
                    )}
                    
                    {this.currentTab === 3 && (
                        <div className="space-y-6">
                            {/* Badges */}
                            <Card
                                title="Badges"
                                subtitle="Small status indicators and counters"
                                className="shadow"
                            >
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Badge Colors</h3>
                                        <div className="flex flex-wrap gap-4">
                                            <Badge content={5} color="primary">
                                                <Button variant="outline">Messages</Button>
                                            </Badge>
                                            
                                            <Badge content={2} color="secondary">
                                                <Button variant="outline">Notifications</Button>
                                            </Badge>
                                            
                                            <Badge content={3} color="success">
                                                <Button variant="outline">Updates</Button>
                                            </Badge>
                                            
                                            <Badge content={1} color="warning">
                                                <Button variant="outline">Warnings</Button>
                                            </Badge>
                                            
                                            <Badge content={4} color="error">
                                                <Button variant="outline">Errors</Button>
                                            </Badge>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Badge Variants</h3>
                                        <div className="flex flex-wrap gap-4">
                                            <Badge content={5} variant="standard" color="primary">
                                                <Button variant="outline">Standard</Button>
                                            </Badge>
                                            
                                            <Badge content={5} variant="outlined" color="primary">
                                                <Button variant="outline">Outlined</Button>
                                            </Badge>
                                            
                                            <Badge variant="dot" color="primary">
                                                <Button variant="outline">Dot</Button>
                                            </Badge>
                                            
                                            <Badge variant="dot" color="error">
                                                <Button variant="outline">Error Dot</Button>
                                            </Badge>
                                        </div>
                                    </div>
                                </div>
                            </Card>
                            
                            {/* Data Table */}
                            <Card
                                title="Data Table"
                                subtitle="For displaying tabular data with sorting and pagination"
                                className="shadow"
                            >
                                <DataTable
                                    columns={[
                                        { key: 'id', header: 'ID', sortable: true },
                                        { key: 'name', header: 'Name', sortable: true },
                                        { key: 'email', header: 'Email', sortable: true },
                                        { key: 'role', header: 'Role', sortable: true },
                                        { 
                                            key: 'status', 
                                            header: 'Status', 
                                            sortable: true,
                                            render: (value) => (
                                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                                    value === 'Active' ? 'bg-green-100 text-green-800' :
                                                    value === 'Inactive' ? 'bg-gray-100 text-gray-800' :
                                                    'bg-yellow-100 text-yellow-800'
                                                }`}>
                                                    {value}
                                                </span>
                                            )
                                        },
                                        { 
                                            key: 'actions', 
                                            header: 'Actions',
                                            render: (_, row) => (
                                                <div className="flex space-x-2">
                                                    <Button 
                                                        variant="ghost" 
                                                        size="small"
                                                        icon={<i className="fas fa-edit"></i>}
                                                        onClick={() => console.log('Edit', row)}
                                                    >
                                                        Edit
                                                    </Button>
                                                    <Button 
                                                        variant="ghost" 
                                                        size="small" 
                                                        icon={<i className="fas fa-trash"></i>}
                                                        onClick={() => console.log('Delete', row)}
                                                    >
                                                        Delete
                                                    </Button>
                                                </div>
                                            )
                                        }
                                    ]}
                                    data={[
                                        { id: 1, name: 'John Doe', email: 'john@example.com', role: 'Admin', status: 'Active' },
                                        { id: 2, name: 'Jane Smith', email: 'jane@example.com', role: 'Editor', status: 'Active' },
                                        { id: 3, name: 'Bob Johnson', email: 'bob@example.com', role: 'Viewer', status: 'Inactive' },
                                        { id: 4, name: 'Alice Brown', email: 'alice@example.com', role: 'Editor', status: 'Pending' },
                                        { id: 5, name: 'Charlie Davis', email: 'charlie@example.com', role: 'Viewer', status: 'Active' },
                                    ]}
                                    pagination={true}
                                    pageSize={2}
                                />
                            </Card>
                        </div>
                    )}
                    
                    {this.currentTab === 4 && (
                        <div className="space-y-6">
                            {/* Tabs */}
                            <Card
                                title="Tabs"
                                subtitle="Navigation tabs for switching between views"
                                className="shadow"
                            >
                                <div className="space-y-6">
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Default Tabs</h3>
                                        <Tabs
                                            tabs={['Home', 'Profile', 'Settings', 'Help']}
                                            activeTab={0}
                                            onChange={(index) => console.log('Tab changed to', index)}
                                        />
                                        <div className="p-4 border border-gray-200 rounded-b">
                                            <p>Content for the Home tab</p>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Pills Tabs</h3>
                                        <Tabs
                                            tabs={['Home', 'Profile', 'Settings', 'Help']}
                                            activeTab={1}
                                            onChange={(index) => console.log('Tab changed to', index)}
                                            variant="pills"
                                        />
                                        <div className="p-4 border border-gray-200 rounded">
                                            <p>Content for the Profile tab</p>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-lg font-medium mb-2">Underline Tabs</h3>
                                        <Tabs
                                            tabs={['Home', 'Profile', 'Settings', 'Help']}
                                            activeTab={2}
                                            onChange={(index) => console.log('Tab changed to', index)}
                                            variant="underline"
                                        />
                                        <div className="p-4 border border-gray-200 rounded-b">
                                            <p>Content for the Settings tab</p>
                                        </div>
                                    </div>
                                </div>
                            </Card>
                            
                            {/* Dropdown */}
                            <Card
                                title="Dropdown"
                                subtitle="Contextual menus for additional options"
                                className="shadow"
                            >
                                <div className="flex flex-wrap gap-6">
                                    <Dropdown
                                        trigger={
                                            <Button variant="primary">
                                                Actions <i className="fas fa-chevron-down ml-2"></i>
                                            </Button>
                                        }
                                        items={[
                                            { label: 'Edit', icon: <i className="fas fa-edit"></i>, onClick: () => console.log('Edit clicked') },
                                            { label: 'Duplicate', icon: <i className="fas fa-copy"></i>, onClick: () => console.log('Duplicate clicked') },
                                            { divider: true },
                                            { label: 'Archive', icon: <i className="fas fa-archive"></i>, onClick: () => console.log('Archive clicked') },
                                            { label: 'Delete', icon: <i className="fas fa-trash"></i>, onClick: () => console.log('Delete clicked') },
                                        ]}
                                    />
                                    
                                    <Dropdown
                                        trigger={
                                            <Button variant="outline">
                                                User <i className="fas fa-chevron-down ml-2"></i>
                                            </Button>
                                        }
                                        items={[
                                            { label: 'Profile', icon: <i className="fas fa-user"></i>, onClick: () => console.log('Profile clicked') },
                                            { label: 'Settings', icon: <i className="fas fa-cog"></i>, onClick: () => console.log('Settings clicked') },
                                            { divider: true },
                                            { label: 'Help', icon: <i className="fas fa-question-circle"></i>, onClick: () => console.log('Help clicked') },
                                            { label: 'Logout', icon: <i className="fas fa-sign-out-alt"></i>, onClick: () => console.log('Logout clicked') },
                                        ]}
                                    />
                                    
                                    <Dropdown
                                        trigger={
                                            <Button variant="ghost" icon={<i className="fas fa-ellipsis-v"></i>}>
                                            </Button>
                                        }
                                        items={[
                                            { label: 'View', icon: <i className="fas fa-eye"></i>, onClick: () => console.log('View clicked') },
                                            { label: 'Edit', icon: <i className="fas fa-edit"></i>, onClick: () => console.log('Edit clicked') },
                                            { label: 'Delete', icon: <i className="fas fa-trash"></i>, onClick: () => console.log('Delete clicked') },
                                            { label: 'Disabled Option', icon: <i className="fas fa-ban"></i>, disabled: true },
                                        ]}
                                    />
                                </div>
                            </Card>
                        </div>
                    )}
                </div>
                
                {/* Render active toasts */}
                {this.toasts.map((toast, index) => (
                    <Toast
                        key={index}
                        message={toast.message}
                        severity={toast.severity}
                        position={toast.position || 'bottom-right'}
                        onClose={() => {
                            this.toasts = this.toasts.filter((_, i) => i !== index);
                            this.renderShowcase();
                        }}
                    />
                ))}
            </div>
        );
        
        // Render the showcase
        window.renderReactComponent(showcase);
    }

    addToast(toast) {
        this.toasts.push(toast);
        this.renderShowcase();
        
        // Auto-remove after duration
        setTimeout(() => {
            this.toasts = this.toasts.filter(t => t !== toast);
            this.renderShowcase();
        }, toast.duration || 3000);
    }
}

// Export to window
window.ComponentShowcase = ComponentShowcase;
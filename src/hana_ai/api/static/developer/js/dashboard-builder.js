/**
 * Dashboard Builder
 * 
 * Interactive drag-and-drop dashboard builder with customizable visualizations
 * for data exploration in the Developer Studio.
 */

// Create the Dashboard Builder React component
const DashboardBuilder = (() => {
    // Dashboard widget types
    const WIDGET_TYPES = {
        LINE_CHART: 'line_chart',
        BAR_CHART: 'bar_chart',
        PIE_CHART: 'pie_chart',
        SCATTER_PLOT: 'scatter_plot',
        DATA_TABLE: 'data_table',
        METRIC_CARD: 'metric_card',
        TEXT_BLOCK: 'text_block',
        FILTER_CONTROL: 'filter_control',
    };

    // Dashboard Builder component
    const DashboardBuilderComponent = ({ apiClient, dataSourceId }) => {
        const { Button, Card, Input, Select, Tabs, Modal, Toast, Dropdown } = window.ReactComponents;
        
        // State
        const [dashboards, setDashboards] = React.useState([]);
        const [activeDashboard, setActiveDashboard] = React.useState(null);
        const [widgets, setWidgets] = React.useState([]);
        const [dataSources, setDataSources] = React.useState([]);
        const [isDragging, setIsDragging] = React.useState(false);
        const [activeTab, setActiveTab] = React.useState(0);
        const [isAddWidgetModalOpen, setIsAddWidgetModalOpen] = React.useState(false);
        const [currentWidgetConfig, setCurrentWidgetConfig] = React.useState(null);
        const [toasts, setToasts] = React.useState([]);
        const [editMode, setEditMode] = React.useState(true);
        const [availableColumns, setAvailableColumns] = React.useState([]);
        
        // Dashboard grid configuration
        const gridRef = React.useRef(null);
        const gridGap = 10;
        const gridColumns = 12;
        const cellSize = 80;
        const cellHeight = 60;
        
        // Load data on mount
        React.useEffect(() => {
            loadDashboards();
            loadDataSources();
        }, []);
        
        // Load dashboard columns when data source changes
        React.useEffect(() => {
            if (activeDashboard && activeDashboard.dataSourceId) {
                loadDataSourceColumns(activeDashboard.dataSourceId);
            }
        }, [activeDashboard]);
        
        // Load dashboards
        const loadDashboards = async () => {
            try {
                // This would normally fetch from the API
                // const response = await apiClient.get('/api/v1/dashboards');
                // For demo, we'll use some sample dashboards
                const sampleDashboards = [
                    {
                        id: 'dashboard1',
                        name: 'Sales Analytics',
                        description: 'Overview of sales performance',
                        dataSourceId: 'sales_data',
                        created: new Date().toISOString(),
                        lastModified: new Date().toISOString(),
                    },
                    {
                        id: 'dashboard2',
                        name: 'Customer Insights',
                        description: 'Customer behavior and engagement',
                        dataSourceId: 'customer_data',
                        created: new Date().toISOString(),
                        lastModified: new Date().toISOString(),
                    },
                ];
                
                setDashboards(sampleDashboards);
                
                // Set the first dashboard as active if none selected
                if (sampleDashboards.length > 0 && !activeDashboard) {
                    setActiveDashboard(sampleDashboards[0]);
                    loadDashboardWidgets(sampleDashboards[0].id);
                }
            } catch (error) {
                console.error('Failed to load dashboards:', error);
                showToast('Failed to load dashboards', 'error');
            }
        };
        
        // Load data sources
        const loadDataSources = async () => {
            try {
                // This would normally fetch from the API
                // const response = await apiClient.get('/api/v1/data-sources');
                // For demo, we'll use some sample data sources
                const sampleDataSources = [
                    {
                        id: 'sales_data',
                        name: 'Sales Data',
                        type: 'table',
                        connection: 'HANA',
                    },
                    {
                        id: 'customer_data',
                        name: 'Customer Data',
                        type: 'table',
                        connection: 'HANA',
                    },
                    {
                        id: 'product_data',
                        name: 'Product Data',
                        type: 'table',
                        connection: 'HANA',
                    },
                ];
                
                setDataSources(sampleDataSources);
            } catch (error) {
                console.error('Failed to load data sources:', error);
                showToast('Failed to load data sources', 'error');
            }
        };
        
        // Load data source columns
        const loadDataSourceColumns = async (dataSourceId) => {
            try {
                // This would normally fetch from the API
                // const response = await apiClient.get(`/api/v1/data-sources/${dataSourceId}/columns`);
                // For demo, we'll use some sample columns based on the data source
                let sampleColumns = [];
                
                if (dataSourceId === 'sales_data') {
                    sampleColumns = [
                        { name: 'date', type: 'date', label: 'Date' },
                        { name: 'product_id', type: 'string', label: 'Product ID' },
                        { name: 'product_name', type: 'string', label: 'Product Name' },
                        { name: 'category', type: 'string', label: 'Category' },
                        { name: 'quantity', type: 'number', label: 'Quantity' },
                        { name: 'unit_price', type: 'number', label: 'Unit Price' },
                        { name: 'total_amount', type: 'number', label: 'Total Amount' },
                        { name: 'discount', type: 'number', label: 'Discount' },
                        { name: 'customer_id', type: 'string', label: 'Customer ID' },
                        { name: 'region', type: 'string', label: 'Region' },
                    ];
                } else if (dataSourceId === 'customer_data') {
                    sampleColumns = [
                        { name: 'customer_id', type: 'string', label: 'Customer ID' },
                        { name: 'first_name', type: 'string', label: 'First Name' },
                        { name: 'last_name', type: 'string', label: 'Last Name' },
                        { name: 'email', type: 'string', label: 'Email' },
                        { name: 'phone', type: 'string', label: 'Phone' },
                        { name: 'address', type: 'string', label: 'Address' },
                        { name: 'city', type: 'string', label: 'City' },
                        { name: 'country', type: 'string', label: 'Country' },
                        { name: 'registration_date', type: 'date', label: 'Registration Date' },
                        { name: 'last_purchase', type: 'date', label: 'Last Purchase' },
                        { name: 'lifetime_value', type: 'number', label: 'Lifetime Value' },
                        { name: 'segment', type: 'string', label: 'Segment' },
                    ];
                } else if (dataSourceId === 'product_data') {
                    sampleColumns = [
                        { name: 'product_id', type: 'string', label: 'Product ID' },
                        { name: 'name', type: 'string', label: 'Name' },
                        { name: 'description', type: 'string', label: 'Description' },
                        { name: 'category', type: 'string', label: 'Category' },
                        { name: 'subcategory', type: 'string', label: 'Subcategory' },
                        { name: 'price', type: 'number', label: 'Price' },
                        { name: 'cost', type: 'number', label: 'Cost' },
                        { name: 'stock_quantity', type: 'number', label: 'Stock Quantity' },
                        { name: 'reorder_level', type: 'number', label: 'Reorder Level' },
                        { name: 'supplier_id', type: 'string', label: 'Supplier ID' },
                    ];
                }
                
                setAvailableColumns(sampleColumns);
            } catch (error) {
                console.error('Failed to load data source columns:', error);
                showToast('Failed to load columns', 'error');
            }
        };
        
        // Load dashboard widgets
        const loadDashboardWidgets = async (dashboardId) => {
            try {
                // This would normally fetch from the API
                // const response = await apiClient.get(`/api/v1/dashboards/${dashboardId}/widgets`);
                // For demo, we'll use some sample widgets
                let sampleWidgets = [];
                
                if (dashboardId === 'dashboard1') {
                    sampleWidgets = [
                        {
                            id: 'widget1',
                            type: WIDGET_TYPES.LINE_CHART,
                            title: 'Sales Over Time',
                            position: { x: 0, y: 0, w: 6, h: 4 },
                            config: {
                                dataSourceId: 'sales_data',
                                xAxis: 'date',
                                yAxis: 'total_amount',
                                groupBy: 'category',
                                aggregation: 'sum',
                            },
                        },
                        {
                            id: 'widget2',
                            type: WIDGET_TYPES.BAR_CHART,
                            title: 'Sales by Category',
                            position: { x: 6, y: 0, w: 6, h: 4 },
                            config: {
                                dataSourceId: 'sales_data',
                                xAxis: 'category',
                                yAxis: 'total_amount',
                                aggregation: 'sum',
                            },
                        },
                        {
                            id: 'widget3',
                            type: WIDGET_TYPES.PIE_CHART,
                            title: 'Sales Distribution',
                            position: { x: 0, y: 4, w: 4, h: 4 },
                            config: {
                                dataSourceId: 'sales_data',
                                dimension: 'category',
                                metric: 'total_amount',
                                aggregation: 'sum',
                            },
                        },
                        {
                            id: 'widget4',
                            type: WIDGET_TYPES.METRIC_CARD,
                            title: 'Total Sales',
                            position: { x: 4, y: 4, w: 2, h: 2 },
                            config: {
                                dataSourceId: 'sales_data',
                                metric: 'total_amount',
                                aggregation: 'sum',
                                format: '$#,##0',
                            },
                        },
                        {
                            id: 'widget5',
                            type: WIDGET_TYPES.METRIC_CARD,
                            title: 'Total Orders',
                            position: { x: 6, y: 4, w: 2, h: 2 },
                            config: {
                                dataSourceId: 'sales_data',
                                metric: 'order_id',
                                aggregation: 'count',
                                format: '#,##0',
                            },
                        },
                        {
                            id: 'widget6',
                            type: WIDGET_TYPES.DATA_TABLE,
                            title: 'Recent Sales',
                            position: { x: 8, y: 4, w: 4, h: 4 },
                            config: {
                                dataSourceId: 'sales_data',
                                columns: ['date', 'product_name', 'quantity', 'total_amount'],
                                sortBy: 'date',
                                sortDirection: 'desc',
                                limit: 10,
                            },
                        },
                        {
                            id: 'widget7',
                            type: WIDGET_TYPES.FILTER_CONTROL,
                            title: 'Filters',
                            position: { x: 0, y: 8, w: 12, h: 1 },
                            config: {
                                filters: [
                                    {
                                        column: 'date',
                                        type: 'date_range',
                                        label: 'Date Range',
                                    },
                                    {
                                        column: 'category',
                                        type: 'multi_select',
                                        label: 'Categories',
                                    },
                                    {
                                        column: 'region',
                                        type: 'multi_select',
                                        label: 'Regions',
                                    },
                                ],
                            },
                        },
                    ];
                } else if (dashboardId === 'dashboard2') {
                    sampleWidgets = [
                        {
                            id: 'widget1',
                            type: WIDGET_TYPES.BAR_CHART,
                            title: 'Customers by Segment',
                            position: { x: 0, y: 0, w: 6, h: 4 },
                            config: {
                                dataSourceId: 'customer_data',
                                xAxis: 'segment',
                                yAxis: 'customer_id',
                                aggregation: 'count',
                            },
                        },
                        {
                            id: 'widget2',
                            type: WIDGET_TYPES.PIE_CHART,
                            title: 'Customers by Country',
                            position: { x: 6, y: 0, w: 6, h: 4 },
                            config: {
                                dataSourceId: 'customer_data',
                                dimension: 'country',
                                metric: 'customer_id',
                                aggregation: 'count',
                            },
                        },
                        {
                            id: 'widget3',
                            type: WIDGET_TYPES.LINE_CHART,
                            title: 'Customer Registrations Over Time',
                            position: { x: 0, y: 4, w: 8, h: 4 },
                            config: {
                                dataSourceId: 'customer_data',
                                xAxis: 'registration_date',
                                yAxis: 'customer_id',
                                aggregation: 'count',
                                timeGranularity: 'month',
                            },
                        },
                        {
                            id: 'widget4',
                            type: WIDGET_TYPES.METRIC_CARD,
                            title: 'Total Customers',
                            position: { x: 8, y: 4, w: 4, h: 2 },
                            config: {
                                dataSourceId: 'customer_data',
                                metric: 'customer_id',
                                aggregation: 'count',
                                format: '#,##0',
                            },
                        },
                        {
                            id: 'widget5',
                            type: WIDGET_TYPES.METRIC_CARD,
                            title: 'Average Lifetime Value',
                            position: { x: 8, y: 6, w: 4, h: 2 },
                            config: {
                                dataSourceId: 'customer_data',
                                metric: 'lifetime_value',
                                aggregation: 'avg',
                                format: '$#,##0',
                            },
                        },
                        {
                            id: 'widget6',
                            type: WIDGET_TYPES.DATA_TABLE,
                            title: 'Top Customers by Value',
                            position: { x: 0, y: 8, w: 12, h: 4 },
                            config: {
                                dataSourceId: 'customer_data',
                                columns: ['customer_id', 'first_name', 'last_name', 'email', 'segment', 'lifetime_value'],
                                sortBy: 'lifetime_value',
                                sortDirection: 'desc',
                                limit: 10,
                            },
                        },
                    ];
                }
                
                setWidgets(sampleWidgets);
            } catch (error) {
                console.error('Failed to load dashboard widgets:', error);
                showToast('Failed to load widgets', 'error');
            }
        };
        
        // Create a new dashboard
        const createNewDashboard = () => {
            const newDashboard = {
                id: `dashboard${Date.now()}`,
                name: 'New Dashboard',
                description: 'A new dashboard',
                dataSourceId: dataSources.length > 0 ? dataSources[0].id : null,
                created: new Date().toISOString(),
                lastModified: new Date().toISOString(),
            };
            
            setDashboards([...dashboards, newDashboard]);
            setActiveDashboard(newDashboard);
            setWidgets([]);
            showToast('New dashboard created', 'success');
        };
        
        // Add a new widget
        const addWidget = (type) => {
            setCurrentWidgetConfig({
                id: `widget${Date.now()}`,
                type,
                title: getDefaultWidgetTitle(type),
                position: { x: 0, y: 0, w: getDefaultWidgetWidth(type), h: getDefaultWidgetHeight(type) },
                config: getDefaultWidgetConfig(type),
            });
            
            setIsAddWidgetModalOpen(true);
        };
        
        // Save widget configuration
        const saveWidgetConfig = () => {
            if (!currentWidgetConfig) return;
            
            const widgetExists = widgets.some(widget => widget.id === currentWidgetConfig.id);
            
            if (widgetExists) {
                // Update existing widget
                setWidgets(widgets.map(widget => 
                    widget.id === currentWidgetConfig.id ? currentWidgetConfig : widget
                ));
            } else {
                // Add new widget
                setWidgets([...widgets, currentWidgetConfig]);
            }
            
            setIsAddWidgetModalOpen(false);
            setCurrentWidgetConfig(null);
            showToast('Widget saved successfully', 'success');
        };
        
        // Delete a widget
        const deleteWidget = (widgetId) => {
            setWidgets(widgets.filter(widget => widget.id !== widgetId));
            showToast('Widget deleted', 'success');
        };
        
        // Edit a widget
        const editWidget = (widget) => {
            setCurrentWidgetConfig({...widget});
            setIsAddWidgetModalOpen(true);
        };
        
        // Update widget position
        const updateWidgetPosition = (widgetId, position) => {
            setWidgets(widgets.map(widget => 
                widget.id === widgetId ? {...widget, position} : widget
            ));
        };
        
        // Handle widget drag start
        const handleWidgetDragStart = (e, widgetId) => {
            e.dataTransfer.setData('text/plain', widgetId);
            setIsDragging(true);
            
            // Set drag image
            const dragElement = document.getElementById(`widget-${widgetId}`);
            if (dragElement) {
                e.dataTransfer.setDragImage(dragElement, 20, 20);
            }
        };
        
        // Handle widget drag over
        const handleGridDragOver = (e) => {
            e.preventDefault();
            if (!isDragging) return;
            
            // Calculate grid position from mouse coordinates
            if (gridRef.current) {
                const rect = gridRef.current.getBoundingClientRect();
                const x = Math.floor((e.clientX - rect.left) / (cellSize + gridGap));
                const y = Math.floor((e.clientY - rect.top) / (cellHeight + gridGap));
                
                // Update preview position
                // This would normally update a visual indicator of where the widget will be placed
            }
        };
        
        // Handle widget drop
        const handleGridDrop = (e) => {
            e.preventDefault();
            const widgetId = e.dataTransfer.getData('text/plain');
            setIsDragging(false);
            
            // Calculate grid position from mouse coordinates
            if (gridRef.current) {
                const rect = gridRef.current.getBoundingClientRect();
                const x = Math.floor((e.clientX - rect.left) / (cellSize + gridGap));
                const y = Math.floor((e.clientY - rect.top) / (cellHeight + gridGap));
                
                // Limit to grid bounds
                const boundedX = Math.max(0, Math.min(x, gridColumns - 1));
                const boundedY = Math.max(0, y);
                
                // Find the widget
                const widget = widgets.find(w => w.id === widgetId);
                if (widget) {
                    // Update widget position
                    updateWidgetPosition(widgetId, {
                        ...widget.position,
                        x: boundedX,
                        y: boundedY,
                    });
                }
            }
        };
        
        // Handle widget resize
        const handleWidgetResize = (widgetId, newSize) => {
            setWidgets(widgets.map(widget => 
                widget.id === widgetId ? {
                    ...widget, 
                    position: {
                        ...widget.position,
                        w: newSize.w,
                        h: newSize.h,
                    }
                } : widget
            ));
        };
        
        // Get default widget title based on type
        const getDefaultWidgetTitle = (type) => {
            switch (type) {
                case WIDGET_TYPES.LINE_CHART:
                    return 'Line Chart';
                case WIDGET_TYPES.BAR_CHART:
                    return 'Bar Chart';
                case WIDGET_TYPES.PIE_CHART:
                    return 'Pie Chart';
                case WIDGET_TYPES.SCATTER_PLOT:
                    return 'Scatter Plot';
                case WIDGET_TYPES.DATA_TABLE:
                    return 'Data Table';
                case WIDGET_TYPES.METRIC_CARD:
                    return 'Metric';
                case WIDGET_TYPES.TEXT_BLOCK:
                    return 'Text Block';
                case WIDGET_TYPES.FILTER_CONTROL:
                    return 'Filters';
                default:
                    return 'New Widget';
            }
        };
        
        // Get default widget width based on type
        const getDefaultWidgetWidth = (type) => {
            switch (type) {
                case WIDGET_TYPES.METRIC_CARD:
                    return 3;
                case WIDGET_TYPES.TEXT_BLOCK:
                    return 4;
                case WIDGET_TYPES.FILTER_CONTROL:
                    return 12;
                default:
                    return 6;
            }
        };
        
        // Get default widget height based on type
        const getDefaultWidgetHeight = (type) => {
            switch (type) {
                case WIDGET_TYPES.METRIC_CARD:
                    return 2;
                case WIDGET_TYPES.TEXT_BLOCK:
                    return 3;
                case WIDGET_TYPES.FILTER_CONTROL:
                    return 1;
                default:
                    return 4;
            }
        };
        
        // Get default widget configuration based on type
        const getDefaultWidgetConfig = (type) => {
            const dataSourceId = activeDashboard ? activeDashboard.dataSourceId : null;
            
            switch (type) {
                case WIDGET_TYPES.LINE_CHART:
                    return {
                        dataSourceId,
                        xAxis: '',
                        yAxis: '',
                        groupBy: '',
                        aggregation: 'sum',
                    };
                case WIDGET_TYPES.BAR_CHART:
                    return {
                        dataSourceId,
                        xAxis: '',
                        yAxis: '',
                        aggregation: 'sum',
                    };
                case WIDGET_TYPES.PIE_CHART:
                    return {
                        dataSourceId,
                        dimension: '',
                        metric: '',
                        aggregation: 'sum',
                    };
                case WIDGET_TYPES.SCATTER_PLOT:
                    return {
                        dataSourceId,
                        xAxis: '',
                        yAxis: '',
                        size: '',
                        color: '',
                    };
                case WIDGET_TYPES.DATA_TABLE:
                    return {
                        dataSourceId,
                        columns: [],
                        sortBy: '',
                        sortDirection: 'asc',
                        limit: 10,
                    };
                case WIDGET_TYPES.METRIC_CARD:
                    return {
                        dataSourceId,
                        metric: '',
                        aggregation: 'sum',
                        format: '#,##0',
                    };
                case WIDGET_TYPES.TEXT_BLOCK:
                    return {
                        content: 'Enter your text here...',
                    };
                case WIDGET_TYPES.FILTER_CONTROL:
                    return {
                        filters: [],
                    };
                default:
                    return {};
            }
        };
        
        // Show toast message
        const showToast = (message, severity = 'info') => {
            const toast = { id: Date.now(), message, severity };
            setToasts([...toasts, toast]);
            
            // Auto-dismiss after 3 seconds
            setTimeout(() => {
                setToasts(current => current.filter(t => t.id !== toast.id));
            }, 3000);
        };
        
        // Render a chart preview
        const renderChartPreview = (widget) => {
            const { type, config } = widget;
            
            switch (type) {
                case WIDGET_TYPES.LINE_CHART:
                    return (
                        <div className="flex items-center justify-center h-full bg-blue-50 rounded p-2">
                            <svg width="100%" height="100%" viewBox="0 0 100 60" preserveAspectRatio="none">
                                <polyline
                                    points="0,50 10,45 20,48 30,40 40,35 50,38 60,30 70,25 80,20 90,15 100,10"
                                    fill="none"
                                    stroke="#4F46E5"
                                    strokeWidth="2"
                                />
                            </svg>
                        </div>
                    );
                case WIDGET_TYPES.BAR_CHART:
                    return (
                        <div className="flex items-end justify-around h-full bg-green-50 rounded p-2">
                            <div className="w-1/6 bg-green-500 rounded-t" style={{ height: '60%' }}></div>
                            <div className="w-1/6 bg-green-500 rounded-t" style={{ height: '80%' }}></div>
                            <div className="w-1/6 bg-green-500 rounded-t" style={{ height: '40%' }}></div>
                            <div className="w-1/6 bg-green-500 rounded-t" style={{ height: '70%' }}></div>
                            <div className="w-1/6 bg-green-500 rounded-t" style={{ height: '90%' }}></div>
                        </div>
                    );
                case WIDGET_TYPES.PIE_CHART:
                    return (
                        <div className="flex items-center justify-center h-full bg-purple-50 rounded p-2">
                            <svg width="80%" height="80%" viewBox="0 0 100 100">
                                <circle cx="50" cy="50" r="40" fill="white" />
                                <path d="M50,50 L50,10 A40,40 0 0,1 88.2,65 Z" fill="#8B5CF6" />
                                <path d="M50,50 L88.2,65 A40,40 0 0,1 11.8,65 Z" fill="#EC4899" />
                                <path d="M50,50 L11.8,65 A40,40 0 0,1 50,10 Z" fill="#10B981" />
                            </svg>
                        </div>
                    );
                case WIDGET_TYPES.SCATTER_PLOT:
                    return (
                        <div className="flex items-center justify-center h-full bg-yellow-50 rounded p-2">
                            <svg width="100%" height="100%" viewBox="0 0 100 60">
                                <circle cx="20" cy="30" r="3" fill="#F59E0B" />
                                <circle cx="30" cy="15" r="4" fill="#F59E0B" />
                                <circle cx="45" cy="40" r="2" fill="#F59E0B" />
                                <circle cx="60" cy="25" r="5" fill="#F59E0B" />
                                <circle cx="75" cy="35" r="3" fill="#F59E0B" />
                                <circle cx="85" cy="20" r="4" fill="#F59E0B" />
                            </svg>
                        </div>
                    );
                case WIDGET_TYPES.DATA_TABLE:
                    return (
                        <div className="h-full bg-gray-50 rounded p-2 overflow-hidden">
                            <div className="w-full h-8 bg-gray-200 mb-2"></div>
                            <div className="w-full h-6 bg-gray-100 mb-1"></div>
                            <div className="w-full h-6 bg-gray-100 mb-1"></div>
                            <div className="w-full h-6 bg-gray-100 mb-1"></div>
                            <div className="w-full h-6 bg-gray-100 mb-1"></div>
                        </div>
                    );
                case WIDGET_TYPES.METRIC_CARD:
                    return (
                        <div className="flex flex-col items-center justify-center h-full bg-red-50 rounded p-2">
                            <div className="text-red-500 text-2xl font-bold">
                                {config.format === '$#,##0' ? '$12,345' : '12,345'}
                            </div>
                            <div className="text-red-700 text-xs">{config.aggregation} of {config.metric}</div>
                        </div>
                    );
                case WIDGET_TYPES.TEXT_BLOCK:
                    return (
                        <div className="h-full bg-blue-50 rounded p-2 overflow-hidden">
                            <div className="w-3/4 h-4 bg-blue-200 mb-2 rounded"></div>
                            <div className="w-full h-3 bg-blue-100 mb-1 rounded"></div>
                            <div className="w-full h-3 bg-blue-100 mb-1 rounded"></div>
                            <div className="w-5/6 h-3 bg-blue-100 rounded"></div>
                        </div>
                    );
                case WIDGET_TYPES.FILTER_CONTROL:
                    return (
                        <div className="flex items-center justify-around h-full bg-gray-100 rounded p-1">
                            <div className="h-6 w-1/4 bg-white rounded border border-gray-300"></div>
                            <div className="h-6 w-1/4 bg-white rounded border border-gray-300"></div>
                            <div className="h-6 w-1/4 bg-white rounded border border-gray-300"></div>
                        </div>
                    );
                default:
                    return (
                        <div className="flex items-center justify-center h-full bg-gray-100 rounded">
                            <span className="text-gray-500">Widget Preview</span>
                        </div>
                    );
            }
        };
        
        // Render widget configuration form based on type
        const renderWidgetConfigForm = () => {
            if (!currentWidgetConfig) return null;
            
            const { type, config } = currentWidgetConfig;
            const dataSourceOptions = dataSources.map(ds => ({ value: ds.id, label: ds.name }));
            dataSourceOptions.unshift({ value: '', label: 'Select a data source', disabled: true });
            
            // Column options for the selected data source
            const columnOptions = availableColumns.map(col => ({ value: col.name, label: col.label }));
            columnOptions.unshift({ value: '', label: 'Select a column', disabled: true });
            
            // Numeric column options for metrics
            const numericColumnOptions = availableColumns
                .filter(col => col.type === 'number')
                .map(col => ({ value: col.name, label: col.label }));
            numericColumnOptions.unshift({ value: '', label: 'Select a column', disabled: true });
            
            // Date column options
            const dateColumnOptions = availableColumns
                .filter(col => col.type === 'date')
                .map(col => ({ value: col.name, label: col.label }));
            dateColumnOptions.unshift({ value: '', label: 'Select a column', disabled: true });
            
            // Categorical column options
            const categoricalColumnOptions = availableColumns
                .filter(col => col.type === 'string')
                .map(col => ({ value: col.name, label: col.label }));
            categoricalColumnOptions.unshift({ value: '', label: 'Select a column', disabled: true });
            
            // Aggregation options
            const aggregationOptions = [
                { value: '', label: 'Select an aggregation', disabled: true },
                { value: 'sum', label: 'Sum' },
                { value: 'avg', label: 'Average' },
                { value: 'min', label: 'Minimum' },
                { value: 'max', label: 'Maximum' },
                { value: 'count', label: 'Count' },
                { value: 'count_distinct', label: 'Count Distinct' },
            ];
            
            // Format options
            const formatOptions = [
                { value: '', label: 'Select a format', disabled: true },
                { value: '#,##0', label: 'Number (1,234)' },
                { value: '$#,##0', label: 'Currency ($1,234)' },
                { value: '0.0', label: 'Decimal (1.2)' },
                { value: '0.00%', label: 'Percentage (12.34%)' },
            ];
            
            // Sort direction options
            const sortDirectionOptions = [
                { value: 'asc', label: 'Ascending' },
                { value: 'desc', label: 'Descending' },
            ];
            
            switch (type) {
                case WIDGET_TYPES.LINE_CHART:
                    return (
                        <div className="space-y-4">
                            <Input
                                label="Widget Title"
                                value={currentWidgetConfig.title}
                                onChange={(e) => setCurrentWidgetConfig({...currentWidgetConfig, title: e.target.value})}
                                fullWidth
                            />
                            
                            <Select
                                label="Data Source"
                                value={config.dataSourceId || ''}
                                options={dataSourceOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, dataSourceId: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="X-Axis (Time/Category)"
                                value={config.xAxis || ''}
                                options={[...dateColumnOptions, ...categoricalColumnOptions]}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, xAxis: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Y-Axis (Metric)"
                                value={config.yAxis || ''}
                                options={numericColumnOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, yAxis: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Group By (Series)"
                                value={config.groupBy || ''}
                                options={categoricalColumnOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, groupBy: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Aggregation"
                                value={config.aggregation || ''}
                                options={aggregationOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, aggregation: e.target.value}
                                })}
                                fullWidth
                            />
                        </div>
                    );
                case WIDGET_TYPES.BAR_CHART:
                    return (
                        <div className="space-y-4">
                            <Input
                                label="Widget Title"
                                value={currentWidgetConfig.title}
                                onChange={(e) => setCurrentWidgetConfig({...currentWidgetConfig, title: e.target.value})}
                                fullWidth
                            />
                            
                            <Select
                                label="Data Source"
                                value={config.dataSourceId || ''}
                                options={dataSourceOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, dataSourceId: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="X-Axis (Category)"
                                value={config.xAxis || ''}
                                options={categoricalColumnOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, xAxis: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Y-Axis (Metric)"
                                value={config.yAxis || ''}
                                options={numericColumnOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, yAxis: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Aggregation"
                                value={config.aggregation || ''}
                                options={aggregationOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, aggregation: e.target.value}
                                })}
                                fullWidth
                            />
                        </div>
                    );
                case WIDGET_TYPES.PIE_CHART:
                    return (
                        <div className="space-y-4">
                            <Input
                                label="Widget Title"
                                value={currentWidgetConfig.title}
                                onChange={(e) => setCurrentWidgetConfig({...currentWidgetConfig, title: e.target.value})}
                                fullWidth
                            />
                            
                            <Select
                                label="Data Source"
                                value={config.dataSourceId || ''}
                                options={dataSourceOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, dataSourceId: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Dimension (Category)"
                                value={config.dimension || ''}
                                options={categoricalColumnOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, dimension: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Metric (Value)"
                                value={config.metric || ''}
                                options={numericColumnOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, metric: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Aggregation"
                                value={config.aggregation || ''}
                                options={aggregationOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, aggregation: e.target.value}
                                })}
                                fullWidth
                            />
                        </div>
                    );
                case WIDGET_TYPES.METRIC_CARD:
                    return (
                        <div className="space-y-4">
                            <Input
                                label="Widget Title"
                                value={currentWidgetConfig.title}
                                onChange={(e) => setCurrentWidgetConfig({...currentWidgetConfig, title: e.target.value})}
                                fullWidth
                            />
                            
                            <Select
                                label="Data Source"
                                value={config.dataSourceId || ''}
                                options={dataSourceOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, dataSourceId: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Metric"
                                value={config.metric || ''}
                                options={numericColumnOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, metric: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Aggregation"
                                value={config.aggregation || ''}
                                options={aggregationOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, aggregation: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Format"
                                value={config.format || ''}
                                options={formatOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, format: e.target.value}
                                })}
                                fullWidth
                            />
                        </div>
                    );
                case WIDGET_TYPES.DATA_TABLE:
                    return (
                        <div className="space-y-4">
                            <Input
                                label="Widget Title"
                                value={currentWidgetConfig.title}
                                onChange={(e) => setCurrentWidgetConfig({...currentWidgetConfig, title: e.target.value})}
                                fullWidth
                            />
                            
                            <Select
                                label="Data Source"
                                value={config.dataSourceId || ''}
                                options={dataSourceOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, dataSourceId: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Columns to Display
                                </label>
                                <div className="border border-gray-300 rounded p-2 max-h-40 overflow-y-auto">
                                    {availableColumns.map(column => (
                                        <div key={column.name} className="flex items-center mb-1">
                                            <input
                                                type="checkbox"
                                                id={`column-${column.name}`}
                                                checked={(config.columns || []).includes(column.name)}
                                                onChange={(e) => {
                                                    const columns = [...(config.columns || [])];
                                                    if (e.target.checked) {
                                                        columns.push(column.name);
                                                    } else {
                                                        const index = columns.indexOf(column.name);
                                                        if (index !== -1) columns.splice(index, 1);
                                                    }
                                                    setCurrentWidgetConfig({
                                                        ...currentWidgetConfig,
                                                        config: {...config, columns}
                                                    });
                                                }}
                                                className="mr-2"
                                            />
                                            <label htmlFor={`column-${column.name}`} className="text-sm">
                                                {column.label}
                                            </label>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            
                            <Select
                                label="Sort By"
                                value={config.sortBy || ''}
                                options={columnOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, sortBy: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Select
                                label="Sort Direction"
                                value={config.sortDirection || 'asc'}
                                options={sortDirectionOptions}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, sortDirection: e.target.value}
                                })}
                                fullWidth
                            />
                            
                            <Input
                                label="Row Limit"
                                type="number"
                                value={config.limit || 10}
                                onChange={(e) => setCurrentWidgetConfig({
                                    ...currentWidgetConfig,
                                    config: {...config, limit: parseInt(e.target.value)}
                                })}
                                fullWidth
                            />
                        </div>
                    );
                case WIDGET_TYPES.TEXT_BLOCK:
                    return (
                        <div className="space-y-4">
                            <Input
                                label="Widget Title"
                                value={currentWidgetConfig.title}
                                onChange={(e) => setCurrentWidgetConfig({...currentWidgetConfig, title: e.target.value})}
                                fullWidth
                            />
                            
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Content
                                </label>
                                <textarea
                                    value={config.content || ''}
                                    onChange={(e) => setCurrentWidgetConfig({
                                        ...currentWidgetConfig,
                                        config: {...config, content: e.target.value}
                                    })}
                                    className="w-full h-32 rounded border border-gray-300 p-2"
                                ></textarea>
                            </div>
                        </div>
                    );
                case WIDGET_TYPES.FILTER_CONTROL:
                    return (
                        <div className="space-y-4">
                            <Input
                                label="Widget Title"
                                value={currentWidgetConfig.title}
                                onChange={(e) => setCurrentWidgetConfig({...currentWidgetConfig, title: e.target.value})}
                                fullWidth
                            />
                            
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Filters
                                </label>
                                <div className="border border-gray-300 rounded p-2">
                                    {(config.filters || []).map((filter, index) => (
                                        <div key={index} className="flex items-center gap-2 mb-2">
                                            <Select
                                                value={filter.column || ''}
                                                options={columnOptions}
                                                onChange={(e) => {
                                                    const filters = [...(config.filters || [])];
                                                    filters[index] = {...filter, column: e.target.value};
                                                    setCurrentWidgetConfig({
                                                        ...currentWidgetConfig,
                                                        config: {...config, filters}
                                                    });
                                                }}
                                                className="flex-1"
                                            />
                                            
                                            <Select
                                                value={filter.type || 'single_select'}
                                                options={[
                                                    { value: 'single_select', label: 'Single Select' },
                                                    { value: 'multi_select', label: 'Multi Select' },
                                                    { value: 'range', label: 'Range' },
                                                    { value: 'date_range', label: 'Date Range' },
                                                ]}
                                                onChange={(e) => {
                                                    const filters = [...(config.filters || [])];
                                                    filters[index] = {...filter, type: e.target.value};
                                                    setCurrentWidgetConfig({
                                                        ...currentWidgetConfig,
                                                        config: {...config, filters}
                                                    });
                                                }}
                                                className="w-1/3"
                                            />
                                            
                                            <Button
                                                variant="ghost"
                                                size="small"
                                                icon={<i className="fas fa-trash"></i>}
                                                onClick={() => {
                                                    const filters = [...(config.filters || [])];
                                                    filters.splice(index, 1);
                                                    setCurrentWidgetConfig({
                                                        ...currentWidgetConfig,
                                                        config: {...config, filters}
                                                    });
                                                }}
                                            />
                                        </div>
                                    ))}
                                    
                                    <Button
                                        variant="outline"
                                        size="small"
                                        icon={<i className="fas fa-plus"></i>}
                                        onClick={() => {
                                            const filters = [...(config.filters || [])];
                                            filters.push({
                                                column: '',
                                                type: 'single_select',
                                                label: '',
                                            });
                                            setCurrentWidgetConfig({
                                                ...currentWidgetConfig,
                                                config: {...config, filters}
                                            });
                                        }}
                                        fullWidth
                                    >
                                        Add Filter
                                    </Button>
                                </div>
                            </div>
                        </div>
                    );
                default:
                    return (
                        <div className="p-4 bg-gray-100 rounded">
                            <p>No configuration options available for this widget type.</p>
                        </div>
                    );
            }
        };
        
        // Render
        return (
            <div className="h-full flex flex-col">
                {/* Tabs for different sections */}
                <Tabs
                    tabs={['Dashboard', 'Data Sources', 'Settings']}
                    activeTab={activeTab}
                    onChange={setActiveTab}
                    className="mb-4"
                />
                
                {/* Dashboard tab */}
                {activeTab === 0 && (
                    <div className="flex flex-col h-full">
                        {/* Dashboard header */}
                        <div className="flex justify-between mb-4">
                            <div className="flex gap-2">
                                <Select
                                    value={activeDashboard ? activeDashboard.id : ''}
                                    options={[
                                        { value: '', label: 'Select a dashboard', disabled: true },
                                        ...dashboards.map(d => ({ value: d.id, label: d.name }))
                                    ]}
                                    onChange={(e) => {
                                        const selectedDashboard = dashboards.find(d => d.id === e.target.value);
                                        if (selectedDashboard) {
                                            setActiveDashboard(selectedDashboard);
                                            loadDashboardWidgets(selectedDashboard.id);
                                        }
                                    }}
                                    className="w-48"
                                />
                                
                                <Button
                                    variant="outline"
                                    size="small"
                                    icon={<i className="fas fa-plus"></i>}
                                    onClick={createNewDashboard}
                                >
                                    New
                                </Button>
                            </div>
                            
                            <div className="flex gap-2">
                                <Button
                                    variant={editMode ? 'primary' : 'outline'}
                                    size="small"
                                    icon={<i className="fas fa-edit"></i>}
                                    onClick={() => setEditMode(true)}
                                >
                                    Edit
                                </Button>
                                
                                <Button
                                    variant={!editMode ? 'primary' : 'outline'}
                                    size="small"
                                    icon={<i className="fas fa-eye"></i>}
                                    onClick={() => setEditMode(false)}
                                >
                                    View
                                </Button>
                                
                                <Dropdown
                                    trigger={
                                        <Button
                                            variant="outline"
                                            size="small"
                                            icon={<i className="fas fa-ellipsis-v"></i>}
                                        />
                                    }
                                    items={[
                                        {
                                            label: 'Save Dashboard',
                                            icon: <i className="fas fa-save"></i>,
                                            onClick: () => showToast('Dashboard saved successfully', 'success')
                                        },
                                        {
                                            label: 'Export Dashboard',
                                            icon: <i className="fas fa-file-export"></i>,
                                            onClick: () => showToast('Dashboard exported', 'success')
                                        },
                                        {
                                            label: 'Share Dashboard',
                                            icon: <i className="fas fa-share-alt"></i>,
                                            onClick: () => showToast('Share functionality not implemented in demo', 'info')
                                        },
                                        { divider: true },
                                        {
                                            label: 'Delete Dashboard',
                                            icon: <i className="fas fa-trash"></i>,
                                            onClick: () => showToast('Delete functionality not implemented in demo', 'info')
                                        },
                                    ]}
                                />
                            </div>
                        </div>
                        
                        {/* Widget toolbar (only in edit mode) */}
                        {editMode && (
                            <div className="flex gap-2 mb-4 p-2 bg-gray-100 rounded overflow-x-auto">
                                <Button
                                    variant="outline"
                                    size="small"
                                    icon={<i className="fas fa-chart-line"></i>}
                                    onClick={() => addWidget(WIDGET_TYPES.LINE_CHART)}
                                >
                                    Line Chart
                                </Button>
                                
                                <Button
                                    variant="outline"
                                    size="small"
                                    icon={<i className="fas fa-chart-bar"></i>}
                                    onClick={() => addWidget(WIDGET_TYPES.BAR_CHART)}
                                >
                                    Bar Chart
                                </Button>
                                
                                <Button
                                    variant="outline"
                                    size="small"
                                    icon={<i className="fas fa-chart-pie"></i>}
                                    onClick={() => addWidget(WIDGET_TYPES.PIE_CHART)}
                                >
                                    Pie Chart
                                </Button>
                                
                                <Button
                                    variant="outline"
                                    size="small"
                                    icon={<i className="fas fa-braille"></i>}
                                    onClick={() => addWidget(WIDGET_TYPES.SCATTER_PLOT)}
                                >
                                    Scatter Plot
                                </Button>
                                
                                <Button
                                    variant="outline"
                                    size="small"
                                    icon={<i className="fas fa-table"></i>}
                                    onClick={() => addWidget(WIDGET_TYPES.DATA_TABLE)}
                                >
                                    Data Table
                                </Button>
                                
                                <Button
                                    variant="outline"
                                    size="small"
                                    icon={<i className="fas fa-tachometer-alt"></i>}
                                    onClick={() => addWidget(WIDGET_TYPES.METRIC_CARD)}
                                >
                                    Metric
                                </Button>
                                
                                <Button
                                    variant="outline"
                                    size="small"
                                    icon={<i className="fas fa-font"></i>}
                                    onClick={() => addWidget(WIDGET_TYPES.TEXT_BLOCK)}
                                >
                                    Text
                                </Button>
                                
                                <Button
                                    variant="outline"
                                    size="small"
                                    icon={<i className="fas fa-filter"></i>}
                                    onClick={() => addWidget(WIDGET_TYPES.FILTER_CONTROL)}
                                >
                                    Filter
                                </Button>
                            </div>
                        )}
                        
                        {/* Dashboard grid */}
                        <div
                            ref={gridRef}
                            className={`flex-1 border border-gray-200 rounded bg-gray-50 overflow-auto relative ${editMode ? 'cursor-grab' : ''}`}
                            style={{
                                backgroundImage: editMode ? 'linear-gradient(to right, #f0f0f0 1px, transparent 1px), linear-gradient(to bottom, #f0f0f0 1px, transparent 1px)' : 'none',
                                backgroundSize: editMode ? `${cellSize + gridGap}px ${cellHeight + gridGap}px` : 'auto',
                                padding: gridGap
                            }}
                            onDragOver={editMode ? handleGridDragOver : null}
                            onDrop={editMode ? handleGridDrop : null}
                        >
                            {/* No widgets message */}
                            {widgets.length === 0 && (
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500">
                                    <i className="fas fa-chart-line text-4xl mb-2"></i>
                                    <p className="text-xl">No widgets added to this dashboard yet</p>
                                    {editMode && (
                                        <p className="text-sm mt-2">Use the toolbar above to add widgets</p>
                                    )}
                                </div>
                            )}
                            
                            {/* Widgets */}
                            {widgets.map(widget => {
                                const { id, type, title, position } = widget;
                                const { x, y, w, h } = position;
                                
                                // Calculate pixel position based on grid
                                const pixelX = x * (cellSize + gridGap);
                                const pixelY = y * (cellHeight + gridGap);
                                const pixelWidth = w * (cellSize + gridGap) - gridGap;
                                const pixelHeight = h * (cellHeight + gridGap) - gridGap;
                                
                                return (
                                    <div
                                        key={id}
                                        id={`widget-${id}`}
                                        className={`absolute bg-white rounded shadow ${editMode ? 'cursor-grab' : ''}`}
                                        style={{
                                            left: `${pixelX}px`,
                                            top: `${pixelY}px`,
                                            width: `${pixelWidth}px`,
                                            height: `${pixelHeight}px`,
                                        }}
                                        draggable={editMode}
                                        onDragStart={editMode ? (e) => handleWidgetDragStart(e, id) : null}
                                    >
                                        {/* Widget header */}
                                        <div className="flex justify-between items-center p-2 border-b border-gray-200 bg-gray-50 rounded-t">
                                            <div className="font-medium text-gray-700 truncate">{title}</div>
                                            
                                            {editMode && (
                                                <div className="flex">
                                                    <button
                                                        className="text-gray-500 hover:text-gray-700 mr-1"
                                                        onClick={() => editWidget(widget)}
                                                    >
                                                        <i className="fas fa-cog"></i>
                                                    </button>
                                                    
                                                    <button
                                                        className="text-gray-500 hover:text-red-600"
                                                        onClick={() => deleteWidget(id)}
                                                    >
                                                        <i className="fas fa-times"></i>
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                        
                                        {/* Widget content */}
                                        <div className="p-2 h-[calc(100%-2rem)]">
                                            {renderChartPreview(widget)}
                                        </div>
                                        
                                        {/* Resize handle (only in edit mode) */}
                                        {editMode && (
                                            <div
                                                className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize"
                                                style={{
                                                    backgroundImage: 'linear-gradient(135deg, transparent 50%, rgba(0,0,0,0.3) 50%)',
                                                    borderBottomRightRadius: '0.25rem',
                                                }}
                                                onMouseDown={(e) => {
                                                    // Implement resize logic here
                                                    e.stopPropagation();
                                                    // This is a simplified version - in a real implementation, 
                                                    // you would track mouse movement and update widget size
                                                    const newSize = {
                                                        w: Math.max(1, Math.min(12, position.w + 1)),
                                                        h: Math.max(1, position.h + 1),
                                                    };
                                                    handleWidgetResize(id, newSize);
                                                }}
                                            ></div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
                
                {/* Data Sources tab */}
                {activeTab === 1 && (
                    <div className="h-full overflow-auto">
                        <Card title="Available Data Sources">
                            <div className="space-y-4">
                                {dataSources.map(dataSource => (
                                    <Card
                                        key={dataSource.id}
                                        title={dataSource.name}
                                        subtitle={`Type: ${dataSource.type}, Connection: ${dataSource.connection}`}
                                        variant="outlined"
                                        actions={
                                            <>
                                                <Button
                                                    variant="outline"
                                                    size="small"
                                                    icon={<i className="fas fa-table"></i>}
                                                    onClick={() => showToast('Preview functionality not implemented in demo', 'info')}
                                                >
                                                    Preview Data
                                                </Button>
                                                
                                                <Button
                                                    variant="outline"
                                                    size="small"
                                                    icon={<i className="fas fa-cog"></i>}
                                                    onClick={() => showToast('Settings functionality not implemented in demo', 'info')}
                                                >
                                                    Settings
                                                </Button>
                                            </>
                                        }
                                    />
                                ))}
                                
                                <Button
                                    variant="outline"
                                    icon={<i className="fas fa-plus"></i>}
                                    onClick={() => showToast('Add data source functionality not implemented in demo', 'info')}
                                    fullWidth
                                >
                                    Add Data Source
                                </Button>
                            </div>
                        </Card>
                    </div>
                )}
                
                {/* Settings tab */}
                {activeTab === 2 && (
                    <div className="h-full overflow-auto">
                        <Card title="Dashboard Settings">
                            <div className="space-y-4">
                                <Input
                                    label="Dashboard Name"
                                    value={activeDashboard ? activeDashboard.name : ''}
                                    onChange={(e) => {
                                        if (activeDashboard) {
                                            setActiveDashboard({
                                                ...activeDashboard,
                                                name: e.target.value
                                            });
                                        }
                                    }}
                                    fullWidth
                                />
                                
                                <Input
                                    label="Dashboard Description"
                                    value={activeDashboard ? activeDashboard.description : ''}
                                    onChange={(e) => {
                                        if (activeDashboard) {
                                            setActiveDashboard({
                                                ...activeDashboard,
                                                description: e.target.value
                                            });
                                        }
                                    }}
                                    fullWidth
                                />
                                
                                <Select
                                    label="Default Data Source"
                                    value={activeDashboard ? activeDashboard.dataSourceId : ''}
                                    options={[
                                        { value: '', label: 'Select a data source', disabled: true },
                                        ...dataSources.map(ds => ({ value: ds.id, label: ds.name }))
                                    ]}
                                    onChange={(e) => {
                                        if (activeDashboard) {
                                            setActiveDashboard({
                                                ...activeDashboard,
                                                dataSourceId: e.target.value
                                            });
                                        }
                                    }}
                                    fullWidth
                                />
                                
                                <Button
                                    variant="primary"
                                    onClick={() => showToast('Settings saved successfully', 'success')}
                                >
                                    Save Settings
                                </Button>
                            </div>
                        </Card>
                    </div>
                )}
                
                {/* Add Widget Modal */}
                {isAddWidgetModalOpen && (
                    <Modal
                        isOpen={isAddWidgetModalOpen}
                        onClose={() => {
                            setIsAddWidgetModalOpen(false);
                            setCurrentWidgetConfig(null);
                        }}
                        title={`${currentWidgetConfig.id.includes('widget') ? 'Add' : 'Edit'} ${getDefaultWidgetTitle(currentWidgetConfig.type)}`}
                        size="large"
                        footer={
                            <div className="flex justify-end gap-2">
                                <Button
                                    variant="secondary"
                                    onClick={() => {
                                        setIsAddWidgetModalOpen(false);
                                        setCurrentWidgetConfig(null);
                                    }}
                                >
                                    Cancel
                                </Button>
                                
                                <Button
                                    variant="primary"
                                    onClick={saveWidgetConfig}
                                >
                                    Save Widget
                                </Button>
                            </div>
                        }
                    >
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-4">
                                <h3 className="text-lg font-medium">Widget Configuration</h3>
                                {renderWidgetConfigForm()}
                            </div>
                            
                            <div className="space-y-4">
                                <h3 className="text-lg font-medium">Preview</h3>
                                <div className="border border-gray-200 rounded p-4 bg-gray-50 h-64">
                                    <div className="border-b border-gray-200 pb-2 mb-2 font-medium">
                                        {currentWidgetConfig.title}
                                    </div>
                                    <div className="h-[calc(100%-2rem)]">
                                        {renderChartPreview(currentWidgetConfig)}
                                    </div>
                                </div>
                                
                                <Card title="Layout Settings">
                                    <div className="grid grid-cols-2 gap-4">
                                        <Input
                                            label="Width (1-12)"
                                            type="number"
                                            min={1}
                                            max={12}
                                            value={currentWidgetConfig.position.w}
                                            onChange={(e) => {
                                                const value = parseInt(e.target.value);
                                                const w = Math.max(1, Math.min(12, value));
                                                setCurrentWidgetConfig({
                                                    ...currentWidgetConfig,
                                                    position: {...currentWidgetConfig.position, w}
                                                });
                                            }}
                                        />
                                        
                                        <Input
                                            label="Height (1-8)"
                                            type="number"
                                            min={1}
                                            max={8}
                                            value={currentWidgetConfig.position.h}
                                            onChange={(e) => {
                                                const value = parseInt(e.target.value);
                                                const h = Math.max(1, Math.min(8, value));
                                                setCurrentWidgetConfig({
                                                    ...currentWidgetConfig,
                                                    position: {...currentWidgetConfig.position, h}
                                                });
                                            }}
                                        />
                                    </div>
                                </Card>
                            </div>
                        </div>
                    </Modal>
                )}
                
                {/* Toast messages */}
                {toasts.map((toast) => (
                    <Toast
                        key={toast.id}
                        message={toast.message}
                        severity={toast.severity}
                        onClose={() => setToasts(toasts.filter(t => t.id !== toast.id))}
                    />
                ))}
            </div>
        );
    };
    
    return {
        render: (containerId, props = {}) => {
            const container = document.getElementById(containerId);
            if (!container) {
                console.error(`Container with ID "${containerId}" not found`);
                return;
            }
            
            // Create a React DOM root if not exists
            if (!container._reactRoot) {
                container._reactRoot = ReactDOM.createRoot(container);
            }
            
            // Render the dashboard builder
            container._reactRoot.render(
                <DashboardBuilderComponent {...props} />
            );
        }
    };
})();

// Export to window
window.DashboardBuilder = DashboardBuilder;
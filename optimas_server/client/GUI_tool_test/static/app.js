/**
 * MCP Tool Dashboard - Logic Overhaul
 */

// Global state
const state = {
    tools: [],
    selectedTool: null,
    currentTab: 'list',
    filteredTools: [],
    viewMode: 'list', // list | grid
    theme: localStorage.getItem('theme') || 'light'
};

// API functions
const api = {
    async listTools() {
        const response = await fetch('/api/tools');
        if (!response.ok) throw new Error('Failed to fetch tools');
        const data = await response.json();
        return data.tools;
    },

    async getToolSchema(toolName) {
        const response = await fetch(`/api/tools/${encodeURIComponent(toolName)}/schema`);
        if (!response.ok) throw new Error('Failed to fetch tool schema');
        return await response.json();
    },

    async callTool(toolName, arguments_) {
        const response = await fetch(`/api/tools/${encodeURIComponent(toolName)}/call`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ arguments: arguments_ })
        });
        return await response.json();
    }
};

// UI utilities
const ui = {
    showLoading() {
        document.getElementById('loadingOverlay').classList.add('active');
    },

    hideLoading() {
        document.getElementById('loadingOverlay').classList.remove('active');
    },

    showToast(message, type = 'success') {
        const toast = document.getElementById('toast');
        const toastMsg = document.getElementById('toastMessage');
        const toastIcon = document.getElementById('toastIcon');
        
        toastMsg.textContent = message;
        toast.className = `toast show toast-${type}`;
        
        if (type === 'success') toastIcon.className = 'ri-checkbox-circle-line';
        else if (type === 'error') toastIcon.className = 'ri-error-warning-line';
        else toastIcon.className = 'ri-information-line';

        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    },

    formatJSON(obj) {
        return JSON.stringify(obj, null, 2);
    },

    formatMaybeJSON(value) {
        // If the value is a string that looks like JSON, try parsing before formatting
        if (typeof value === 'string') {
            const trimmed = value.trim();
            if ((trimmed.startsWith('{') && trimmed.endsWith('}')) ||
                (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
                try {
                    const parsed = JSON.parse(trimmed);
                    return this.formatJSON(parsed);
                } catch (e) {
                    // Parse failed; return as-is to avoid silently swallowing errors
                    return value;
                }
            }
            return value;
        }
        // Objects/arrays: format normally
        return this.formatJSON(value);
    },

    highlightJSON(jsonString) {
        return jsonString
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/(".*?")\s*:/g, '<span class="json-key">$1</span>:')
            .replace(/:\s*(".*?")/g, (match, p1) => {
                // Convert escaped \n inside strings to real newlines
                const unescaped = p1.replace(/\\n/g, '\n');
                return `: <span class="json-string">${unescaped}</span>`;
            })
            .replace(/:\s*(\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
            .replace(/:\s*(true|false)/g, ': <span class="json-boolean">$1</span>')
            .replace(/:\s*(null)/g, ': <span class="json-null">$1</span>');
    },

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showToast('Copied to clipboard!', 'success');
        }).catch(err => {
            this.showToast('Failed to copy', 'error');
        });
    }
};

// Tool list management
const toolList = {
    async load() {
        try {
            ui.showLoading();
            state.tools = await api.listTools();
            state.filteredTools = state.tools;
            this.render();
            this.renderSidebar();
            ui.hideLoading();
        } catch (error) {
            ui.hideLoading();
            ui.showToast(`Failed to load tools: ${error.message}`, 'error');
        }
    },

    render() {
        const grid = document.getElementById('toolsGrid');
        const count = document.getElementById('toolCount');
        
        count.textContent = `${state.filteredTools.length} Tool${state.filteredTools.length !== 1 ? 's' : ''}`;
        
        if (state.filteredTools.length === 0) {
            grid.classList.remove('list-view');
            grid.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1;">
                    <i class="ri-search-eye-line" style="font-size: 3rem; display: block; margin-bottom: 16px;"></i>
                    <p>No tools found matching your search</p>
                </div>`;
            return;
        }

        if (state.viewMode === 'list') {
            grid.classList.add('list-view');
            grid.innerHTML = state.filteredTools.map(tool => `
                <div class="tool-row" onclick="toolList.selectTool('${tool.name}')">
                    <div class="tool-row-main">
                        <div class="tool-row-icon">
                            <i class="ri-tools-line"></i>
                        </div>
                        <div class="tool-row-texts">
                            <div class="tool-row-name">${tool.name}</div>
                            <div class="tool-row-desc">${tool.description || 'No description provided for this tool.'}</div>
                        </div>
                    </div>
                    <div class="tool-row-meta">
                        <span class="tool-row-pill">Click to test</span>
                    </div>
                </div>
            `).join('');
        } else {
            grid.classList.remove('list-view');
            grid.innerHTML = state.filteredTools.map(tool => `
                <div class="tool-card" onclick="toolList.selectTool('${tool.name}')">
                    <div class="tool-card-header">
                        <div class="tool-card-icon">
                            <i class="ri-tools-line"></i>
                        </div>
                        <div class="tool-card-name">${tool.name}</div>
                        <div class="tool-card-desc">${tool.description || 'No description provided for this tool.'}</div>
                    </div>
                    <div class="tool-card-footer">
                        <button class="action-btn">
                            <i class="ri-arrow-right-line"></i>
                        </button>
                    </div>
                </div>
            `).join('');
        }
    },

    renderSidebar() {
        const list = document.getElementById('toolList');
        
        if (state.filteredTools.length === 0) {
            list.innerHTML = '<div class="empty-state" style="padding: 20px;"><p>No tools found</p></div>';
            return;
        }

        list.innerHTML = state.filteredTools.map(tool => `
            <div class="tool-item ${state.selectedTool?.name === tool.name ? 'active' : ''}" 
                 onclick="toolList.selectTool('${tool.name}')">
                <div class="tool-item-name">${tool.name}</div>
                <div class="tool-item-desc">${tool.description || 'No description'}</div>
            </div>
        `).join('');
    },

    async selectTool(toolName) {
        const tool = state.tools.find(t => t.name === toolName);
        if (!tool) return;

        state.selectedTool = tool;
        this.renderSidebar();
        
        // Update breadcrumb and UI
        document.getElementById('breadcrumb-tool').textContent = tool.name;
        document.getElementById('selectedToolNameTester').textContent = tool.name;

        // Auto-switch to tester if on list tab
        if (state.currentTab === 'list') {
            tabs.switchTab('tester');
        } else {
            // Refresh current tab data
            if (state.currentTab === 'schema') await schemaViewer.load(toolName);
            if (state.currentTab === 'tester') await toolTester.load(toolName);
        }
    },

    filter(searchTerm) {
        const term = searchTerm.toLowerCase();
        state.filteredTools = state.tools.filter(tool =>
            tool.name.toLowerCase().includes(term) ||
            (tool.description && tool.description.toLowerCase().includes(term))
        );
        this.render();
        this.renderSidebar();
    }
};

// Schema viewer
const schemaViewer = {
    currentSchema: null,
    async load(toolName) {
        try {
            ui.showLoading();
            this.currentSchema = await api.getToolSchema(toolName);
            this.render(this.currentSchema);
            ui.hideLoading();
        } catch (error) {
            ui.hideLoading();
            ui.showToast(`Failed to load schema: ${error.message}`, 'error');
            document.getElementById('schemaViewer').innerHTML = 
                `<div class="empty-state"><p>Failed to load schema</p></div>`;
        }
    },

    render(schema) {
        const viewer = document.getElementById('schemaViewer');
        const jsonString = ui.formatMaybeJSON(schema);
        viewer.innerHTML = `<div class="json-view">${ui.highlightJSON(jsonString)}</div>`;
    }
};

// Tool tester
const toolTester = {
    currentSchema: null,
    lastResult: null,

    async load(toolName) {
        try {
            ui.showLoading();
            this.currentSchema = await api.getToolSchema(toolName);
            this.renderForm();
            ui.hideLoading();
        } catch (error) {
            ui.hideLoading();
            ui.showToast(`Failed to load tool schema: ${error.message}`, 'error');
            document.getElementById('formContainer').innerHTML = 
                `<div class="empty-state"><p>Failed to load tool schema</p></div>`;
        }
    },

    renderForm() {
        if (!this.currentSchema || !this.currentSchema.inputSchema) {
            document.getElementById('formContainer').innerHTML = 
                `<div class="empty-state"><p>No input schema available</p></div>`;
            return;
        }

        const schema = this.currentSchema.inputSchema;
        const properties = schema.properties || {};
        const required = schema.required || [];

        let formHTML = '<form id="toolForm">';
        
        for (const [key, prop] of Object.entries(properties)) {
            const isRequired = required.includes(key);
            formHTML += this.renderField(key, prop, isRequired);
        }

        formHTML += `
            <div style="margin-top: 32px;">
                <button type="submit" class="action-btn" style="width: 100%; padding: 12px; font-size: 1rem;">
                    <i class="ri-play-fill"></i> Execute Tool
                </button>
            </div>
        </form>`;

        document.getElementById('formContainer').innerHTML = formHTML;

        document.getElementById('toolForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.executeTool();
        });
    },

    renderField(name, prop, isRequired) {
        const type = prop.type || 'string';
        const description = prop.description || '';
        const defaultValue = prop.default !== undefined ? prop.default : null;
        
        let fieldHTML = `
            <div class="form-group">
                <label class="form-label">
                    ${name} ${isRequired ? '<span class="required">*</span>' : ''}
                </label>
        `;

        if (description) {
            fieldHTML += `<div class="form-help">${description}</div>`;
        }

        switch (type) {
            case 'boolean':
                const defaultBoolValue = defaultValue === true ? 'true' : (defaultValue === false ? 'false' : 'false');
                fieldHTML += `
                    <select class="input-field" id="field_${name}" name="${name}" ${!isRequired ? '' : 'required'}>
                        <option value="true" ${defaultBoolValue === 'true' ? 'selected' : ''}>true</option>
                        <option value="false" ${defaultBoolValue === 'false' ? 'selected' : ''}>false</option>
                    </select>
                `;
                break;
            case 'integer':
            case 'number':
                fieldHTML += `
                    <input type="number" class="input-field" id="field_${name}" name="${name}" 
                           value="${defaultValue !== null ? defaultValue : ''}"
                           ${prop.minimum !== undefined ? `min="${prop.minimum}"` : ''}
                           ${prop.maximum !== undefined ? `max="${prop.maximum}"` : ''}
                           ${!isRequired ? '' : 'required'}>
                `;
                break;
            case 'array':
            case 'object':
                fieldHTML += `
                    <textarea class="input-field textarea-field" id="field_${name}" name="${name}" 
                              placeholder='Enter JSON ${type}...'
                              ${!isRequired ? '' : 'required'}></textarea>
                `;
                break;
            default: {
                const lowerName = name.toLowerCase();
                // Fields that should use textarea for multi-line input
                const multiLineFields = ['old_lines', 'new_lines', 'pre_context', 'post_context'];
                const isLongText = lowerName === 'content' || 
                                 lowerName.endsWith('_content') || 
                                 multiLineFields.includes(lowerName) ||
                                 (description && description.toLowerCase().includes('content'));

                if (isLongText) {
                    fieldHTML += `
                        <textarea class="input-field textarea-field" id="field_${name}" name="${name}" 
                                  placeholder="Enter text..."
                                  ${!isRequired ? '' : 'required'}>${defaultValue !== null ? defaultValue : ''}</textarea>
                    `;
                } else {
                    fieldHTML += `
                        <input type="text" class="input-field" id="field_${name}" name="${name}" 
                               value="${defaultValue !== null ? defaultValue : ''}"
                               ${!isRequired ? '' : 'required'}>
                    `;
                }
                break;
            }
        }

        fieldHTML += '</div>';
        return fieldHTML;
    },

    getFormData() {
        const form = document.getElementById('toolForm');
        const formData = new FormData(form);
        const data = {};
        const schema = this.currentSchema.inputSchema;
        const properties = schema.properties || {};
        const required = schema.required || [];

        // First, process all fields from FormData (non-boolean fields)
        for (const [key, value] of formData.entries()) {
            const prop = properties[key];
            const type = prop?.type || 'string';

            // Skip boolean fields here, they will be handled separately
            if (type === 'boolean') {
                continue;
            }

            if (type === 'integer' || type === 'number') {
                data[key] = value === '' ? null : (type === 'integer' ? parseInt(value) : parseFloat(value));
            } else if (type === 'array' || type === 'object') {
                try {
                    data[key] = value === '' ? null : JSON.parse(value);
                } catch (e) {
                    throw new Error(`Invalid JSON in "${key}": ${e.message}`);
                }
            } else {
                data[key] = value === '' ? null : value;
            }
        }

        // Then, handle all boolean fields explicitly
        for (const [key, prop] of Object.entries(properties)) {
            if (prop.type === 'boolean') {
                const field = document.getElementById(`field_${key}`);
                if (field) {
                    // Get value from select dropdown: "true" or "false"
                    data[key] = field.value === 'true';
                } else if (required.includes(key)) {
                    // Required boolean field but field not found, default to false
                    data[key] = false;
                }
            }
        }

        // Remove optional fields with null values
        for (const key in data) {
            if (!required.includes(key) && data[key] === null) {
                delete data[key];
            }
        }

        return data;
    },

    async executeTool() {
        try {
            const args = this.getFormData();
            ui.showLoading();
            const result = await api.callTool(state.selectedTool.name, args);
            this.lastResult = result;
            this.renderResult(result);
            ui.hideLoading();
            
            if (result.success) ui.showToast('Executed successfully!', 'success');
            else ui.showToast('Execution failed', 'error');
        } catch (error) {
            ui.hideLoading();
            ui.showToast(error.message, 'error');
            document.getElementById('resultContainer').innerHTML = `
                <div class="result-box result-error">
                    <strong>Error:</strong> ${error.message}
                </div>`;
        }
    },

    renderResult(result) {
        const container = document.getElementById('resultContainer');
        if (result.success) {
            const dataString = ui.formatMaybeJSON(result.data);
            container.innerHTML = `
                <div class="result-box result-success" style="margin-bottom: 16px;">
                    <i class="ri-checkbox-circle-fill"></i> Tool execution successful
                </div>
                <div class="json-view">${ui.highlightJSON(dataString)}</div>`;
        } else {
            const errorPayload = result.error ?? 'Unknown error';
            const errorString = ui.formatMaybeJSON(errorPayload);
            container.innerHTML = `
                <div class="result-box result-error">
                    <i class="ri-close-circle-fill"></i>
                    <strong>Error:</strong>
                </div>
                <div class="json-view" style="margin-top: 12px;">${ui.highlightJSON(errorString)}</div>`;
        }
    }
};

// Theme Management
const theme = {
    init() {
        document.documentElement.setAttribute('data-theme', state.theme);
        this.updateIcon();
        
        document.getElementById('themeToggle').addEventListener('click', () => {
            state.theme = state.theme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', state.theme);
            localStorage.setItem('theme', state.theme);
            this.updateIcon();
        });
    },
    updateIcon() {
        const icon = document.getElementById('themeIcon');
        icon.className = state.theme === 'light' ? 'ri-moon-line' : 'ri-sun-line';
    }
};

// Tab management
const tabs = {
    switchTab(tabName) {
        state.currentTab = tabName;
        
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `tab-${tabName}`);
        });

        // Update breadcrumb
        const tabLabels = { 'list': 'Tools Grid', 'schema': 'Schema', 'tester': 'Tester' };
        document.getElementById('breadcrumb-tab').textContent = tabLabels[tabName];

        if (state.selectedTool) {
            if (tabName === 'schema') schemaViewer.load(state.selectedTool.name);
            else if (tabName === 'tester') toolTester.load(state.selectedTool.name);
        }
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    theme.init();

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => tabs.switchTab(btn.dataset.tab));
    });

    document.getElementById('toolSearch').addEventListener('input', (e) => {
        toolList.filter(e.target.value);
    });

    document.getElementById('copySchema').addEventListener('click', () => {
        if (schemaViewer.currentSchema) {
            ui.copyToClipboard(ui.formatJSON(schemaViewer.currentSchema));
        }
    });

    document.getElementById('copyResult').addEventListener('click', () => {
        if (toolTester.lastResult) {
            const data = toolTester.lastResult.success ? toolTester.lastResult.data : toolTester.lastResult.error;
            ui.copyToClipboard(typeof data === 'string' ? data : ui.formatJSON(data));
        }
    });

    // View toggle: list / grid
    const viewToggle = document.getElementById('viewToggle');
    if (viewToggle) {
        viewToggle.querySelectorAll('.view-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const view = btn.dataset.view;
                if (!view || view === state.viewMode) return;
                state.viewMode = view;
                viewToggle.querySelectorAll('.view-toggle-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                toolList.render();
            });
        });
    }

    toolList.load();
});

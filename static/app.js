// State
let currentDomain = null;
let selectedFiles = [];
let conversationHistory = {};  // Store conversation history per domain
let currentConversationId = null;  // Track active conversation
let conversations = {};  // All saved conversations
let conversationsDisplayLimit = 20;  // Number of conversations to show initially

// Theme handling
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const themeText = document.getElementById('themeText');

// Load saved theme or default to dark
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);
updateThemeButton(savedTheme);

function updateThemeButton(theme) {
    if (theme === 'light') {
        themeIcon.textContent = '🌙';
        themeText.textContent = 'Dark';
    } else {
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light';
    }
}

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeButton(newTheme);
});

// DOM Elements
const domainList = document.getElementById('domainList');
const newDomainBtn = document.getElementById('newDomainBtn');
const newDomainModal = document.getElementById('newDomainModal');
const domainNameInput = document.getElementById('domainNameInput');
const createDomainBtn = document.getElementById('createDomainBtn');
const cancelDomainBtn = document.getElementById('cancelDomainBtn');

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');

const domainInfo = document.getElementById('domainInfo');
const infoFiles = document.getElementById('infoFiles');
const infoStatus = document.getElementById('infoStatus');
const infoUpdated = document.getElementById('infoUpdated');
const reindexBtn = document.getElementById('reindexBtn');
const deleteBtn = document.getElementById('deleteBtn');

const welcome = document.getElementById('welcome');
const chatContainer = document.getElementById('chatContainer');
const messages = document.getElementById('messages');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const askBtnText = document.getElementById('askBtnText');
const askBtnLoader = document.getElementById('askBtnLoader');

const toast = document.getElementById('toast');

// Initialize
async function init() {
    loadConversations();
    await loadDomains();
    setupEventListeners();
}

// API Functions
async function loadDomains() {
    try {
        const response = await fetch('/api/domains');
        const domains = await response.json();
        
        if (domains.length === 0) {
            domainList.innerHTML = '<div class="loading">No domains yet</div>';
            return;
        }
        
        domainList.innerHTML = '';
        domains.forEach(domain => {
            const item = createDomainItem(domain);
            domainList.appendChild(item);
        });
    } catch (error) {
        console.error('Failed to load domains:', error);
        showToast('Failed to load domains', 'error');
    }
}

async function createDomain(name) {
    try {
        const response = await fetch(`/api/domains/${name}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to create domain');
        }
        
        showToast(`Domain '${name}' created!`, 'success');
        await loadDomains();
        selectDomain(name);
        return true;
    } catch (error) {
        console.error('Failed to create domain:', error);
        showToast(error.message, 'error');
        return false;
    }
}

async function uploadFiles(domain, files) {
    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }
    
    try {
        const response = await fetch(`/api/upload/${domain}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const result = await response.json();
        showToast(`Uploaded ${result.files.length} file(s). Processing started...`, 'success');
        
        // Reload domains after a delay to show updated status
        setTimeout(() => loadDomains(), 2000);
        
        return true;
    } catch (error) {
        console.error('Failed to upload files:', error);
        showToast('Upload failed', 'error');
        return false;
    }
}

async function queryDomain(domain, question) {
    try {
        // Get conversation history for this domain
        const history = conversationHistory[domain] || [];
        
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                domain, 
                question,
                conversation_history: history
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Query failed:', error);
        throw error;
    }
}

async function reindexDomain(domain) {
    try {
        const response = await fetch(`/api/process/${domain}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Reindex failed');
        }
        
        showToast('Reindexing started...', 'success');
        setTimeout(() => loadDomains(), 2000);
    } catch (error) {
        console.error('Reindex failed:', error);
        showToast('Reindex failed', 'error');
    }
}

async function deleteDomain(domain) {
    if (!confirm(`Are you sure you want to delete '${domain}' and all its data?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/domains/${domain}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Delete failed');
        }
        
        showToast(`Domain '${domain}' deleted`, 'success');
        currentDomain = null;
        await loadDomains();
        showWelcome();
    } catch (error) {
        console.error('Delete failed:', error);
        showToast('Delete failed', 'error');
    }
}

// UI Functions
function createDomainItem(domain) {
    const item = document.createElement('div');
    item.className = 'domain-item';
    if (domain.name === currentDomain) {
        item.classList.add('active');
    }
    
    const statusClass = domain.indexed ? 'indexed' : 'not-indexed';
    const statusText = domain.indexed ? 'Indexed' : 'Not Indexed';
    
    item.innerHTML = `
        <span class="domain-name">${domain.name}</span>
        <div class="domain-meta">
            ${domain.file_count} files · 
            <span class="status-badge ${statusClass}">${statusText}</span>
        </div>
    `;
    
    item.addEventListener('click', () => selectDomain(domain.name));
    
    return item;
}

function selectDomain(name) {
    currentDomain = name;
    
    // Initialize conversation history for this domain if not exists
    if (!conversationHistory[name]) {
        conversationHistory[name] = [];
    }
    
    // Update UI
    document.querySelectorAll('.domain-item').forEach(item => {
        item.classList.remove('active');
        if (item.querySelector('.domain-name').textContent === name) {
            item.classList.add('active');
        }
    });
    
    // Update domain info
    updateDomainInfo();
    
    // Show chat container
    showChat();
    
    // Enable upload
    uploadBtn.disabled = false;
    askBtn.disabled = false;
}

async function updateDomainInfo() {
    try {
        const response = await fetch('/api/domains');
        const domains = await response.json();
        const domain = domains.find(d => d.name === currentDomain);
        
        if (domain) {
            infoFiles.textContent = domain.file_count;
            
            if (domain.indexed) {
                infoStatus.textContent = 'Indexed';
                infoStatus.className = 'status-badge indexed';
            } else {
                infoStatus.textContent = 'Not Indexed';
                infoStatus.className = 'status-badge not-indexed';
            }
            
            if (domain.last_updated) {
                const date = new Date(domain.last_updated);
                infoUpdated.textContent = date.toLocaleString();
            } else {
                infoUpdated.textContent = 'Never';
            }
            
            domainInfo.style.display = 'block';
        }
    } catch (error) {
        console.error('Failed to update domain info:', error);
    }
}

function showWelcome() {
    welcome.style.display = 'flex';
    chatContainer.style.display = 'none';
    domainInfo.style.display = 'none';
}

function showChat() {
    welcome.style.display = 'none';
    chatContainer.style.display = 'flex';
    messages.innerHTML = '';
}

function addMessage(question, answer, sources) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    
    let html = `
        <div class="message-question">
            <strong>Q:</strong>${escapeHtml(question)}
        </div>
        <div class="message-answer">
            <strong>A:</strong>
            <div class="answer-text">${escapeHtml(answer)}</div>
    `;
    
    if (sources && sources.length > 0) {
        html += '<div class="sources"><div class="sources-header">Sources:</div>';
        
        // Group sources by document
        const grouped = {};
        sources.forEach(source => {
            if (!grouped[source.source]) {
                grouped[source.source] = [];
            }
            grouped[source.source].push(source);
        });
        
        // Display grouped sources
        Object.keys(grouped).forEach(filename => {
            const docSources = grouped[filename];
            const sourceId = `source-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            
            html += `
                <div class="source-group">
                    <div class="source-group-header" onclick="toggleSourceGroup('${sourceId}')">
                        <span class="source-toggle">▶</span>
                        <a href="#" 
                           class="source-file-link"
                           onclick="event.stopPropagation(); openLocalFile('${currentDomain}', '${escapeHtml(filename).replace(/'/g, "\\'")}'); return false;">
                            ${escapeHtml(filename)}
                        </a>
                        <span class="source-count">${docSources.length} chunk${docSources.length > 1 ? 's' : ''}</span>
                    </div>
                    <div class="source-group-items" id="${sourceId}" style="display: none;">
            `;
            
            docSources.forEach(source => {
                const location = [
                    source.page ? `p.${source.page}` : null,
                    source.section ? `s.${source.section}` : null
                ].filter(Boolean).join(', ');
                
                const scorePercent = Math.round(source.score * 100);
                const scoreClass = scorePercent >= 80 ? 'high' : scorePercent >= 60 ? 'medium' : 'low';
                const chunkId = `chunk-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                
                html += `
                    <div class="source-item">
                        <div class="source-item-header" onclick="toggleChunk('${chunkId}')">
                            <span class="source-ref">[${source.rank}]</span>
                            ${location ? `<span class="source-location">${location}</span>` : ''}
                            <span class="source-score ${scoreClass}">${scorePercent}%</span>
                            <span class="chunk-toggle">▼</span>
                        </div>
                        <div class="source-text" id="${chunkId}" style="display: none;">${escapeHtml(source.text)}</div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
    }
    
    html += '</div>';
    messageDiv.innerHTML = html;
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function showToast(message, type = 'success') {
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Polling for processing status
function startPolling() {
    // Poll every 5 seconds if there are processing domains
    pollInterval = setInterval(async () => {
        if (processingDomains.size > 0) {
            await loadDomains(true);
        }
    }, 5000);
}

// Open local file in default system application
async function openLocalFile(domain, filename) {
    try {
        // Try to use the API endpoint to trigger opening
        const response = await fetch(`/api/open-file/${encodeURIComponent(domain)}/${encodeURIComponent(filename)}`);
        const data = await response.json();
        
        if (!response.ok) {
            alert(`Could not open file: ${data.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error opening file:', error);
        alert('Error opening file. Check console for details.');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Toggle functions for expandable sources
function toggleSourceGroup(id) {
    const element = document.getElementById(id);
    const toggle = element.previousElementSibling.querySelector('.source-toggle');
    if (element.style.display === 'none') {
        element.style.display = 'block';
        toggle.textContent = '▼';  // Down arrow
    } else {
        element.style.display = 'none';
        toggle.textContent = '▶';  // Right arrow
    }
}

function toggleChunk(id) {
    const element = document.getElementById(id);
    const toggle = element.previousElementSibling.querySelector('.chunk-toggle');
    if (element.style.display === 'none') {
        element.style.display = 'block';
        toggle.textContent = '▲';  // Up arrow
    } else {
        element.style.display = 'none';
        toggle.textContent = '▼';  // Down arrow
    }
}

// Event Listeners
function setupEventListeners() {
    // New Domain Modal
    newDomainBtn.addEventListener('click', () => {
        newDomainModal.classList.add('active');
        domainNameInput.value = '';
        domainNameInput.focus();
    });
    
    // New Conversation Button
    const newConversationBtn = document.getElementById('newConversationBtn');
    if (newConversationBtn) {
        newConversationBtn.addEventListener('click', () => {
            createNewConversation();
        });
    }
    
    cancelDomainBtn.addEventListener('click', () => {
        newDomainModal.classList.remove('active');
    });
    
    createDomainBtn.addEventListener('click', async () => {
        const name = domainNameInput.value.trim();
        if (!name) {
            showToast('Please enter a domain name', 'error');
            return;
        }
        
        if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
            showToast('Domain name can only contain letters, numbers, hyphens, and underscores', 'error');
            return;
        }
        
        const success = await createDomain(name);
        if (success) {
            newDomainModal.classList.remove('active');
        }
    });
    
    domainNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            createDomainBtn.click();
        }
    });
    
    // File Upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files);
        handleFileSelection(files);
    });
    
    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        handleFileSelection(files);
    });
    
    uploadBtn.addEventListener('click', async () => {
        if (!currentDomain || selectedFiles.length === 0) {
            return;
        }
        
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';
        
        await uploadFiles(currentDomain, selectedFiles);
        
        selectedFiles = [];
        fileInput.value = '';
        uploadBtn.textContent = 'Upload Files';
        uploadBtn.disabled = false;
        
        updateDomainInfo();
    });
    
    // Chat
    askBtn.addEventListener('click', handleAsk);
    
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleAsk();
        }
    });
    
    // Domain Actions
    reindexBtn.addEventListener('click', () => {
        if (currentDomain) {
            reindexDomain(currentDomain);
        }
    });
    
    deleteBtn.addEventListener('click', () => {
        if (currentDomain) {
            deleteDomain(currentDomain);
        }
    });
}

function handleFileSelection(files) {
    selectedFiles = files;
    uploadBtn.disabled = false;
    uploadZone.querySelector('p').textContent = `${files.length} file(s) selected`;
}

async function handleAsk() {
    const question = questionInput.value.trim();
    if (!question || !currentDomain) {
        return;
    }
    
    // Show question immediately
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    messageDiv.innerHTML = `
        <div class="message-question">
            <strong>Q:</strong>${escapeHtml(question)}
        </div>
        <div class="message-answer">
            <strong>A:</strong>
            <div class="answer-text"><span class="typing-indicator"><span></span><span></span><span></span></span></div>
        </div>
    `;
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
    
    // Clear input and disable
    questionInput.value = '';
    askBtn.disabled = true;
    askBtnText.style.display = 'none';
    askBtnLoader.style.display = 'block';
    questionInput.disabled = true;
    
    try {
        const result = await queryDomain(currentDomain, question);
        
        // Update the answer in the existing message
        let answerHtml = `<strong>A:</strong><div class="answer-text">${escapeHtml(result.answer)}</div>`;
        
        if (result.sources && result.sources.length > 0) {
            answerHtml += '<div class="sources"><div class="sources-header">Sources:</div>';
            
            // Group sources by document
            const grouped = {};
            result.sources.forEach(source => {
                if (!grouped[source.source]) {
                    grouped[source.source] = [];
                }
                grouped[source.source].push(source);
            });
            
            // Display grouped sources
            Object.keys(grouped).forEach(filename => {
                const docSources = grouped[filename];
                const sourceId = `source-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                
                answerHtml += `
                    <div class="source-group">
                        <div class="source-group-header" onclick="toggleSourceGroup('${sourceId}')">
                            <span class="source-toggle">▶</span>
                            <span class="source-file">${escapeHtml(filename)}</span>
                            <span class="source-count">${docSources.length} chunk${docSources.length > 1 ? 's' : ''}</span>
                        </div>
                        <div class="source-group-items" id="${sourceId}" style="display: none;">
                `;
                
                docSources.forEach(source => {
                    const location = [
                        source.page ? `p.${source.page}` : null,
                        source.section ? `s.${source.section}` : null
                    ].filter(Boolean).join(', ');
                    
                    const scorePercent = Math.round(source.score * 100);
                    const scoreClass = scorePercent >= 80 ? 'high' : scorePercent >= 60 ? 'medium' : 'low';
                    const chunkId = `chunk-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                    
                    answerHtml += `
                        <div class="source-item">
                            <div class="source-item-header" onclick="toggleChunk('${chunkId}')">
                                <span class="source-ref">[${source.rank}]</span>
                                ${location ? `<span class="source-location">${location}</span>` : ''}
                                <span class="source-score ${scoreClass}">${scorePercent}%</span>
                                <span class="chunk-toggle">▼</span>
                            </div>
                            <div class="source-text" id="${chunkId}" style="display: none;">${escapeHtml(source.text)}</div>
                        </div>
                    `;
                });
                
                answerHtml += `
                        </div>
                    </div>
                `;
            });
            
            answerHtml += '</div>';
        }
        
        messageDiv.querySelector('.message-answer').innerHTML = answerHtml;
        messages.scrollTop = messages.scrollHeight;
        
        // Store in conversation history
        if (!conversationHistory[currentDomain]) {
            conversationHistory[currentDomain] = [];
        }
        conversationHistory[currentDomain].push({
            question: question,
            answer: result.answer,
            sources: result.sources,
            timestamp: Date.now()
        });
        console.log(`[DEBUG] Stored conversation, total items: ${conversationHistory[currentDomain].length}`);
        
        // Save conversation to localStorage
        saveCurrentConversation(question);
        
    } catch (error) {
        messageDiv.querySelector('.answer-text').textContent = 'Error: ' + (error.message || 'Query failed');
        showToast(error.message || 'Query failed', 'error');
    } finally {
        askBtn.disabled = false;
        askBtnText.style.display = 'block';
        askBtnLoader.style.display = 'none';
        questionInput.disabled = false;
        questionInput.focus();
    }
}

// ===== CONVERSATION MANAGEMENT =====

function loadConversations() {
    const stored = localStorage.getItem('rag_conversations');
    conversations = stored ? JSON.parse(stored) : {};
    renderConversationsList();
}

function saveConversations() {
    localStorage.setItem('rag_conversations', JSON.stringify(conversations));
}

function createNewConversation() {
    if (!currentDomain) {
        showToast('Please select a domain first', 'error');
        return;
    }
    
    // Clear current conversation
    currentConversationId = null;
    conversationHistory[currentDomain] = [];
    messages.innerHTML = '';
    
    showToast('New conversation started', 'success');
}

function saveCurrentConversation(firstQuestion = null) {
    if (!currentDomain || !conversationHistory[currentDomain] || conversationHistory[currentDomain].length === 0) {
        return;
    }
    
    // Create new conversation ID if needed
    if (!currentConversationId) {
        currentConversationId = `conv_${Date.now()}`;
        const title = firstQuestion || conversationHistory[currentDomain][0].question;
        conversations[currentConversationId] = {
            id: currentConversationId,
            domain: currentDomain,
            title: title.substring(0, 50) + (title.length > 50 ? '...' : ''),
            history: [],
            created: Date.now(),
            updated: Date.now()
        };
    }
    
    // Update conversation
    conversations[currentConversationId].history = conversationHistory[currentDomain];
    conversations[currentConversationId].updated = Date.now();
    
    saveConversations();
    renderConversationsList();
}

function loadConversation(convId) {
    const conv = conversations[convId];
    if (!conv) return;
    
    // Switch domain if needed
    if (conv.domain !== currentDomain) {
        selectDomain(conv.domain);
    }
    
    // Load conversation
    currentConversationId = convId;
    conversationHistory[conv.domain] = conv.history;
    
    // Render messages
    messages.innerHTML = '';
    conv.history.forEach(item => {
        addMessage(item.question, item.answer, item.sources);
    });
    
    showToast('Conversation loaded', 'success');
    renderConversationsList();
}

function deleteConversation(convId) {
    if (confirm('Delete this conversation?')) {
        delete conversations[convId];
        
        if (currentConversationId === convId) {
            currentConversationId = null;
            conversationHistory[currentDomain] = [];
            messages.innerHTML = '';
        }
        
        saveConversations();
        renderConversationsList();
        showToast('Conversation deleted', 'success');
    }
}

function renderConversationsList(showAll = false) {
    const conversationsList = document.getElementById('conversationsList');
    if (!conversationsList) return;
    
    const convArray = Object.values(conversations)
        .sort((a, b) => b.updated - a.updated);
    
    if (convArray.length === 0) {
        conversationsList.innerHTML = '<div class="conversations-empty">No conversations yet</div>';
        return;
    }
    
    // Determine how many to show
    const displayCount = showAll ? convArray.length : Math.min(conversationsDisplayLimit, convArray.length);
    const displayConvs = convArray.slice(0, displayCount);
    
    // Group by domain
    const byDomain = {};
    displayConvs.forEach(conv => {
        if (!byDomain[conv.domain]) {
            byDomain[conv.domain] = [];
        }
        byDomain[conv.domain].push(conv);
    });
    
    let html = '';
    Object.keys(byDomain).sort().forEach(domain => {
        html += `<div class="conv-domain-group">
            <div class="conv-domain-name">${escapeHtml(domain)}</div>`;
        
        byDomain[domain].forEach(conv => {
            const isActive = conv.id === currentConversationId;
            const date = new Date(conv.updated).toLocaleDateString();
            html += `
                <div class="conv-item ${isActive ? 'active' : ''}" onclick="loadConversation('${conv.id}')">
                    <div class="conv-title">${escapeHtml(conv.title)}</div>
                    <div class="conv-meta">
                        <span>${date}</span>
                        <span>${conv.history.length} msgs</span>
                    </div>
                    <button class="conv-delete" onclick="event.stopPropagation(); deleteConversation('${conv.id}')">×</button>
                </div>
            `;
        });
        
        html += '</div>';
    });
    
    // Add "Load More" button if there are more conversations
    if (!showAll && convArray.length > conversationsDisplayLimit) {
        const remaining = convArray.length - conversationsDisplayLimit;
        html += `
            <div class="conv-load-more">
                <button class="btn-load-more" onclick="renderConversationsList(true)">
                    Load ${remaining} more...
                </button>
            </div>
        `;
    }
    
    conversationsList.innerHTML = html;
}

// Start
init();

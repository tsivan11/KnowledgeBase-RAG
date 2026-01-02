// State
let currentDomain = null;
let selectedFiles = [];
let conversationHistory = {};  // Store conversation history per domain

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
        sources.forEach(source => {
            const location = [
                source.page ? `p.${source.page}` : null,
                source.section ? `s.${source.section}` : null
            ].filter(Boolean).join(', ');
            
            html += `
                <div class="source-item">
                    <span class="source-ref">[${source.rank}]</span>
                    <span class="source-file">${escapeHtml(source.source)}</span>
                    ${location ? ` (${location})` : ''}
                    <div class="source-preview">${escapeHtml(source.text_preview)}</div>
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

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Event Listeners
function setupEventListeners() {
    // New Domain Modal
    newDomainBtn.addEventListener('click', () => {
        newDomainModal.classList.add('active');
        domainNameInput.value = '';
        domainNameInput.focus();
    });
    
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
            <div class="answer-text">Thinking...</div>
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
            result.sources.forEach(source => {
                const location = [
                    source.page ? `p.${source.page}` : null,
                    source.section ? `s.${source.section}` : null
                ].filter(Boolean).join(', ');
                
                answerHtml += `
                    <div class="source-item">
                        <span class="source-ref">[${source.rank}]</span>
                        <span class="source-file">${escapeHtml(source.source)}</span>
                        ${location ? ` (${location})` : ''}
                        <div class="source-preview">${escapeHtml(source.text_preview)}</div>
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
            sources: result.sources
        });
        console.log(`[DEBUG] Stored conversation, total items: ${conversationHistory[currentDomain].length}`);
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

// Start
init();

// ========================================================================
// SANSKRITY - Frontend JavaScript
// ChatGPT-inspired UI with Phosphor Icons
// ========================================================================

const API_BASE = '';  // Same origin

// ========================================================================
// STATE MANAGEMENT
// ========================================================================

let isRecording = false;
let recognition = null;
let chatHistory = [];
let sidebarOpen = true;

// ========================================================================
// INITIALIZATION
// ========================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeEventListeners();
    loadContextList();
    initializeSpeechRecognition();
    autoResizeTextarea();
    
    // Check mobile view
    if (window.innerWidth <= 768) {
        sidebarOpen = false;
        document.getElementById('sidebar').classList.remove('open');
    }
});

// ========================================================================
// TAB NAVIGATION
// ========================================================================

function initializeTabs() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.dataset.tab;
            
            // Update nav items
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            // Update tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`${tabId}-tab`).classList.add('active');
            
            // Close sidebar on mobile after selection
            if (window.innerWidth <= 768) {
                toggleSidebar();
            }
        });
    });
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebarOpen = !sidebarOpen;
    
    if (window.innerWidth <= 768) {
        sidebar.classList.toggle('open', sidebarOpen);
    } else {
        sidebar.classList.toggle('collapsed', !sidebarOpen);
    }
}

function newChat() {
    const chatMessages = document.getElementById('chatMessages');
    chatHistory = [];
    
    // Restore welcome screen
    chatMessages.innerHTML = `
        <div class="welcome-screen">
            <div class="welcome-logo">
                <span class="om-symbol">ॐ</span>
            </div>
            <h2>How can I help you today?</h2>
            <p class="welcome-subtitle">Ask me about Sanskrit scriptures, Ramayana, Mahabharata, or dharmic teachings</p>
            
            <div class="suggestion-grid">
                <button class="suggestion-card" onclick="setQuery('What is dharma according to Ramayana?')">
                    <i class="ph ph-book-open-text"></i>
                    <span>What is dharma according to Ramayana?</span>
                </button>
                <button class="suggestion-card" onclick="setQuery('Explain the concept of karma')">
                    <i class="ph ph-infinity"></i>
                    <span>Explain the concept of karma</span>
                </button>
                <button class="suggestion-card" onclick="setQuery('What did Lord Rama teach about duty?')">
                    <i class="ph ph-crown"></i>
                    <span>What did Lord Rama teach about duty?</span>
                </button>
                <button class="suggestion-card" onclick="setQuery('Summarize the Bhagavad Gita teachings')">
                    <i class="ph ph-scroll"></i>
                    <span>Summarize the Bhagavad Gita teachings</span>
                </button>
            </div>
        </div>
    `;
    
    // Switch to chat tab
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelector('.nav-item[data-tab="chat"]').classList.add('active');
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('chat-tab').classList.add('active');
}

// ========================================================================
// EVENT LISTENERS
// ========================================================================

function initializeEventListeners() {
    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleFileDrop);

    // Text context
    document.getElementById('addTextBtn').addEventListener('click', handleTextContext);

    // Query submission
    document.getElementById('submitBtn').addEventListener('click', handleQuery);
    
    const queryInput = document.getElementById('queryInput');
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            handleQuery();
        }
        // Enter without shift for single-line submit
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
            e.preventDefault();
            handleQuery();
        }
    });
    
    // Auto-resize textarea
    queryInput.addEventListener('input', autoResizeTextarea);

    // Voice input
    document.getElementById('voiceBtn').addEventListener('click', toggleVoiceInput);
}

function autoResizeTextarea() {
    const textarea = document.getElementById('queryInput');
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

// ========================================================================
// CHAT FUNCTIONS
// ========================================================================

function setQuery(query) {
    document.getElementById('queryInput').value = query;
    autoResizeTextarea();
    document.getElementById('queryInput').focus();
}

function addMessageToChat(type, content, metadata = null) {
    const chatMessages = document.getElementById('chatMessages');
    
    // Remove welcome screen if present
    const welcomeScreen = chatMessages.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    
    if (type === 'user') {
        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="ph ph-user"></i></div>
            <div class="message-content">
                <div class="message-text">${escapeHtml(content)}</div>
            </div>
        `;
    } else if (type === 'assistant') {
        let sourcesHtml = '';
        if (metadata && metadata.scripture_context && metadata.scripture_context.length > 0) {
            sourcesHtml = `
                <div class="message-sources">
                    <div class="sources-toggle" onclick="toggleSources(this)">
                        <i class="ph ph-books"></i> View Sources (${metadata.scripture_context.length})
                    </div>
                    <div class="sources-content" style="display: none;">
                        ${metadata.scripture_context.map(s => `
                            <div class="source-item">
                                <div class="source-text">${escapeHtml(s.text.substring(0, 200))}...</div>
                                <div class="source-ref">${s.book || ''} ${s.chapter || ''}.${s.shloka_num || ''}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        let citationHtml = '';
        if (metadata && metadata.citation) {
            const hasShloka = metadata.shloka && metadata.shloka.trim();
            const hasTranslation = metadata.translation && metadata.translation.trim();
            
            if (hasShloka || hasTranslation) {
                citationHtml = `
                    <div class="citation-block">
                        <div class="citation-toggle" onclick="toggleCitation(this)">
                            <i class="ph ph-scroll"></i> ${escapeHtml(metadata.citation)}
                            <i class="ph ph-caret-down citation-caret"></i>
                        </div>
                        <div class="citation-content" style="display: none;">
                            ${hasShloka ? `<div class="shloka-text">${escapeHtml(metadata.shloka)}</div>` : ''}
                            ${hasTranslation ? `<div class="translation-text">${escapeHtml(metadata.translation)}</div>` : ''}
                        </div>
                    </div>
                `;
            } else {
                citationHtml = `<div class="message-citation"><i class="ph ph-scroll"></i> ${escapeHtml(metadata.citation)}</div>`;
            }
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="ph ph-sparkle"></i></div>
            <div class="message-content">
                ${citationHtml}
                <div class="message-text">${formatResponse(content)}</div>
                ${sourcesHtml}
            </div>
        `;
    } else if (type === 'loading') {
        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="ph ph-sparkle"></i></div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function toggleSources(element) {
    const sourcesContent = element.nextElementSibling;
    const isHidden = sourcesContent.style.display === 'none';
    sourcesContent.style.display = isHidden ? 'block' : 'none';
    const count = sourcesContent.querySelectorAll('.source-item').length;
    element.innerHTML = isHidden 
        ? `<i class="ph ph-books"></i> Hide Sources` 
        : `<i class="ph ph-books"></i> View Sources (${count})`;
}

function toggleCitation(element) {
    const citationContent = element.nextElementSibling;
    const caret = element.querySelector('.citation-caret');
    const isHidden = citationContent.style.display === 'none';
    
    citationContent.style.display = isHidden ? 'block' : 'none';
    
    if (caret) {
        caret.className = isHidden 
            ? 'ph ph-caret-up citation-caret'
            : 'ph ph-caret-down citation-caret';
    }
}

function formatResponse(text) {
    // Convert line breaks to <br> and preserve paragraphs
    return escapeHtml(text)
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
}

function removeLoadingMessage() {
    const loadingMsg = document.querySelector('.chat-message.loading');
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

// ========================================================================
// FILE UPLOAD
// ========================================================================

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

async function uploadFile(file) {
    // Validate file type
    const allowedTypes = ['pdf', 'docx', 'txt'];
    const fileExt = file.name.split('.').pop().toLowerCase();

    if (!allowedTypes.includes(fileExt)) {
        showToast('Invalid file type. Please upload PDF, DOCX, or TXT files.', 'error');
        return;
    }

    // Validate file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('File too large. Maximum size is 10MB.', 'error');
        return;
    }

    // Show progress
    showUploadProgress(true);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/api/context/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            showToast(`✓ ${file.name} processed successfully! ${data.chunks_stored} chunks created.`, 'success');
            loadContextList();
        } else {
            throw new Error(data.detail || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        showUploadProgress(false);
        document.getElementById('fileInput').value = '';
    }
}

function showUploadProgress(show) {
    const uploadArea = document.getElementById('uploadArea');
    const uploadProgress = document.getElementById('uploadProgress');

    if (show) {
        uploadArea.style.display = 'none';
        uploadProgress.style.display = 'block';

        // Animate progress bar
        const progressFill = document.getElementById('progressFill');
        progressFill.style.width = '0%';
        setTimeout(() => progressFill.style.width = '90%', 100);
    } else {
        uploadArea.style.display = 'block';
        uploadProgress.style.display = 'none';
        document.getElementById('progressFill').style.width = '0%';
    }
}

// ========================================================================
// TEXT CONTEXT
// ========================================================================

async function handleTextContext() {
    const title = document.getElementById('textTitle').value.trim() || 'Untitled';
    const text = document.getElementById('textContent').value.trim();

    if (!text || text.length < 10) {
        showToast('Please enter at least 10 characters of text.', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/context/text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text, title })
        });

        const data = await response.json();

        if (response.ok && data.success) {
            showToast(`✓ Text context added! ${data.chunks_stored} chunks created.`, 'success');
            document.getElementById('textTitle').value = '';
            document.getElementById('textContent').value = '';
            loadContextList();
        } else {
            throw new Error(data.detail || 'Failed to add text context');
        }
    } catch (error) {
        console.error('Text context error:', error);
        showToast(`Error: ${error.message}`, 'error');
    }
}

// ========================================================================
// CONTEXT LIST
// ========================================================================

async function loadContextList() {
    try {
        const response = await fetch(`${API_BASE}/api/context/list`);
        const data = await response.json();

        if (response.ok && data.success) {
            renderContextList(data.contexts);
        } else {
            throw new Error('Failed to load context list');
        }
    } catch (error) {
        console.error('Load context list error:', error);
    }
}

function renderContextList(contexts) {
    const container = document.getElementById('contextList');

    if (!contexts || contexts.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="ph ph-folder-notch-open"></i>
                <span>No contexts uploaded yet</span>
            </div>
        `;
        return;
    }

    container.innerHTML = contexts.map(ctx => `
        <div class="context-item">
            <div class="context-icon">
                <i class="ph ph-file-text"></i>
            </div>
            <div class="context-info">
                <div class="context-filename">${escapeHtml(ctx.filename)}</div>
                <div class="context-meta">
                    ${(ctx.file_type || 'text').toUpperCase()} • ${ctx.chunk_count} chunks
                </div>
            </div>
            <button class="context-delete" onclick="deleteContext('${escapeHtml(ctx.filename)}')" title="Delete">
                <i class="ph ph-trash"></i>
            </button>
        </div>
    `).join('');
}

async function deleteContext(filename) {
    if (!confirm(`Delete "${filename}"?`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/context/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (response.ok && data.success) {
            showToast(`✓ Deleted ${filename}`, 'success');
            loadContextList();
        } else {
            throw new Error(data.detail || 'Delete failed');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showToast(`Error: ${error.message}`, 'error');
    }
}

// ========================================================================
// SPEECH-TO-TEXT (STT) - Backend Whisper
// ========================================================================

let mediaRecorder = null;
let audioChunks = [];

function initializeSpeechRecognition() {
    // Check if MediaRecorder is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('MediaRecorder not supported');
        document.getElementById('voiceBtn').style.display = 'none';
        return;
    }
    
    // Will initialize on first use to avoid permission prompt on page load
    console.log('Voice input ready (backend Whisper STT)');
}

async function toggleVoiceInput() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Use webm format (widely supported)
        const mimeType = MediaRecorder.isTypeSupported('audio/webm') 
            ? 'audio/webm' 
            : 'audio/ogg';
        
        mediaRecorder = new MediaRecorder(stream, { mimeType });
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
            
            // Create audio blob
            const audioBlob = new Blob(audioChunks, { type: mimeType });
            
            // Send to backend for transcription
            await transcribeAudio(audioBlob, mimeType);
        };
        
        mediaRecorder.start();
        isRecording = true;
        document.getElementById('voiceBtn').classList.add('recording');
        showToast('🎤 Recording... Click again to stop', 'success');
        
    } catch (error) {
        console.error('Failed to start recording:', error);
        showToast('Microphone access denied or not available', 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        isRecording = false;
        document.getElementById('voiceBtn').classList.remove('recording');
    }
}

async function transcribeAudio(audioBlob, mimeType) {
    showToast('🔄 Transcribing...', 'success');
    
    try {
        const formData = new FormData();
        const ext = mimeType.includes('webm') ? 'webm' : 'ogg';
        formData.append('audio', audioBlob, `recording.${ext}`);
        
        const response = await fetch(`${API_BASE}/api/stt`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            document.getElementById('queryInput').value = data.text;
            autoResizeTextarea();
            showToast('✓ Transcription complete', 'success');
        } else {
            throw new Error(data.detail || 'Transcription failed');
        }
    } catch (error) {
        console.error('Transcription error:', error);
        showToast(`Error: ${error.message}`, 'error');
    }
}

// ========================================================================
// QUERY PROCESSING
// ========================================================================

async function handleQuery() {
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();

    if (!query || query.length < 3) {
        showToast('Please enter a question (at least 3 characters).', 'error');
        return;
    }

    // Add user message to chat
    addMessageToChat('user', query);
    
    // Clear input
    queryInput.value = '';
    autoResizeTextarea();
    
    // Show loading
    addMessageToChat('loading', '');

    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        const data = await response.json();

        // Remove loading message
        removeLoadingMessage();

        if (response.ok) {
            addMessageToChat('assistant', data.answer, {
                scripture_context: data.scripture_context,
                citation: data.citation,
                shloka: data.shloka,
                translation: data.translation
            });
        } else {
            throw new Error(data.detail || 'Query failed');
        }
    } catch (error) {
        console.error('Query error:', error);
        removeLoadingMessage();
        addMessageToChat('assistant', `Sorry, an error occurred: ${error.message}`);
        showToast(`Error: ${error.message}`, 'error');
    }
}

// ========================================================================
// UTILITIES
// ========================================================================

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const iconEl = toast.querySelector('.toast-icon');
    const msgEl = toast.querySelector('.toast-message');
    
    // Set icon based on type
    if (iconEl) {
        iconEl.className = `toast-icon ph ${type === 'error' ? 'ph-x-circle' : 'ph-check-circle'}`;
    }
    
    if (msgEl) {
        msgEl.textContent = message;
    } else {
        toast.textContent = message;
    }
    
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Configurable Backend API URL (for Vercel frontend + Render backend)
const BACKEND_URL = window.API_BASE_URL || (window.location.origin.includes('vercel.app') ? 'https://medical-rag-chatbot-7hoq.onrender.com' : '');

const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const submitBtn = document.getElementById('submit-btn');
const newChatBtn = document.getElementById('new-chat-btn');

// Navigation Elements
const navItems = document.querySelectorAll('.nav-item');
const contentSections = document.querySelectorAll('.content-section');
const displaySessionId = document.getElementById('display-session-id');
const refreshSessionBtn = document.getElementById('refresh-session-btn');

// Generate valid UUID v4
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
let sessionId = localStorage.getItem('med_rag_session_id') || generateUUID();
localStorage.setItem('med_rag_session_id', sessionId);
if (displaySessionId) displaySessionId.textContent = sessionId;

function getAvatarUrl(type) {
    if (type === 'user') return "/static/images/user-avatar.svg";
    return "/static/images/bot-avatar.svg";
}

function addMessage(text, type, intent = null) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${type}-message`);
    
    let intentHtml = '';
    if (intent) {
        intentHtml = `<span class="intent-tag intent-${intent.toLowerCase()}">${intent}</span>`;
    }

    messageDiv.innerHTML = `
        <img src="${getAvatarUrl(type)}" alt="${type} Avatar" class="message-avatar">
        <div class="message-content-wrapper">
            <div class="message-content">
                ${intentHtml}
                <p>${text.replace(/\n/g, '<br>')}</p>
                <div class="timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message', 'ai-message', 'typing-indicator');
    typingDiv.innerHTML = `
        <img src="${getAvatarUrl('ai')}" alt="AI Avatar" class="message-avatar">
        <div class="message-content-wrapper">
            <div class="message-content typing">
                <span class="status-text">Thinking...</span>
                <div class="dots"><span></span><span></span><span></span></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingDiv;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;

    // Disable input
    userInput.value = '';
    userInput.disabled = true;
    submitBtn.disabled = true;

    // Add user message to UI
    addMessage(message, 'user');

    // Show typing indicator
    const typingIndicator = showTypingIndicator();

    try {
        const formData = new FormData();
        formData.append('message', message);
        formData.append('session_id', sessionId);

        const response = await fetch(`${BACKEND_URL}/chat/stream`, {
            method: 'POST',
            body: formData
        });

        // Prepare AI message bubble (but don't add to DOM yet)
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'ai-message');
        
        const avatarImg = document.createElement('img');
        avatarImg.src = getAvatarUrl('ai');
        avatarImg.alt = "AI Avatar";
        avatarImg.classList.add('message-avatar');
        
        const wrapperDiv = document.createElement('div');
        wrapperDiv.classList.add('message-content-wrapper');

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        
        const intentTag = document.createElement('span');
        intentTag.classList.add('intent-tag');
        intentTag.style.display = 'none';
        
        const textP = document.createElement('p');
        
        const timestampDiv = document.createElement('div');
        timestampDiv.classList.add('timestamp');
        timestampDiv.style.display = 'none'; // Hide until done

        const actionsDiv = document.createElement('div');
        actionsDiv.classList.add('message-actions');
        actionsDiv.innerHTML = '<i data-feather="copy" class="action-icon" title="Copy text"></i>';

        contentDiv.appendChild(intentTag);
        contentDiv.appendChild(textP);
        contentDiv.appendChild(timestampDiv);
        wrapperDiv.appendChild(contentDiv);
        wrapperDiv.appendChild(actionsDiv);
        
        messageDiv.appendChild(avatarImg);
        messageDiv.appendChild(wrapperDiv);

        // Typewriter queue for smooth word-by-word / char-by-char streaming
        let displayedText = '';
        let charQueue = [];
        let isStreamFinished = false;
        let bubbleAdded = false;

        const typewriterPromise = new Promise((resolve) => {
            const timer = setInterval(() => {
                if (charQueue.length > 0) {
                    // Dynamically scale speed: 1 char per tick for smooth typing, slightly faster if queue builds up
                    const chunkSize = charQueue.length > 80 ? 4 : (charQueue.length > 30 ? 2 : 1);
                    const charsToAppend = charQueue.splice(0, chunkSize).join('');
                    
                    if (!bubbleAdded) {
                        typingIndicator.remove();
                        chatMessages.appendChild(messageDiv);
                        bubbleAdded = true;
                    }
                    
                    displayedText += charsToAppend;
                    textP.innerHTML = displayedText.replace(/\n/g, '<br>');
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                } else if (isStreamFinished) {
                    clearInterval(timer);
                    resolve();
                }
            }, 18);
        });

        // Read the SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.error) {
                            isStreamFinished = true;
                            typingIndicator.remove();
                            addMessage('Error: ' + data.error, 'ai');
                            return;
                        }
                        if (data.done) continue;

                        // Update status text on status event
                        if (data.status) {
                            const statusEl = typingIndicator.querySelector('.status-text');
                            if (statusEl) statusEl.textContent = data.status;
                        }

                        // Show intent tag
                        if (data.intent && data.intent !== 'PROCESSING' && intentTag.style.display === 'none') {
                            intentTag.textContent = data.intent;
                            intentTag.classList.add('intent-' + data.intent.toLowerCase());
                            intentTag.style.display = '';
                        }

                        // Push tokens into character queue
                        if (data.token) {
                            charQueue.push(...data.token.split(''));
                        }
                    } catch (parseErr) { }
                }
            }
        }

        isStreamFinished = true;
        await typewriterPromise;

        // Streaming done — show timestamp
        timestampDiv.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        timestampDiv.style.display = '';

        if (!bubbleAdded && !displayedText) {
            typingIndicator.remove();
            addMessage('No response received.', 'ai');
        }

    } catch (error) {
        typingIndicator.remove();
        addMessage('Sorry, something went wrong. Please check your connection.', 'ai');
        console.error('Error:', error);
    } finally {
        userInput.disabled = false;
        submitBtn.disabled = false;
        userInput.focus();
    }
});

newChatBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear this session?')) {
        chatMessages.innerHTML = `
            <div class="message ai-message">
                <img src="${getAvatarUrl('ai')}" alt="AI Avatar" class="message-avatar">
                <div class="message-content-wrapper">
                    <div class="message-content">
                        <div class="disclaimer">
                            <i data-feather="alert-triangle"></i>
                            <span>DISCLAIMER: This system is for educational purposes. Consult a physician for medical advice.</span>
                        </div>
                        <p>New session started. How can I help you today?</p>
                        <div class="timestamp">Just now</div>
                    </div>
                </div>
            </div>
        `;
        feather.replace();
    }
});

// Tab Switching Logic
navItems.forEach(item => {
    item.addEventListener('click', () => {
        const targetId = item.id.replace('nav-', 'section-');
        
        // Update active nav
        navItems.forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');
        
        // Update active section
        contentSections.forEach(section => section.classList.remove('active'));
        const targetSection = document.getElementById(targetId);
        if (targetSection) {
            targetSection.classList.add('active');
        }
        
        // Refresh icons if needed
        if (window.feather) {
            feather.replace();
        }
    });
});

if (refreshSessionBtn) {
    refreshSessionBtn.addEventListener('click', () => {
        if (confirm('Create a new session identity? History will be separate.')) {
            const newId = generateUUID();
            localStorage.setItem('med_rag_session_id', newId);
            displaySessionId.textContent = newId;
            window.location.reload();
        }
    });
}

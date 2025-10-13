let currentPrediction = "No prediction yet.";

function toggleChat() {
    const body = document.getElementById('chat-body');
    const footer = document.getElementById('chat-footer');
    if (!body || !footer) {
        console.warn('Chat body or footer not found.');
        return;
    }
    if (body.style.display === 'none') {
        body.style.display = 'block';
        footer.style.display = 'block';
        body.scrollTop = body.scrollHeight;
    } else {
        body.style.display = 'none';
        footer.style.display = 'none';
    }
}

function appendMessage(sender, message) {
    const chatBody = document.getElementById('chat-body');
    if (!chatBody) {
        console.warn('Chat body not found.');
        return;
    }
    const msgDiv = document.createElement('div');
    msgDiv.className = sender === 'bot' ? 'bot-message' : 'user-message';
    msgDiv.textContent = sender === 'bot' ? `Bot: ${message}` : `You: ${message}`;
    chatBody.appendChild(msgDiv);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function sendMessage() {
    const inputField = document.getElementById('chat-input');
    if (!inputField) {
        console.warn('Chat input field not found.');
        return;
    }
    const message = inputField.value.trim();
    if (message === "") return;
    appendMessage('user', message);
    inputField.value = '';
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message,
            prediction: currentPrediction // Send the current prediction result
        })
    })
    .then(response => response.json())
    .then(data => {
        appendMessage('bot', data.response);
    })
    .catch(error => {
        console.error('Chat error:', error);
        appendMessage('bot', 'Sorry, I am having trouble connecting.');
    });
}

// Initialize the chat and welcome message
window.onload = function() {
    appendMessage('bot', "Hello! I'm your Medical Assistant bot. Ask me about your prediction or symptoms.");
};

// Example diabetes form handler - keeps currentPrediction in sync with predictions
(function attachDiabetesHandler(){
    const diabetesForm = document.getElementById('diabetes-form');
    if (!diabetesForm) return;
    diabetesForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(diabetesForm);
        const formObj = {};
        formData.forEach((value, key) => {
            formObj[key] = value;
        });
        fetch('/predict_diabetes', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            currentPrediction = data.prediction;
            appendMessage('bot', `Diabetes Prediction: ${data.prediction}`);
            // Render result to page
            const container = document.getElementById('diabetes-result');
            if (container) {
                const expl = data.explanation || {};
                const imgHtml = expl.image ? `<img src="data:image/png;base64,${expl.image}" alt="SHAP explanation" style="max-width:100%;height:auto;"/>` : '';
                let valuesHtml = '';
                if (expl.values) {
                    valuesHtml = '<table class="table table-sm table-bordered"><thead><tr><th>Feature</th><th>SHAP Value</th></tr></thead><tbody>' +
                        Object.entries(expl.values).map(([k,v]) => `<tr><td>${k}</td><td>${v.toFixed(4)}</td></tr>`).join('') +
                        '</tbody></table>';
                }
                container.innerHTML = `
                    <div class="card p-3">
                        <h5>Prediction: ${data.prediction}</h5>
                        <div class="row">
                            <div class="col-md-6">${imgHtml}</div>
                            <div class="col-md-6">${valuesHtml}</div>
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Prediction error:', error);
            appendMessage('bot', 'Error getting diabetes prediction.');
        });
    });
})();

// Attach handler for view-inputs links (delegated in case table is loaded later)
document.addEventListener('click', function(e){
    const el = e.target.closest && e.target.closest('.view-inputs');
    if (!el) return;
    e.preventDefault();
    const raw = el.getAttribute('data-input');
    try {
        const obj = JSON.parse(raw || '{}');
        // Pretty-print in an alert (or replace with modal if desired)
        alert(JSON.stringify(obj, null, 2));
    } catch (err) {
        console.error('Failed to parse input data', err);
        alert('Unable to show input details.');
    }
});

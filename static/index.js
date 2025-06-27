let selectedLanguage = 'en';
const detectBtn = document.querySelector('.btn-submit');

const textarea = document.querySelector('.headline');
const enButton = document.querySelector('.btn-language img[alt="english"]').parentElement;
const roButton = document.querySelector('.btn-language img[alt="romanian"]').parentElement;
const huButton = document.querySelector('.btn-language img[alt="hungarian"]').parentElement;
const resultElement = document.querySelector('#result');

highlightSelected(enButton);

async function makePrediction(text, endpoint) {
    if (!text) {
        resultElement.textContent = 'Please enter a headline.';
        resultElement.classList.remove('blink');
        void resultElement.offsetWidth;
        resultElement.classList.add('blink');
        setTimeout(() => resultElement.classList.remove('blink'), 600);
        return;
    }

    try {
        const response = await fetch(`http://localhost:5000${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        const data = await response.json();

        if (data.error) {
            resultElement.textContent = `Error: ${data.error}`;
            resultElement.classList.remove('blink');
            void resultElement.offsetWidth;
            resultElement.classList.add('blink');
            setTimeout(() => resultElement.classList.remove('blink'), 600);
            return;
        }

        const prediction = data.prediction;
        const confidence = data.confidence;
        resultElement.innerHTML = `Your headline is likely: <span class="prediction-text">${prediction === 1 ? 'Clickbait' : 'Not Clickbait'}</span>`;
        const predictionSpan = resultElement.querySelector('.prediction-text');
        predictionSpan.classList.remove('clickbait', 'not-clickbait');
        predictionSpan.classList.add(prediction === 1 ? 'clickbait' : 'not-clickbait');
        resultElement.classList.remove('blink');
        void resultElement.offsetWidth;
        resultElement.classList.add('blink');
        setTimeout(() => resultElement.classList.remove('blink'), 600);
        renderConfidenceChart(confidence);
    } catch (error) {
        resultElement.textContent = `Error: ${error.message}`;
        resultElement.classList.remove('blink');
        void resultElement.offsetWidth;
        resultElement.classList.add('blink');
        setTimeout(() => resultElement.classList.remove('blink'), 600);
    }
}

enButton.addEventListener('click', () => {
    selectedLanguage = 'en';
    highlightSelected(enButton);
});

roButton.addEventListener('click', () => {
    selectedLanguage = 'ro';
    highlightSelected(roButton);
});

huButton.addEventListener('click', () => {
    selectedLanguage = 'hu';
    highlightSelected(huButton);
});

function highlightSelected(button) {
    document.querySelectorAll('.btn-language').forEach(btn => btn.classList.remove('selected'));
    button.classList.add('selected');
}

detectBtn.addEventListener('click', async () => {
    const text = textarea.value.trim();
    const warningWrapper = document.querySelector('.warning');
    const warningElement = document.getElementById('languageWarning');

    //clear previous warning
    warningWrapper.style.display = 'none';
    warningElement.textContent = '';

    if (!text) {
        resultElement.textContent = 'Please enter a headline.';
        return;
    }

    let detectedLang = null;

    //detect language
    try {
        const langResponse = await fetch('http://localhost:5000/detect_language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const langData = await langResponse.json();

        if (langData.language) {
            detectedLang = langData.language;
        } else {
            warningElement.textContent = "Language detection failed.";
            warningWrapper.style.display = 'flex';
        }
    } catch (error) {
        console.error("Language detection error:", error);
        warningElement.textContent = "Error during language detection.";
    }

    //warning
    if (detectedLang && selectedLanguage && detectedLang !== selectedLanguage) {
        warningElement.textContent = "Are you sure you selected the right language?";
        warningWrapper.style.display = 'flex';
    }

    //prediction
    if (!selectedLanguage) {
        resultElement.textContent = 'Please select a language first.';
        return;
    }

    const endpoint = `/predict/${selectedLanguage}`;
    makePrediction(text, endpoint);
});


//--------------------------- chart
let chart;

function renderConfidenceChart(confidence) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Clickbait', 'Not Clickbait'],
            datasets: [{
                label: 'Model Confidence',
                data: [
                    (confidence.clickbait * 100).toFixed(2),
                    (confidence.not_clickbait * 100).toFixed(2)
                ],
                backgroundColor: ['rgba(240, 150, 82, 0.75)', 'rgba(37, 100, 202, 0.75)'],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: value => `${value}%`
                    }
                }
            }
        }
    });
}

//----------------------------- paste and delete

const pasteBtn = document.getElementById('pasteBtn');
const deleteBtn = document.getElementById('deleteBtn');

function updateIcons() {
    const text = textarea.value.trim();
    if (text.length === 0) {
        pasteBtn.style.display = 'inline';
        deleteBtn.style.display = 'none';
    } else {
        pasteBtn.style.display = 'none';
        deleteBtn.style.display = 'inline';
    }
}

textarea.addEventListener('input', updateIcons);
updateIcons();

//paste button 
pasteBtn.addEventListener('click', async () => {
    try {
        const text = await navigator.clipboard.readText();
        textarea.value = text;
        updateIcons(); // update button visibility
    } catch (err) {
        alert('Could not read from clipboard. Make sure clipboard permissions are allowed.');
        console.error('Clipboard error:', err);
    }
});

//delete button
deleteBtn.addEventListener('click', () => {
    textarea.value = '';
    updateIcons(); // update button visibility
});

//character counter
const charCount = document.querySelector('.char-count');
textarea.addEventListener('input', () => {
    charCount.textContent = `${textarea.value.length} / 200`;
    updateIcons(); 
});

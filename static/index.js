let selectedLanguage = 'en';
const detectBtn = document.querySelector('.btn-submit');

const textarea = document.querySelector('.headline');
const enButton = document.querySelector('.btn-language img[alt="english"]').parentElement;
const roButton = document.querySelector('.btn-language img[alt="romanian"]').parentElement;
const huButton = document.querySelector('.btn-language img[alt="hungarian"]').parentElement;
const resultElement = document.querySelector('#result');
const loadingIndicator = document.getElementById('csvLoading');

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
        const response = await fetch(endpoint, {
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

async function processCsvFile() { 
    const file = csvInput.files[0];
    if (!file) return;

    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: async (results) => {
            const rows = results.data;
            const headers = results.meta.fields;

            if (!headers.includes("Headline")) {
                alert("CSV must contain a column named 'Headline'.");
                return;
            }

            const outputRows = [];

            for (const row of rows) {
                const headline = row.Headline?.trim();

                if (!headline) {
                    outputRows.push({ ...row, Prediction: 'Error: Empty' });
                    continue;
                }

                try {
                    const res = await fetch(`/predict/${selectedLanguage}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: headline })
                    });

                    const data = await res.json();

                    if (data.error) {
                        outputRows.push({ ...row, Prediction: `Error: ${data.error}` });
                    } else {
                        const label = data.prediction === 1 ? 1 : 0;
                        outputRows.push({ ...row, Prediction: label });
                    }
                } catch (err) {
                    outputRows.push({ ...row, Prediction: 'Error: Request failed' });
                }
            }

            const newCsv = Papa.unparse(outputRows);
            const blob = new Blob([newCsv], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);

            const link = document.createElement('a');
            link.href = url;
            link.download = file.name.replace('.csv', '_with_predictions.csv');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        },
        error: (err) => {
            alert("Error parsing CSV: " + err.message);
        }
    });
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

    warningWrapper.style.display = 'none';
    warningElement.textContent = '';
    resultElement.textContent = '';

    if (csvUploaded) {
        processCsvFile();
        return;
    }

    const words = text.split(/\s+/).filter(word => /\w/.test(word));
    if (words.length < 4) {
        resultElement.textContent = 'Please enter at least 4 words.';
        flashResult();
        return;
    }

    if (!text) {
        resultElement.textContent = 'Please enter a headline.';
        return;
    }

    let detectedLang = null;

    try {
        const langResponse = await fetch('/detect_language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const langData = await langResponse.json();
        if (langData.language) detectedLang = langData.language;
        else warningElement.textContent = "Language detection failed.";
    } catch (error) {
        warningElement.textContent = "Error during language detection.";
        console.error(error);
    }

    if (detectedLang && selectedLanguage && detectedLang !== selectedLanguage) {
        warningElement.textContent = "Are you sure you selected the right language?";
        warningWrapper.style.display = 'flex';
    }

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
const csvBtn = document.getElementById('csvBtn');
const csvInput = document.getElementById('csvInput');
const csvStatus = document.getElementById('csvStatus');
const originalPlaceholder = textarea.placeholder;
let csvUploaded = false;

function updateIcons() {
    const text = textarea.value.trim();
    if (text.length === 0 && !csvUploaded) {
        pasteBtn.style.display = 'inline';
        deleteBtn.style.display = 'none';
    } else {
        pasteBtn.style.display = 'none';
        deleteBtn.style.display = 'inline';
    }
}

textarea.addEventListener('input', updateIcons);
updateIcons();

//paste button functionality
pasteBtn.addEventListener('click', async () => {
    try {
        const text = await navigator.clipboard.readText();
        textarea.value = text;
        updateIcons(); 
    } catch (err) {
        alert('Could not read from clipboard. Make sure clipboard permissions are allowed.');
        console.error('Clipboard error:', err);
    }
});

//delete button functionality
deleteBtn.addEventListener('click', () => {
    textarea.value = '';

    if (csvUploaded) {
        csvInput.value = ''; 
        csvStatus.textContent = '';
        csvUploaded = false;
        textarea.disabled = false;
        textarea.placeholder = originalPlaceholder;
    }

    resultElement.textContent = '';
    updateIcons();

});

//character counter
const charCount = document.querySelector('.char-count');
textarea.addEventListener('input', () => {
    charCount.textContent = `${textarea.value.length} / 200`;
    updateIcons();
});

//csv
csvBtn.addEventListener('click', () => {
    csvInput.click();
});

csvInput.addEventListener('change', () => {
    const file = csvInput.files[0];

    if (file && file.name.endsWith('.csv')) {
        textarea.value = '';
        textarea.disabled = true;
        textarea.placeholder = '';
        csvUploaded = true;

        csvStatus.textContent = `${file.name}`;
        csvStatus.style.color = '#58A3AD';
    } else {
        csvStatus.textContent = 'Invalid file. Please upload a CSV.';
        csvStatus.style.color = 'red';
        csvUploaded = false;
        textarea.disabled = false;
    }
    updateIcons();
});


// dark mode

const darkBtn = document.getElementById('darkBtn');
const lightBtn = document.getElementById('lightBtn');

function toggleTheme() {
    const isDark = document.body.classList.toggle('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// Add event listeners for both buttons
darkBtn.addEventListener('click', toggleTheme);
lightBtn.addEventListener('click', toggleTheme);

// Apply saved theme on load
window.addEventListener('DOMContentLoaded', () => {
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-mode');
    }
});

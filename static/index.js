let selectedLanguage = 'en';
let selectedModel = 'bert'

let isCancelled = false; 

const detectBtn = document.querySelector('.btn-submit');

const textarea = document.querySelector('.headline');
const enButton = document.querySelector('.btn-language img[alt="english"]').parentElement;
const roButton = document.querySelector('.btn-language img[alt="romanian"]').parentElement;
const huButton = document.querySelector('.btn-language img[alt="hungarian"]').parentElement;

const bertButton = document.querySelector('.btn-model[data-model="bert"]');
const svmButton = document.querySelector('.btn-model[data-model="svm"]');
const rfButton = document.querySelector('.btn-model[data-model="rf"]');
const lstmButton = document.querySelector('.btn-model[data-model="lstm"]');

const resultElement = document.querySelector('#result');
const loadingIndicator = document.getElementById('csvLoading');

const cancelButton = document.getElementById('cancelProcessingBtn');

highlightSelectedLanguage(enButton);
highlightSelectedModel(bertButton);

cancelButton.addEventListener('click', () => {
    isCancelled = true;
    const progressContainer = document.getElementById('progressContainer');
    progressContainer.style.display = 'none';
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressLabel').textContent = '0%';
    resultElement.textContent = 'Processing cancelled.';
    resultElement.classList.remove('blink');
    void resultElement.offsetWidth;
    resultElement.classList.add('blink');
    setTimeout(() => resultElement.classList.remove('blink'), 600);
});

// Result
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

// CSV
async function processCsvFile() { 
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressLabel = document.getElementById('progressLabel');

    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    progressLabel.textContent = '0%';
    isCancelled = false; 

    const file = csvInput.files[0];
    if (!file) return;

    let rows;
    let fileExtension = file.name.split('.').pop().toLowerCase();

    if (fileExtension === 'csv') {
        const parsed = await new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                skipEmptyLines: true,
                complete: results => resolve(results),
                error: err => reject(err)
            });
        });

        rows = parsed.data;

        if (!parsed.meta.fields.includes("Headline")) {
            alert("CSV must contain a column named 'Headline'.");
            progressContainer.style.display = 'none';
            return;
        }
    } else if (fileExtension === 'xlsx') {
        const data = await file.arrayBuffer();
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const sheet = workbook.Sheets[sheetName];
        rows = XLSX.utils.sheet_to_json(sheet, { defval: '' });

        if (!Object.keys(rows[0]).includes("Headline")) {
            alert("Excel must contain a column named 'Headline'.");
            progressContainer.style.display = 'none';
            return;
        }
    } else {
        alert('Unsupported file format.');
        progressContainer.style.display = 'none';
        return;
    }

    const outputRows = [];

    for (const row of rows) {
        if (isCancelled) {
            // If cancelled, clean up and exit
            progressContainer.style.display = 'none';
            return;
        }

        const headline = row.Headline?.trim();
        if (!headline) {
            outputRows.push({ ...row, Prediction: 'Error: Empty' });
            continue;
        }

        try {
            const endpoint = `/predict/${selectedLanguage}${selectedModel === 'bert' ? '' : '/' + selectedModel}`;
            const res = await fetch(endpoint, {
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

        const percent = Math.round(((outputRows.length / rows.length) * 100));
        progressBar.style.width = `${percent}%`;
        progressLabel.textContent = `${percent}%`;
    }

    if (isCancelled) {
        // If cancelled during final processing, clean up and exit
        progressContainer.style.display = 'none';
        return;
    }

    // Export to CSV
    const newCsv = Papa.unparse(outputRows);
    const blob = new Blob([newCsv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = file.name.replace(/\.(csv|xlsx)$/, '_with_predictions.csv');
    document.body.appendChild(link);
    link.click();
    progressContainer.style.display = 'none';
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

//sentiment
async function analyzeSentiment(text) {
    if (!text) return;

    try {
        const response = await fetch('/sentiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        const data = await response.json();
        if (data.error) {
            console.error("Sentiment error:", data.error);
            return;
        }

        const sentimentBox = document.getElementById('sentimentResult');
        sentimentBox.innerHTML = `
            <div><strong>Sentiment:</strong> ${data.sentiment} (${data.polarity.toFixed(2)})</div>
            <div><strong>Subjectivity:</strong> ${data.subjectivity_description} (${data.subjectivity.toFixed(2)})</div>
        `;
        sentimentBox.style.display = 'block';
        sentimentCharts.style.display = 'block'
        renderPolarityPie(data.polarity);
        renderSubjectivityPie(data.subjectivity);

    } catch (error) {
        console.error("Sentiment fetch failed:", error);
    }

}

async function analyzeSentimentRO(text) {
    try {
        const response = await fetch('/sentiment/ro', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        const data = await response.json();
        if (data.error) {
            console.error("Error:", data.error);
            return;
        }

        const sentimentBox = document.getElementById('sentimentResult');
        sentimentBox.innerHTML = `
            <div><strong>Sentiment:</strong> ${data.sentiment} (${(data.confidence * 100).toFixed(1)}%)</div>
            <div><strong>Polarity:</strong> ${data.polarity}</div>
            <div><strong>Subjectivity:</strong> ${data.subjectivity}</div>
        `;
        sentimentBox.style.display = 'block';
        renderPolarityPie(polarity);
        renderSubjectivityPie(subjectivity);

    } catch (err) {
        console.error("RO sentiment fetch error:", err);
    }

}

async function analyzeSentimentHU(text) {
     try {
        const response = await fetch('/sentiment/hu', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        if (data.error) {
            console.error("HU sentiment error:", data.error);
            return;
        }
        const box = document.getElementById('sentimentResult');
        box.innerHTML = `
            <div><strong>Sentiment:</strong> ${data.sentiment} (${(data.confidence * 100).toFixed(1)}%)</div>
            <div><strong>Polarity:</strong> ${data.polarity}</div>
            <div><strong>Subjectivity:</strong> ${data.subjectivity}</div>
        `;
        box.style.display = 'block';
        renderPolarityPie(polarity);
        renderSubjectivityPie(subjectivity);
    } catch (err) {
        console.error("Error calling HU sentiment:", err);
    }

}




// HIGHTLIGHT AND SELECT BUTTONS
//language
enButton.addEventListener('click', () => {
    selectedLanguage = 'en';
    highlightSelectedLanguage(enButton);
});

roButton.addEventListener('click', () => {
    selectedLanguage = 'ro';
    highlightSelectedLanguage(roButton);
});

huButton.addEventListener('click', () => {
    selectedLanguage = 'hu';
    highlightSelectedLanguage(huButton);
});

//models
bertButton.addEventListener('click', () => {
    selectedModel = 'bert';
    highlightSelectedModel(bertButton);
});

svmButton.addEventListener('click', () => {
    selectedModel = 'svm';
    highlightSelectedModel(svmButton);
});

rfButton.addEventListener('click', () => {
    selectedModel = 'rf';
    highlightSelectedModel(rfButton);
});

lstmButton.addEventListener('click', () => {
    selectedModel = 'lstm';
    highlightSelectedModel(lstmButton);
});

function highlightSelectedLanguage(button) {
    document.querySelectorAll('.btn-language').forEach(btn => btn.classList.remove('selectedLang'));
    button.classList.add('selectedLang');
}

function highlightSelectedModel(button) {
    document.querySelectorAll('.btn-model').forEach(btn => btn.classList.remove('selectedModel'));
    button.classList.add('selectedModel');
}

// -------------------------------------- Detection
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

    //const endpoint = `/predict/${selectedLanguage}`;
    const endpoint = `/predict/${selectedLanguage}${selectedModel === 'bert' ? '' : '/' + selectedModel}`;

    makePrediction(text, endpoint);

    if (selectedLanguage === 'en') {
        analyzeSentiment(text);
    }
    if (selectedLanguage === 'ro') {
        analyzeSentimentRO(text);
    }
    if (selectedLanguage === 'hu') {
        analyzeSentimentHU(text);
    }

});


//------------------------------------- CHART
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


let polarityChart, subjectivityChart;

function renderPolarityPie(polarity) {
    const ctx = document.getElementById('polarityChart').getContext('2d');
    if (polarityChart) polarityChart.destroy();

    const value = Math.abs(polarity);
    const isPositive = polarity >= 0;

    const label = isPositive ? 'Positive' : 'Negative';
    const color = isPositive ? '#4caf50' : '#f44336';

    polarityChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: [label, 'Other'],
            datasets: [{
                data: [value, 1 - value],
                backgroundColor: [color, '#e0e0e0']
            }]
        },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: 'Polarity Confidence'
                }
            }
        }
    });
}

function renderSubjectivityPie(subjectivity) {
    const ctx = document.getElementById('subjectivityChart').getContext('2d');
    if (subjectivityChart) subjectivityChart.destroy();

    subjectivityChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Subjective', 'Objective'],
            datasets: [{
                data: [subjectivity, 1 - subjectivity],
                backgroundColor: ['#2196f3', '#9e9e9e']
            }]
        },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: 'Subjectivity'
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

    if (file && file.name.endsWith('.csv') || file.name.endsWith('.xlsx')) {
        textarea.value = '';
        textarea.disabled = true;
        textarea.placeholder = '';
        csvUploaded = true;

        csvStatus.textContent = `${file.name}`;
        csvStatus.style.color = '#58A3AD';
    } else {
        csvStatus.textContent = 'Invalid file. Please upload a CSV or Excel file.';
        csvStatus.style.color = 'red';
        csvUploaded = false;
        textarea.disabled = false;
    }
    updateIcons();
});


// ------------------------------------- dark mode

const darkBtn = document.getElementById('darkBtn');
const lightBtn = document.getElementById('lightBtn');

function toggleTheme() {
    const isDark = document.body.classList.toggle('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

darkBtn.addEventListener('click', toggleTheme);
lightBtn.addEventListener('click', toggleTheme);

// Apply theme
window.addEventListener('DOMContentLoaded', () => {
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-mode');
    }
});

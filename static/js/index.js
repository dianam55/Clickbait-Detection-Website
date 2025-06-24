/*
const textarea = document.querySelector('.headline');
const enButton = document.querySelector('.btn-language img[alt="english"]').parentElement;
const resultElement = document.querySelector('#result');

// Add event listener to the EN button
enButton.addEventListener('click', async () => {
    const text = textarea.value.trim();

    if (!text) {
        resultElement.textContent = 'Please enter a headline.';
        // Reset and trigger blink
        resultElement.classList.remove('blink');
        void resultElement.offsetWidth; // Force reflow to restart animation
        resultElement.classList.add('blink');
        setTimeout(() => resultElement.classList.remove('blink'), 600);
        return;
    }

    try {
        // Send text to the backend
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        const data = await response.json();

        if (data.error) {
            resultElement.textContent = `Error: ${data.error}`;
            // Reset and trigger blink
            resultElement.classList.remove('blink');
            void resultElement.offsetWidth; // Force reflow to restart animation
            resultElement.classList.add('blink');
            setTimeout(() => resultElement.classList.remove('blink'), 600);
            return;
        }

        // Update result text and classes based on prediction
        const prediction = data.prediction;
        resultElement.textContent = `Your headline is likely: ${prediction === 1 ? 'Clickbait' : 'Not Clickbait'}`;
        
        // Remove previous classes and add the correct one
        resultElement.classList.remove('clickbait', 'not-clickbait');
        resultElement.classList.add(prediction === 1 ? 'clickbait' : 'not-clickbait');
        
        // Reset and trigger blink animation
        resultElement.classList.remove('blink');
        void resultElement.offsetWidth; // Force reflow to restart animation
        resultElement.classList.add('blink');
        setTimeout(() => resultElement.classList.remove('blink'), 600);
    } catch (error) {
        resultElement.textContent = `Error: ${error.message}`;
        // Reset and trigger blink
        resultElement.classList.remove('blink');
        void resultElement.offsetWidth; // Force reflow to restart animation
        resultElement.classList.add('blink');
        setTimeout(() => resultElement.classList.remove('blink'), 600);
    }
});*/


const textarea = document.querySelector('.headline');
const enButton = document.querySelector('.btn-language img[alt="english"]').parentElement;
const roButton = document.querySelector('.btn-language img[alt="romanian"]').parentElement;
const huButton = document.querySelector('.btn-language img[alt="hungarian"]').parentElement;
const resultElement = document.querySelector('#result');

// Function to handle prediction requests
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
        resultElement.textContent = `Your headline is likely: ${prediction === 1 ? 'Clickbait' : 'Not Clickbait'}`;
        resultElement.classList.remove('clickbait', 'not-clickbait');
        resultElement.classList.add(prediction === 1 ? 'clickbait' : 'not-clickbait');
        resultElement.classList.remove('blink');
        void resultElement.offsetWidth;
        resultElement.classList.add('blink');
        setTimeout(() => resultElement.classList.remove('blink'), 600);
    } catch (error) {
        resultElement.textContent = `Error: ${error.message}`;
        resultElement.classList.remove('blink');
        void resultElement.offsetWidth;
        resultElement.classList.add('blink');
        setTimeout(() => resultElement.classList.remove('blink'), 600);
    }
}

// Event listeners to buttons
enButton.addEventListener('click', () => makePrediction(textarea.value.trim(), '/predict/en'));
roButton.addEventListener('click', () => makePrediction(textarea.value.trim(), '/predict/ro'));
huButton.addEventListener('click', () => makePrediction(textarea.value.trim(), '/predict/hu'));
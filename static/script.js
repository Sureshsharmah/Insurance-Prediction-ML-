document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = {
        age: document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        marital_status: document.getElementById('marital_status').value,
        salary: document.getElementById('salary').value,
        employment_type: document.getElementById('employment_type').value,
        region: document.getElementById('region').value,
        has_dependents: document.getElementById('has_dependents').value,
        tenure_years: document.getElementById('tenure_years').value
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        const resultDiv = document.getElementById('predictionResult');
        if (data.error) {
            resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        } else {
            const predictionText = data.prediction === 1 ? 
                '<span class="text-success">Likely to enroll</span>' : 
                '<span class="text-danger">Unlikely to enroll</span>';
            
            resultDiv.innerHTML = `
                <p>Prediction: ${predictionText}</p>
                <p>Probability: ${(data.probability * 100).toFixed(1)}%</p>
                ${data.shap_explanation ? `<img src="data:image/png;base64,${data.shap_explanation}" class="img-fluid">` : ''}
            `;
        }
    })
    .catch(error => {
        document.getElementById('predictionResult').innerHTML = 
            `<div class="alert alert-danger">Error: Failed to get prediction. Please try again.</div>`;
        console.error('Prediction error:', error);
    });
});
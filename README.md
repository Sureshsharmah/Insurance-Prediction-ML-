# 🧠 Insurance Enrollment Prediction

A professional-grade machine learning web application that predicts employee enrollment in voluntary insurance plans using demographic and employment data. The system leverages multiple machine learning classifiers and provides real-time, interpretable predictions through a user-friendly web interface.

---

## 🚀 Key Features

- ✅ **Multi-Model Architecture**: Supports Logistic Regression, XGBoost, Random Forest, Gradient Boosting, SVM, LightGBM, Neural Networks, and more.
- 🎯 **Real-Time Predictions**: Predict insurance enrollment likelihood instantly based on user input.
- 📊 **Data Visualization**: Built-in dashboards for understanding patterns and trends.
- 📈 **Model Explainability**: Integrated SHAP analysis to interpret model decisions.
- 🖥️ **Responsive UI**: Clean, mobile-friendly interface using HTML/CSS/JavaScript.
- 🔒 **Model Persistence**: Pretrained `.pkl` files for fast loading and inference.

---

## 🗂️ Project Structure

insurance-prediction/
├── data/
│ └── employee_data.csv
├── models/
│ ├── Logistic_Regression.pkl
│ ├── Random_Forest.pkl
│ ├── XGBoost.pkl
│ └── ... (more models)
├── static/
│ ├── style.css
│ └── script.js
├── templates/
│ └── index.html
├── app.py
├── requirements.txt
├── README.md
└── report.md


---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sureshsharmah/insurance-prediction.git
cd insurance-prediction


 Set up a virtual environment

python -m venv venv
source venv/bin/activate      

Install dependencies
pip install -r requirements.txt

Launch the Flask app
python app.py

📊 Model Overview
Each model was trained and evaluated using metrics such as:

Accuracy

Precision, Recall, F1 Score

ROC-AUC

The best-performing models are stored in the /models directory and used for real-time prediction.

📁 Dataset Details
Source: data/employee_data.csv

Target Variable: Insurance Enrollment Status (Yes/No)

Attributes: Age, Gender, Department, Tenure, Salary, Marital Status, etc.

🔍 Explainability with SHAP
The app integrates SHAP (SHapley Additive exPlanations) to provide feature-level insights for each prediction, promoting transparency and trust in model outputs.

🛠️ Future Enhancements
Add Streamlit-based interactive UI option

Integrate MongoDB or SQLite for data logging

Support for online learning and model re-training

Add Docker support for containerized deployment

🤝 Contributing
We welcome contributions from the community! To contribute:

Fork the repo

Create a new branch (git checkout -b feature/your-feature)

Commit your changes (git commit -am 'Add feature')

Push to the branch (git push origin feature/your-feature)

Create a new Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Developed By
Suresh Sharma
Aspiring Data Scientist | Machine Learning Enthusiast
LinkedIn = 'https://www.linkedin.com/in/suresh-sharma-90942b231/'
GitHub = 'https://github.com/Sureshsharmah'

import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

model = None
preprocessor = None
explainer = None
model_loaded = False

def load_data():
    try:
        df = pd.read_csv('data/employee_data.csv')
        df = df.dropna().drop_duplicates()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return generate_synthetic_data()

def generate_synthetic_data():
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'employee_id': np.arange(1, n_samples+1),
        'age': np.random.randint(22, 65, size=n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], size=n_samples, p=[0.48, 0.5, 0.02]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                         size=n_samples, p=[0.3, 0.5, 0.15, 0.05]),
        'salary': np.random.normal(70000, 20000, size=n_samples).astype(int),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Contract', 'Temporary'],
                                          size=n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'],
                                 size=n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
        'has_dependents': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.4, 0.6]),
        'tenure_years': np.random.randint(0, 30, size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    prob = (
        0.3 * (df['age'] / 65) + 
        0.2 * (df['salary'] / 150000) + 
        0.2 * (df['marital_status'] == 'Married') + 
        0.2 * (df['has_dependents'] == 'Yes') +     
        0.1 * (df['tenure_years'] / 30) + 
        np.random.normal(0, 0.1, size=n_samples))
    
    df['enrolled'] = (prob > 0.5).astype(int)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/employee_data.csv', index=False)
    return df

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True, kernel='rbf'),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150),
        "XGBoost": XGBClassifier(),
        "LightGBM": LGBMClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Neural Network": MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,))
    }
    
    results = []
    for name, model in models.items():
        try:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, y_proba)
            }
            results.append(metrics)
            
            joblib.dump(pipeline, f'models/{name.replace(" ", "_")}.pkl')
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
    
    return pd.DataFrame(results).sort_values('ROC AUC', ascending=False)

def train_model():
    global model, preprocessor, explainer
    
    df = load_data()
    if df is None:
        print("Failed to load or generate data")
        return False

    X = df.drop(['employee_id', 'enrolled'], axis=1)
    y = df['enrolled']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ['age', 'salary', 'tenure_years']
    categorical_features = ['gender', 'marital_status', 'employment_type', 'region', 'has_dependents']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    print("Model Performance Comparison:\n", model_results)

    best_model_name = model_results.iloc[0]['Model']
    model = joblib.load(f'models/{best_model_name.replace(" ", "_")}.pkl')
    
    try:
        if hasattr(model.named_steps['classifier'], 'predict_proba'):
            explainer = shap.Explainer(model.named_steps['classifier'])
        else:
            explainer = None
    except Exception as e:
        print(f"Could not create SHAP explainer: {e}")
        explainer = None

    print(f"\nBest model selected: {best_model_name}")
    
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        print("\nFeature Importances:")
        importances = model.named_steps['classifier'].feature_importances_
        features = numeric_features + categorical_features
        print(pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False))
    
    return True

def generate_visualizations():
    df = load_data()
    if df is None:
        return {}

    visualizations = {}

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='enrolled', kde=True, bins=30)
    plt.title('Age Distribution by Enrollment Status')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visualizations['age_dist'] = base64.b64encode(buf.read()).decode('ascii')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='enrolled', y='salary')
    plt.title('Salary Distribution by Enrollment Status')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visualizations['salary_dist'] = base64.b64encode(buf.read()).decode('ascii')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='region', hue='enrolled')
    plt.title('Enrollment by Region')
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visualizations['region_dist'] = base64.b64encode(buf.read()).decode('ascii')
    plt.close()

    return visualizations

@app.before_request
def load_model():
    global model, preprocessor, explainer, model_loaded
    
    if not model_loaded:
        try:
            if not os.path.exists('models'):
                os.makedirs('models', exist_ok=True)
                
            if not os.listdir('models'): 
                print("No models found. Training new models...")
                if not train_model():
                    print("Failed to train models")
                    return
            else:
                print("Loading existing best model...")
                try:
                    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join('models', x)))
                    model = joblib.load(os.path.join('models', model_files[-1]))
                    preprocessor = model.named_steps['preprocessor']
                    
                    try:
                        explainer = shap.Explainer(model.named_steps['classifier'])
                    except Exception as e:
                        print(f"Could not create SHAP explainer: {e}")
                        explainer = None
                except Exception as e:
                    print(f"Failed to load model: {e}")
                    if not train_model():
                        print("Failed to train model")
                        return
            
            model_loaded = True
            print("Model initialization complete")
        except Exception as e:
            print(f"Model initialization failed: {str(e)}")

@app.route('/')
def home():
    visualizations = generate_visualizations()
    return render_template('index.html', visualizations=visualizations)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        input_data = pd.DataFrame([{
            'age': int(data.get('age', 0)),
            'gender': data.get('gender', ''),
            'marital_status': data.get('marital_status', ''),
            'salary': int(data.get('salary', 0)),
            'employment_type': data.get('employment_type', ''),
            'region': data.get('region', ''),
            'has_dependents': data.get('has_dependents', ''),
            'tenure_years': int(data.get('tenure_years', 0))
        }])

        proba = model.predict_proba(input_data)[0][1]
        threshold = 0.4 
        prediction = int(proba >= threshold)

        shap_image = None
        if explainer is not None:
            try:
                transformed_data = preprocessor.transform(input_data)
                shap_values = explainer(transformed_data)
                plt.figure()
                shap.plots.bar(shap_values[0], show=False)
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                shap_image = base64.b64encode(buf.read()).decode('ascii')
                plt.close()
            except Exception as e:
                print(f"SHAP explanation failed: {e}")

        return jsonify({
            'prediction': prediction,
            'probability': float(proba),
            'shap_explanation': shap_image
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
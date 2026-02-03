<div align="center">
  <img src="images/raisin-logo.svg" alt="Raisin Logo" width="150"/>
  
  # 🍇 Raisin Classification with Logistic Regression
  
  A machine learning project for classifying raisins using logistic regression implemented from scratch with NumPy
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![NumPy](https://img.shields.io/badge/NumPy-1.19+-orange.svg)](https://numpy.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
  
</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Jupyter Notebook](#jupyter-notebook)
  - [API Deployment](#api-deployment)
- [Model Explainability](#model-explainability)
- [Docker Deployment](#docker-deployment)
- [Dataset](#dataset)
- [Results](#results)

---

## 🎯 Overview

This project demonstrates a complete machine learning pipeline for **raisin classification**. The implementation uses pure NumPy (no scikit-learn) to build a logistic regression classifier from scratch.

The project includes:
- ✅ Custom logistic regression implementation using only NumPy
- ✅ Interactive Jupyter notebook for exploration and training
- ✅ REST API built with FastAPI for serving predictions
- ✅ Docker containerization for easy deployment
- ✅ SHAP analysis for model explainability and feature importance

---

## ✨ Features

### 🧮 Pure NumPy Implementation
- Custom gradient descent optimizer
- Binary cross-entropy loss function
- Sigmoid activation
- No ML framework dependencies (except for SHAP analysis)

### 📊 Interactive Notebook
- Data exploration and visualization
- Step-by-step model training
- Performance metrics and evaluation
- Confusion matrix

### 🚀 Production-Ready API
- FastAPI-based REST endpoints
- Input validation with Pydantic
- Health check endpoints
- Prediction endpoint with probability scores

### 🐳 Docker Support
- Multi-stage Docker build
- Optimized image size
- Easy deployment to any cloud platform
- Docker Compose support

### 🔍 Model Explainability
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance visualization
- Individual prediction explanations
- Global model interpretation

---

## 📁 Project Structure

```
fuzzyfun/
├── images/                    # Images and assets for README
│   └── raisin-logo.svg       # Project logo
├── notebooks/                 # Jupyter notebooks
│   └── logistic_regression.ipynb  # Main analysis notebook
├── src/                       # Source code
│   ├── model.py              # Logistic regression implementation
│   ├── preprocessing.py      # Data preprocessing utilities
│   └── explainability.py     # SHAP analysis functions
├── api/                       # FastAPI application
│   ├── main.py               # API entry point
│   ├── models.py             # Pydantic models
│   └── predict.py            # Prediction logic
├── data/                      # Dataset directory
│   └── raisin_dataset.csv    # Raisin classification dataset
├── models/                    # Saved model artifacts
│   └── logistic_model.pkl    # Trained model
├── tests/                     # Unit tests
│   └── test_model.py         # Model tests
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore rules
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/daftlad-1983/fuzzyfun.git
   cd fuzzyfun
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Jupyter Notebook

The interactive notebook provides a complete walkthrough of the logistic regression implementation:

```bash
jupyter notebook notebooks/logreg.ipynb
```

The notebook covers:
1. **Data Loading & Exploration** - Understanding the raisin dataset
2. **Data Prep** - Preparing features for classification
3. **Model Implementation** - Building logistic regression from scratch
4. **Training** - Gradient descent optimization
5. **Evaluation** - confusion matrix
6. **SHAP Analysis** - Interpreting model predictions

### API Deployment

#### Run Locally

Start the FastAPI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive API documentation at `http://localhost:8000/docs`

#### Example API Request

```python

import requests

data_dict = x_df.to_dict(orient='list')

response = requests.post("http://127.0.0.1/predict/", json=data_dict).json()

print('\nConfusion matrix\n')
print(confusion_matrix(y_numpy , np.array(response['predictions'])>0.5, normalize='true' ))

```

#### Available Endpoints

- `GET /` - Welcome message
- `POST /predict` - Make predictions

## 🐳 Docker Deployment

### Build the Docker Image

```bash
docker build -t raisin-classifier .
```

### Run the Container

```bash
docker run -p 8000:8000 raisin-classifier
```

### Using Docker Compose

```bash
docker-compose up
```

The API will be available at `http://localhost:8000`

---

## 📊 Dataset

The project uses a **Raisin Dataset** containing measurements of two varieties of raisins:

### Features
1. **Area** - Number of pixels within the boundaries of the raisin
2. **Perimeter** - Circumference measurement
3. **Major Axis Length** - Length of the main axis
4. **Minor Axis Length** - Length of the perpendicular axis
5. **Eccentricity** - Measure of the elongation
6. **Convex Area** - Number of pixels in the convex hull
7. **Extent** - Ratio of the raisin area to the bounding box area

### Target Variable
- **Class** - Binary classification (Kecimen or Besni variety)

### Data Source
The dataset is publicly available and contains high-quality measurements obtained through computer vision techniques.

---

## 📈 Results

### Model Performance

Confusion matrix with the customer classifier

<img src="images/cusom cm.png" alt="cm1" width="500"/>

Confusion matrix with the sklearn classifier

<img src="images/sk cm.png" alt="cm2" width="500"/>


---

## 🔍 Model Explainability

This project uses **SHAP (SHapley Additive exPlanations)** to explain model predictions:

SHAP helps answer:
- 🎯 Which features most influence the model's decisions?
- 📊 How does each feature contribute to a specific prediction?

### Feature Importance

As we can see, perimeter is  the most important feature. So the lower the perimeter, the more likely it is to be 
class 1, kecimin

<img src="images/shap.png" alt="shap graph" width="600"/>

---

### Key Findings

- 🎯 **Most Important Features**: Perimeter strongest predictor
- 📊 **Model Complexity**: My GD version of the model works ok
- ⚡ **API deployment**: works ok. Could be deployed on ECS aws for scaling

---



<div align="center">

</div>

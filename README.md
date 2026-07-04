# 🍷 End-to-End Wine Quality Prediction

A production-ready machine learning project that predicts wine quality based on physicochemical properties. This repository demonstrates best practices in ML engineering with modular pipelines, YAML-based configuration management, and integration with MLflow for experiment tracking.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Configuration](#configuration)
- [Model Evaluation](#model-evaluation)
- [Web Interface](#web-interface)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete machine learning workflow for predicting wine quality on a scale of 0-10 based on 11 physicochemical properties. The implementation follows ML engineering best practices with modular components, configuration-driven architecture, and experiment tracking capabilities.

### Key Highlights

- **11 Input Features**: Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol content
- **Target Variable**: Wine quality score (integer 0-10)
- **Model**: ElasticNet regression with configurable hyperparameters
- **Experiment Tracking**: MLflow integration for metrics and model logging
- **Web UI**: Flask-based prediction interface for end-users

## ✨ Features

- ✅ **Modular Pipeline Architecture**: Each stage (ingestion, validation, transformation, training, evaluation) is independently testable
- ✅ **Configuration Management**: YAML-based configs for easy parameter tuning without code changes
- ✅ **Schema Validation**: Data validation against predefined schema to ensure data integrity
- ✅ **MLflow Integration**: Experiment tracking with metrics logging and model registry
- ✅ **Train-Test Split**: Automatic data splitting for training and evaluation
- ✅ **Web Interface**: Flask application for interactive predictions
- ✅ **Logging**: Comprehensive logging for debugging and monitoring
- ✅ **Error Handling**: Robust exception handling throughout the pipeline

## 🏗️ Project Architecture

```
├── src/datascience/              # Main source code
│   ├── components/               # Core ML components
│   │   ├── data_ingestion.py    # Download and extract data
│   │   ├── data_validation.py   # Schema validation
│   │   ├── data_transformation.py # Train-test split
│   │   ├── model_trainer.py     # Model training with ElasticNet
│   │   └── model_evaluation.py  # Metrics calculation & MLflow logging
│   ├── config/                   # Configuration management
│   │   └── configuration.py      # Config loader & entity builders
│   ├── constants/                # Constants definition
│   ├── entity/                   # Data models & configs
│   ├── pipeline/                 # Pipeline orchestration
│   │   ├── data_ingestion_pipeline.py
│   │   ├── data_validation_pipeline.py
│   │   ├── data_transformation_pipeline.py
│   │   ├── model_trainer_pipeline.py
│   │   ├── model_evaluation_pipeline.py
│   │   └── prediction_pipeline.py
│   └── utils/                    # Utility functions
├── research/                     # Jupyter notebooks for experimentation
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_validation.ipynb
│   ├── 3_data_transformation.ipynb
│   ├── 4_model_trainer.ipynb
│   ├── 5_model_evaluation.ipynb
│   └── research.ipynb
├── templates/                    # Flask HTML templates
│   ├── index.html               # Prediction form UI
│   └── results.html             # Results display
├── config/                       # Configuration files (YAML)
├── main.py                      # End-to-end pipeline orchestrator
├── app.py                       # Flask web application
├── params.yaml                  # Model hyperparameters
├── schema.yaml                  # Data schema definition
├── requirements.txt             # Python dependencies
└── LICENSE                      # GPL v3.0 License
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, numpy |
| **Experiment Tracking** | MLflow |
| **Web Framework** | Flask |
| **Configuration** | PyYAML |
| **Data Format** | CSV |
| **Model Serialization** | joblib |

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/humayun-mhk/End-to-end-wine-quality-prediction.git
   cd End-to-end-wine-quality-prediction
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

## 🚀 Usage

### Running the Complete Pipeline

Execute the end-to-end ML pipeline:

```bash
python main.py
```

This will:
1. Download and extract the wine quality dataset
2. Validate data against the defined schema
3. Split data into training and test sets
4. Train an ElasticNet model with configured hyperparameters
5. Evaluate the model and log metrics to MLflow

### Training Only

```bash
python app.py
# Then navigate to http://localhost:5000/train
```

### Making Predictions via Web UI

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Fill in the physicochemical properties form and submit to get quality predictions

### Making Predictions Programmatically

```python
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline
import numpy as np

# Create sample data (11 features)
data = np.array([[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]])

# Make prediction
pipeline = PredictionPipeline()
prediction = pipeline.predict(data)
print(f"Predicted wine quality: {prediction}")
```

## 📊 ML Pipeline

### 1. Data Ingestion (`data_ingestion.py`)
- Downloads wine quality dataset from a remote URL
- Extracts compressed data files
- Organizes data in the artifacts directory

### 2. Data Validation (`data_validation.py`)
- Validates all columns exist according to `schema.yaml`
- Ensures data integrity before processing
- Logs validation status to status file

### 3. Data Transformation (`data_transformation.py`)
- Performs 75-25 train-test split
- Saves training and test datasets separately
- Logs dataset shapes for verification

### 4. Model Training (`model_trainer.py`)
- Trains ElasticNet regression model
- Uses hyperparameters from `params.yaml`
- Serializes trained model using joblib
- Default hyperparameters: `alpha=0.2, l1_ratio=0.1`

### 5. Model Evaluation (`model_evaluation.py`)
- Calculates evaluation metrics:
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **R² Score** (Coefficient of Determination)
- Logs metrics and parameters to MLflow
- Supports model registry for production deployment

## ⚙️ Configuration

### `params.yaml` - Model Hyperparameters
```yaml
ElasticNet:
  alpha: 0.2          # Regularization strength
  l1_ratio: 0.1       # L1 ratio (0=Ridge, 1=Lasso)
```

### `schema.yaml` - Data Schema
Defines the expected columns and their data types:
```yaml
COLUMNS:
  fixed acidity: float64
  volatile acidity: float64
  citric acid: float64
  residual sugar: float64
  chlorides: float64
  free sulfur dioxide: float64
  total sulfur dioxide: float64
  density: float64
  pH: float64
  sulphates: float64
  alcohol: float64
  quality: int64

TARGET_COLUMN:
  name: quality
```

## 📈 Model Evaluation

The model's performance is tracked using standard regression metrics:

- **RMSE**: Measures average prediction error magnitude
- **MAE**: Measures average absolute deviation from true values
- **R² Score**: Indicates proportion of variance explained (0-1 range)

Metrics are automatically logged to MLflow and saved to `metrics.json` for review.

## 🌐 Web Interface

### Flask Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Display prediction form |
| `/train` | GET | Trigger full pipeline training |
| `/predict` | POST, GET | Submit features and get quality prediction |

### Input Form Fields
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

## 📝 Logging

The project includes comprehensive logging to both file and console:
- Log file location: `logs/logging.log`
- Log format: `[timestamp: level: module: message]`
- Useful for debugging and monitoring pipeline execution

## 🔍 Experiment Tracking with MLflow

To view MLflow tracking results:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` (MLflow UI runs on port 5000 by default).

### Available Metrics
- Hyperparameters (alpha, l1_ratio)
- RMSE, MAE, R² scores
- Model artifacts and registry

## 🚨 Error Handling

The pipeline includes robust error handling:
- Custom exceptions for each stage
- Detailed error logging
- Graceful failure with informative messages
- Validation of data schema before processing

## 📚 Research Notebooks

The `research/` directory contains Jupyter notebooks documenting the experimental process:
- Individual experimentation for each pipeline stage
- Data exploration and analysis
- Model development and tuning

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Technologies

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

## 📞 Contact & Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Built with ❤️ by Humayun MHK**

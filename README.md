# Codesoft MLOps Project

## Overview
Codesoft MLOps Project is an end-to-end machine learning system that demonstrates how a model evolves from experimentation to a fully automated, monitored production deployment on AWS.

The project covers the entire ML lifecycle: problem definition, experimentation, business-aware model selection, automated retraining pipelines, CI/CD, cloud deployment, and observability.

Although built for learning, the system closely follows production-grade MLOps patterns.

---

## Problem Statement
The project starts with a clearly defined problem statement where the goal is not just to build an accurate model, but to design a maintainable and deployable ML system.

**Key considerations:**
- Model performance alone is not sufficient
- Business requirements influence thresholding and model promotion
- The system must support retraining, versioning, and monitoring

---

## End-to-End Workflow

### 1. Experimentation & Model Exploration
- Multiple models are evaluated during the experimentation phase
- Hyperparameter tuning and metric comparisons are performed
- MLflow integrated with DagsHub is used to track:
  - Parameters
  - Metrics
  - Artifacts
  - Experiment history

This enables reproducibility and transparent comparison across experiments.

---

### 2. Business-Driven Model Selection
Instead of selecting models purely on accuracy:
- The best model and hyperparameters are chosen based on business needs
- Decision thresholds are tuned to balance metrics such as accuracy and F1-score
- The selected configuration becomes the baseline for the training pipeline

This reflects real-world ML systems where engineering and business constraints matter.

---

### 3. Automated Training & Retraining Pipeline
A fully automated pipeline operationalizes the selected experiment.

**Pipeline stages:**
1. **Data Ingestion**  
   Data is ingested directly from MongoDB Atlas

2. **Data Preprocessing**  
   Cleaning and transformation of ingested data

3. **Feature Engineering**  
   Feature extraction on preprocessed data

4. **Model Building**  
   Model training using the selected algorithm and tuned hyperparameters

5. **Model Evaluation**  
   Model performance is evaluated  
   Metrics are logged to MLflow  
   Trained model and supporting artifacts (vectorizer/tokenizer) are saved

6. **Model Registration**  
   The newly trained model is compared with the currently deployed model  
   If performance improves (accuracy/F1), the model is registered and promoted to production

All versions are stored in a centralized model registry.

---

### 4. Application Layer (Model Serving)
The project includes a Flask-based inference application that serves the production model.

**Application features:**
- Loads the production-registered model from the registry
- Accepts user input via API endpoints
- Returns real-time predictions
- Supports multi-model serving
- Exposes application metrics for monitoring

This layer simulates how ML models are consumed in real systems.

---

### 5. CI/CD with GitHub Actions
A complete CI/CD pipeline automates training, testing, and deployment.

**CI/CD steps:**
1. Set up Python environment
2. Install project dependencies
3. Execute the training pipeline (`dvc repro`)
4. Run Flask application tests
5. Authenticate with AWS ECR
6. Build and push Docker image
7. Deploy the application to AWS EKS

This ensures deployments are reproducible, automated, and consistent.

---

### 6. AWS Infrastructure
The system is deployed using multiple AWS services:
- **S3**: Storage backend for DVC-managed data and artifacts
- **ECR**: Docker image repository
- **EKS**: Kubernetes cluster for application deployment
- **EC2 instances**:
  - Prometheus server
  - Grafana dashboards

---

### 7. Deployment & Monitoring

#### Monitoring & Observability
Once deployed, the system is continuously monitored using:
- **Prometheus**
  - Collects application and runtime metrics
- **Grafana**
  - Visualizes metrics through dashboards

**Tracked metrics include:**
- Request count
- Latency
- Application health

This enables visibility into model behavior after deployment, closing the MLOps loop.

---

## Why This Project Matters
This project demonstrates:
- Ownership of the entire ML lifecycle
- Practical use of MLflow, DagsHub, DVC, CI/CD, and Kubernetes
- Business-aware model selection and promotion
- Cloud-native deployment and monitoring
- Understanding that ML systems must be observable, reproducible, and maintainable

It goes beyond notebooks and shows how ML behaves in production.

---

## Future Improvements
- Automated data and model drift detection
- Alerting on monitoring metrics
- Canary or blue-green deployments
- Automated rollback strategies
- Enhanced API documentation
- Feature store integration

---

## Disclaimer
This project was built primarily for learning and skill development, while closely following industry-standard MLOps practices.

---

## Application Deployment
<img src="https://github.com/user-attachments/assets/f0b52a75-7e7d-4e8e-8304-ab3ac3287229" width="700" />
<img src="https://github.com/user-attachments/assets/b7835647-3b16-4fef-8566-6f88494974af" width="700" />
<img src="https://github.com/user-attachments/assets/ee36520f-60c9-4c59-b430-75fff51736d3" width="700" />

---

## Monitoring (Grafana)
<img src="https://github.com/user-attachments/assets/b7391395-974d-4a2d-92c1-08d22c5e51ff" width="600" />

---

## Monitoring (Prometheus)
<img src="https://github.com/user-attachments/assets/884859d1-8edd-4c50-a718-8f0d7e8f084c" width="600" />

---

## CI/CD Pipeline
<img src="https://github.com/user-attachments/assets/c5b015a7-85ab-48d6-bc55-3dff94e964e5" width="700" />

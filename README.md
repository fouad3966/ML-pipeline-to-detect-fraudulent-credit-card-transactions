# 💳 Credit Card Fraud Detection System

## 🔗 Quick Access Links

| Resource | Link | Description |
|----------|------|-------------|
| 📊 **Live Web Application** | [🚀 Try the App](https://gtc-fraud-detection-9mtefvddmsmjyzeictpraw.streamlit.app/) | Interactive fraud detection system |
| 📋 **Project Presentation** | [📑 View Slides](https://credit-card-fraud-detect-9vsoerj.gamma.site/) | Detailed project overview & results |
| 💻 **Source Code** | [GitHub Repository](https://github.com/FarahYehia824/GTC-Fraud-Detection) | Complete project codebase |
| 📊 **Dataset** | [Synthetic Credit Card Transactions](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data) | Original dataset source |

---
## 🎥 Project Demo
**Demo Video:** [Watch the complete project demonstration](https://drive.google.com/file/d/1b5iZRU0zhRLcPxt_ifUXNzQ1urUydBF_/view?usp=sharing) (4 minutes)

## 📋 Project Overview

This project implements a comprehensive machine learning pipeline to detect fraudulent credit card transactions using advanced data science techniques. The system analyzes transaction patterns, customer behavior, geographic data, and temporal features to identify potentially fraudulent activities with high accuracy.

### 🎯 Business Problem

Credit card fraud causes billions of dollars in losses annually and undermines customer trust in financial institutions. Traditional rule-based systems often have high false positive rates, leading to legitimate transactions being declined. This project aims to build an intelligent fraud detection system that:

- Accurately identifies fraudulent transactions with high precision
- Minimizes false positives (legitimate transactions flagged as fraud)  
- Provides real-time fraud scoring capabilities
- Delivers interpretable results for fraud analysts
- Maintains customer satisfaction while preventing financial losses

## 🏆 Key Results

### Model Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **99.31%** | **41.84%** | **96.24%** | **58.32%** | **99.88%** |
| Random Forest | 99.82% | 97.52% | 65.20% | 78.15% | 98.52% |
| Logistic Regression | 87.32% | 3.22% | 83.11% | 6.20% | 93.54% |

**Selected Model: XGBoost** - Optimal balance of high recall (catches 96% of fraud) with reasonable precision and excellent ROC-AUC score.

## 📊 Dataset Description

**Source:** Synthetic credit card transactions dataset  
**Size:** 1,852,394 transactions  
**Time Period:** Transaction data spanning multiple months  
**Target Variable:** `is_fraud` (binary: 0 = legitimate, 1 = fraudulent)

### Dataset Characteristics:
- **Legitimate Transactions:** ~99.35% (highly imbalanced dataset)
- **Fraudulent Transactions:** ~0.65%
- **Features:** 23+ original features plus 15+ engineered features
- **Geographic Coverage:** Multiple US states and cities
- **Transaction Categories:** 26+ merchant categories

### Key Features:
- **Transaction Details:** Amount, timestamp, transaction number
- **Customer Demographics:** Age, gender, location, job
- **Merchant Information:** Name, category, geographic location
- **Geographic Data:** Customer and merchant coordinates with distance calculations
- **Temporal Patterns:** Hour, day of week, business hours indicators

## 🔧 Technical Implementation

### 1. Data Preprocessing Pipeline

#### Missing Values Handling
- **Geographic data:** Filled with median values for coordinates
- **Categorical data:** Filled with mode (most frequent values)
- **Target variable:** Missing fraud labels assumed as legitimate transactions
- **Temporal data:** Forward/backward fill for datetime columns

#### Outlier Treatment
- **Geographic outliers:** Capped using IQR method (7,063 merchant lat, 59,972 merchant long)
- **Amount outliers:** Removed only clearly invalid values (negative/zero amounts)
- **Temporal outliers:** Minimal adjustment 
- **Fraud cases preserved:** 9,651 fraud cases kept as legitimate targets

#### Data Type Corrections
- **DateTime conversion:** Transaction timestamps and date of birth
- **Numeric conversion:** Amount, coordinates, population data
- **Categorical encoding:** Label encoding for high-cardinality variables

### 2. Advanced Feature Engineering

#### Time-Based Features (7 features)
```python
# Risk-based temporal features
'hour', 'day_of_week', 'month', 'is_weekend'
'is_night_transaction'     # 10PM-6AM (3x higher fraud rate)
'is_business_hours'        # 9AM-5PM business hours
'is_high_risk_hours'       # 12AM-3AM (highest risk period)
```

#### Geographic Features (3 features)
```python
# Location-based risk indicators
'distance_km'              # Haversine distance calculation
'is_far_transaction'       # >100km from home (suspicious)
'is_very_far_transaction'  # >500km (very suspicious)
```

#### Customer Demographics (2 features)
```python
'customer_age'             # Calculated from date of birth
'age_risk_category'        # Age groups with different risk profiles
```

#### Transaction Amount Features (5 features)
```python
'log_amount'               # Log transformation for skewed amounts
'is_high_amount'           # >95th percentile transactions
'is_low_amount'            # <5th percentile transactions  
'is_round_amount'          # Round dollar amounts (e.g., $100.00)
'amt_per_pop'              # Amount relative to city population
```

#### Behavioral Pattern Features (4 features)
```python
'transactions_per_hour'    # Transaction velocity per card
'is_high_velocity'         # Multiple transactions in short timeframe
'category_risk_score'      # Fraud rate by merchant category
'is_high_risk_category'    # High-risk merchant categories
```

#### Encoded Categorical Features (7 features)
```python
# Label-encoded categorical variables
'merchant_encoded', 'category_encoded', 'state_encoded'
'job_encoded', 'gender_encoded', 'city_encoded', 'street_encoded'
```

**Total Engineered Features:** 28 features created from 23 original features

### 3. Model Training & Selection

#### Class Imbalance Handling
- **XGBoost:** `scale_pos_weight` parameter for automatic balancing
- **Random Forest:** `class_weight='balanced'` parameter
- **Logistic Regression:** Manual class weights optimization

#### Model Optimization
- **Hyperparameter tuning** for Logistic Regression (12 configurations tested)
- **Feature correlation analysis** with automatic removal of correlated features (>0.9)
- **Cross-validation** for model stability assessment

#### Performance Evaluation Metrics
- **Precision:** Minimize false alarms to customers
- **Recall:** Maximize fraud detection rate  
- **F1-Score:** Balanced precision-recall measure
- **ROC-AUC:** Overall model discrimination ability
- **Business metrics:** False positive rate, fraud detection rate

### 4. Model Deployment

#### Streamlit Web Application Features
- **Interactive fraud prediction** with real-time scoring
- **Risk factor analysis** with detailed explanations
- **Feature importance visualization** using Plotly
- **User-friendly interface** for transaction input
- **Professional dashboard** with metrics and charts

#### Technical Specifications
- **Model size optimization:** XGBoost model compressed to 0.4MB
- **Real-time prediction:** Sub-second response time
- **Scalable architecture:** Cloud-ready deployment
- **Error handling:** Robust input validation and error recovery

## 📈 Business Impact & Insights

### Key Fraud Indicators Discovered
1. **Time-based patterns:**
   - Night transactions (10PM-6AM) have 3x higher fraud rate
   - Weekend transactions show elevated risk
   - Early morning hours (12AM-3AM) are highest risk

2. **Geographic patterns:**
   - Transactions >100km from home are 5x more suspicious
   - Certain merchant categories have significantly higher fraud rates
   - Cross-state transactions require additional scrutiny

3. **Amount patterns:**
   - Round amounts ($100, $500) are more likely fraudulent
   - Very high amounts (>95th percentile) require verification
   - Amount relative to city population is a strong indicator

4. **Velocity patterns:**
   - Multiple transactions per hour from same card are highly suspicious
   - High-velocity patterns are strongest fraud predictors

### Business Recommendations
1. **Real-time alerts** for night and high-velocity transactions
2. **Geographic verification** for distant transactions
3. **Dynamic transaction limits** based on risk scores
4. **Enhanced authentication** for high-risk scenarios

## 📁 Project Structure

```
GTC-Fraud-Detection/
│
├── 📊 Data Files
│   ├── fraudTrain.csv                    # Original training dataset
│   ├── fraudTest.csv                     # Original test dataset  
│   
│
├── 📓 Notebooks
│   ├── Credit_Card_Fraud.ipynb           # Main analysis notebook
│   └── Copy_of_Credit_Card_Fraud.py      # Python script version
│
├── 🤖 Model Files
│   ├── fraud_detection_model.pkl         # Trained XGBoost model (0.4MB)
│   ├── robust_scaler.pkl                 # Feature scaler
│   ├── label_encoders.pkl                # Categorical encoders
│   ├── feature_columns.pkl               # Feature names list
│   └── model_metadata.pkl                # Model information
│
├── 🌐 Deployment Files  
│   ├── app.py                           # Streamlit web application
│   ├── requirements.txt                 # Python dependencies
│   └── README.md                        # Project documentation
│
└── 📋 Documentation  
    └── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn plotly xgboost joblib
```

### Local Development
```bash
# Clone the repository
git clone https://github.com/FarahYehia824/GTC-Fraud-Detection.git
cd GTC-Fraud-Detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Cloud Deployment
The application is deployed on Streamlit Cloud with automatic updates from the GitHub repository.

## 🔍 Usage Examples

### Web Application
1. Access the live web application using the link above
2. Input transaction details (amount, time, location, etc.)
3. Get instant fraud risk assessment with explanations
4. View detailed risk factor analysis and recommendations

### API Integration (Future Enhancement)
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('robust_scaler.pkl')

# Make prediction
features = prepare_transaction_features(transaction_data)
scaled_features = scaler.transform([features])
fraud_probability = model.predict_proba(scaled_features)[0][1]
```

## 📊 Performance Metrics Detail

### Confusion Matrix Analysis
- **True Negatives:** 83,147 (legitimate correctly identified)
- **False Positives:** 197 (legitimate flagged as fraud) - 0.24% false alarm rate
- **False Negatives:** 32 (fraud missed) - 3.76% of actual fraud
- **True Positives:** 817 (fraud correctly identified) - 96.24% detection rate

### Business KPIs
- **Customer Impact:** Only 0.24% of legitimate customers affected
- **Fraud Prevention:** 96.24% of fraudulent transactions detected
- **Cost Savings:** Significant reduction in fraud losses vs false positive costs
- **Processing Speed:** Real-time prediction capability

## 🔮 Future Enhancements

### Technical Improvements
- **Deep Learning Models:** Neural networks for complex pattern detection
- **Ensemble Methods:** Combining multiple models for improved accuracy
- **Real-time Streaming:** Apache Kafka for live transaction processing
- **AutoML Pipeline:** Automated model retraining and deployment

### Feature Enhancements
- **Network Analysis:** Merchant connection patterns
- **Time Series Features:** Historical spending behavior analysis  
- **External Data:** Weather, events, economic indicators
- **Behavioral Biometrics:** Typing patterns, device fingerprinting

### Business Integration
- **Risk-based Authentication:** Dynamic security based on fraud scores
- **Merchant Risk Scoring:** Real-time merchant reputation system
- **Customer Communication:** Intelligent fraud alerts and notifications
- **Regulatory Compliance:** GDPR, PCI-DSS compliance features



### Development Guidelines
- Follow PEP 8 Python style guidelines
- Add unit tests for new features
- Update documentation for any changes
- Ensure model performance is not degraded

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset Provider:** Synthetic credit card transaction dataset creators
- **Open Source Community:** Scikit-learn, XGBoost, Streamlit contributors
- **Academic Research:** Fraud detection literature and best practices
- **Industry Experts:** Financial fraud prevention professionals

---

**⚠️ Important Disclaimer:** This project uses synthetic data for educational and demonstration purposes. In production fraud detection systems, additional security measures, privacy protections, and regulatory compliance requirements must be implemented. The models and results shown here are for academic and portfolio demonstration only.

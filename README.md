# Accident Severity Prediction

This project aims to analyze and predict the severity of traffic accidents in the United States using machine learning. The dataset contains various features related to traffic accidents, such as location, weather conditions, and traffic-related attributes. The goal is to build a predictive model that can classify accident severity based on these features.

## Project Structure

```
Accident_Severity_Prediction/
├── data/
│   ├── US_Accidents_Dataset.csv        # Raw dataset
├── notebooks/
│   ├── ML_Task.ipynb                  # Jupyter Notebook for analysis and model training
├── scripts/
│   ├── data_cleaning.py               # Data cleaning and preprocessing script
│   ├── feature_engineering.py         # Feature engineering script
│   ├── model_training.py              # Model training script
├── requirements.txt                   # Python dependencies
```
# Install Requirements

You can install the required Python packages with the following command:
```
pip install -r requirements.txt
```
# Project Workflow
1. Data Cleaning
  The first step involves cleaning the data. This includes:
  * Removing columns with excessive missing values.
  * Imputing missing numeric values with the median and categorical values with the mode.
  Run the following script for data cleaning:
  ```
  python scripts/data_cleaning.py
  ```
  The cleaned dataset will be saved as US_Accidents_Cleaned.csv in the data/ folder.
  
2. Feature Engineering
  The next step is feature engineering, where we:
  * Convert categorical features to numeric using label encoding.
  * Select relevant features for prediction.
  Run the following script for feature engineering:
  ```
  python scripts/feature_engineering.py
  ```
  The processed dataset will be saved as US_Accidents_Features.csv in the data/ folder.
  
3. Model Training
  We use a Random Forest Classifier to predict accident severity based on the processed features. The data is split into training, validation, and test sets.
  Run the following script to train the model:
  ```
  python scripts/model_training.py
  ```

4. Exploratory Data Analysis (EDA)
  Use the Jupyter notebook notebooks/ML_Task.ipynb for:
  * Data exploration and visualization.
  * Examining correlations between features.
  * Understanding the distribution of accident severity.
    
5. Model Evaluation
  After training the model, it will be evaluated using accuracy and classification reports based on the validation and test sets.

# Recommendations
1. Incorporate weather conditions as important features for severity prediction.
2. Improve the dataset quality by addressing missing data more robustly.
3. Use advanced models like XGBoost or Neural Networks for better accuracy.
4. Conduct hyperparameter tuning for improved model performance.
5. Explore geographic clustering of accidents to develop location-specific strategies.

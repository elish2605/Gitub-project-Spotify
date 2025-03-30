# Table of Contents

I. 🎧 Spotify Popularity Prediction (Regression Phase)
   - 📌 Project Overview
   - 🧭 Project Stages Summary
   - 📊 Results
   - ⚠️ Limitations
   - 🧾 Conclusion

II. 🎯 Spotify Popularity Prediction – Classification Phase
   - 📌 Project Overview
   - ✅ Target Definition
   - 🧭 Project Stages Summary 
   - 🔍 Next Steps
   - 📊 Results
   - ⚠️ Limitations
   - 🧾 Conclusion

III. 🔮 Recommendations and Future Work

Overall Conclusion

---
# I. 🎧 Spotify Popularity Prediction (Regression Phase)

## 📌 Project Overview

### 🔍 What are we trying to find out?
The main goal of this phase of the project was to **predict the popularity score of a song** on Spotify using a supervised regression model. We aimed to determine whether audio features and metadata of songs could accurately estimate how popular a song would be on a scale from 0 to 100.

### 📚 What do we already know?
We had access to a structured dataset of Spotify tracks containing:
- Audio features (`danceability`, `energy`, `valence`, `tempo`, etc.)
- Metadata (`duration_ms`, `explicit`, `release_year`, etc.)
- Target variable: `popularity` (0–100)

Our own exploratory analysis revealed that audio features only weakly correlate with popularity, which raised concerns about the effectiveness of a regression-based approach.

### 🎯 What are we aiming to achieve?
Success was defined as achieving a **R² score above 0.50** on a test set, indicating the model could explain at least half the variance in the popularity score. We also aimed to gain insights into which features contributed most to a song's success.

### ⚠️ What factors affect our results?
- **Known factors**: audio features, release date, song duration, explicit content  
- **Unknown/missing variables**: artist fame, playlist inclusion, social media trends, marketing — not included in the dataset

These missing variables limited the accuracy of regression.

### 💡 Is there something new we can use?
To improve performance, we explored:
- Tree-based models (XGBoost, LightGBM, Random Forest)
- Feature engineering (interactions and transformations)
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)


These methods improved results slightly, but performance plateaued, leading us to shift to a classification approach in the next phase.

---

## 🧭 Project Stages Summary

### 1. Data Preparation
- Combined relevant data sources
- Reduced large categorical variables
- Cleaned and formatted data

### 2. EDA – Exploratory Data Analysis
- Visualized target distribution
- Investigated feature relationships and correlations

### 3. Data Cleansing – Outliers and Missing Values
- Identified and addressed outliers
- Handled missing values via imputation or removal

### 4. Feature Engineering & selection
- A new feature track_age was computed as 2025 - release_year.
- A filtered dataset df_recent was created, focusing only on songs released in the last 20 years.
- Skewed numeric features were automatically transformed
- Several custom features were engineered to enhance the predictive power
- A voting-based system was implemented

### 5. One-hot Encoding : 
- StandardScaler was applied to the selected features.
- Boolean features were encoded properly.


### 6. Imbalanced Data – When Relevant
- Analyzed score distribution
- Imbalance handled more during classification phase

### 7. Model Selection and Fine-tuning
- A set of regressors were evaluated using `Pipeline` and `GridSearchCV` :Linear Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost and SVR
- The evaluation metrics included: R² (coefficient of determination), MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)
- The best model was selected based on the highest R² score on the validation set.

### 8. Model Evaluation on Test Set
- The best model was retrained on the training + validation set (85%).
- Predictions were made on the held-out test set (15%).
- Final test performance was reported using: R²,MAE, and RMSE


---

## 📊 Results

- **Best R² Score**: ~0.27
- **Other metrics**:
  - MAE: moderate
  - RMSE: revealed consistent underperformance on extreme values

Despite tuning and feature engineering, none of the models surpassed an R² of 0.30.

---

## ⚠️ Limitations

- **Limited input features**: The dataset only included audio and song-level metadata, excluding crucial variables like artist reputation, marketing impact, playlist features, etc.
- **Skewed distribution**: Popularity scores were unevenly distributed, making it difficult for models to predict high or low extremes accurately.
- **Weak feature-target correlation**: Most audio features had low correlation with the popularity score.

---

## 🧾 Conclusion

The regression phase revealed that predicting Spotify song popularity using audio features alone is a highly challenging task. Even advanced models like Random Forest and XGBoost could not reliably predict popularity with high accuracy.

This led us to pivot toward a **binary classification approach**, predicting whether a song is "popular" (popularity ≥ 50) or not. This simplified task proved more practical and yielded better predictive performance.

---

# 🎯 Spotify Popularity Prediction – Classification Phase

## 📌 Project Overview

### 🔍 What are we trying to find out?
Following the limitations encountered in the regression phase, we redefined the task as a **binary classification problem**. The main objective was to predict whether a song would be considered **popular** or **not popular**, based on its Spotify audio and metadata features.

### 🧠 Why Classification?
- The regression models previously failed to achieve high predictive accuracy (R² < 0.5).
- Popularity prediction is more actionable when defined as a **thresholded decision**.
- Binary classification allows for handling class imbalance and focusing on specific evaluation metrics (e.g. F1-score, ROC AUC).

---

## ✅ Target Definition

A new binary target was created based on the `popularity` score:
- **Popular**: `popularity >= 50`
- **Not Popular**: `popularity < 50`

This allowed us to frame the task as a supervised binary classification problem.

---

## 🧭 Project Stages Summary (Classification)

### 1. Data Preparation
- Reused and enhanced the cleaned dataset from the regression phase

### 2. Class Imbalance Handling
- The dataset was slightly imbalanced (more "not popular" than "popular")
- Techniques used:
  - **SMOTE (Synthetic Minority Oversampling Technique)**
  - **Random UnderSampling** (in some tests)


### 3. Model Selection & Fine Tuning
-Tested and compared several classification models: Logistic Regression, Random Forest, XGBoost, LightGBM, SVC, AdaBoost, Decision Tree and Gradient Boosting Machine (GBM)
- Used `GridSearchCV`
- Focused on optimizing F1-score and ROC AUC
- Constructed a pipeline combining:
  - Preprocessing (scaling, encoding)
  - Classifier
  - **SMOTE** for class imbalance handling
  - **Threshold tuning** to optimize decision boundaries
  - **GridSearchCV** for hyperparameter optimization
  
  
### 4. Evaluation Metrics
- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **ROC AUC**


---

## 🔍 Next Steps

- Try ranking methods (e.g., XGBRanker, LightGBM Ranker)
- Explore adding external data (artist metadata, playlist features)


---

## 📊 Results

- **Best performing model**: Random Forest
- **Evaluation Metrics** (on the test set – Random Forest):
  -Accuracy: 0.691
  - Precision: 0.580
  - Recall: 0.626
  - F1-score: 0.605
  - ROC AUC: 0.752
- Confusion matrix analysis:
  - 813 true positives (popular songs correctly predicted)
  - 486 false negatives (popular songs missed)
  - 574 false positives (non-popular songs predicted as popular)
  - 1553 true negatives (correctly predicted as non-popular)

- **Threshold tuning** significantly improved performance over the default 0.5 threshold, especially for maximizing the F1-score.


---

## ⚠️ Limitations

- **Binary thresholding** simplifies the reality: popularity is a continuous measure with cultural and contextual components.
- **Limited input features**: important aspects such as artist fame, marketing campaigns, playlist placements, and social media trends were not included.
- **Potential overfitting**: despite cross-validation, the model might not generalize to newly released songs outside the training distribution.

---

## 🧾 Conclusion

Transforming the popularity prediction task into a binary classification problem significantly improved model performance and interpretability.

Models like Random Forest and XGBoost, combined with proper preprocessing and class balancing techniques, were able to identify popular songs with reasonable accuracy.

This phase demonstrated that while predicting exact popularity is difficult, classifying songs as "likely popular" is a much more achievable and practical goal for many real-world applications.


---

## 🔮 Recommendations and Future Work

### 🔧 Limitations of Available Data
Despite careful preprocessing and advanced modeling techniques, the available features (audio characteristics + basic metadata) were not sufficient to accurately predict a song’s popularity. Popularity is influenced by various external factors not captured in the dataset, such as:

- Artist popularity and fanbase
- Marketing and promotional campaigns
- Playlist inclusion and curation
- Social media trends and virality
- Timing of release (seasonality, events)

### 🤖 Deep Learning as an Exploratory Direction
Deep learning models could be explored to:
- Better capture complex nonlinear interactions (via MLPs)
- Process additional data types (lyrics, spectrograms, images, etc.)
- Learn abstract representations of songs

However, without richer data sources, the benefit of deep learning may be limited. It becomes more powerful when trained on **diverse, multimodal datasets**.

### 🌍 Dataset Enrichment Opportunities
To improve prediction quality, the following enhancements are suggested:
- Social signals (followers, likes, virality metrics)
- Playlist metadata (positions, frequency of appearance)
- Artist or label-level features
- User feedback (likes/dislikes, comments, engagement)


---

##  Overall Conclusion
This project successfully explored different modeling strategies and clearly revealed the structural limitations of the dataset. Binary classification proved to be a more robust and realistic approach for the given data. However, integrating **external data and exploring deeper models** could open the door to significantly better performance in future work.


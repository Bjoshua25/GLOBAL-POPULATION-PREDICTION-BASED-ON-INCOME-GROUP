# **GLOBAL POPULATION PREDICTION BASED ON INCOME GROUP**  
*Demographic Analysis Using Random Forest Regression*  

## **INTRODUCTION**  
Population growth patterns vary significantly based on a country’s **income classification** (Low, Middle, or High-income). This project applies **Random Forest Regression** to analyze **global population trends** and predict future growth based on income categories.  

By leveraging **machine learning and data visualization**, we identify **key economic indicators** influencing demographic shifts.  

---

## **PROBLEM STATEMENT**  
Understanding **population trends across income groups** is essential for:  
- **Government Planning** – Infrastructure, healthcare, and education.  
- **Economic Forecasting** – Workforce availability, labor market trends.  
- **Sustainable Development Goals (SDGs)** – Addressing social inequalities.  

This project aims to:  
- **Analyze global population trends** from 1960 to 2017.  
- **Predict future population growth rates** based on income groups.  
- **Evaluate model accuracy** using standard regression metrics.  

---

## **SKILL DEMONSTRATION**  
- **Exploratory Data Analysis (EDA) & Correlation Analysis**  
- **Feature Engineering & Encoding**  
- **Random Forest Regression Modeling**  
- **Model Evaluation & Performance Visualization**  

---

## **DATA SOURCING**  
The dataset is sourced from [World Bank Open Data](https://databank.worldbank.org/source/population-estimates-and-projections) and includes:  

### **1. Population Data (1960-2017)**  
- **Annual Population Estimates** – Total population for each country.  
- **Country Income Classification** – High, Middle, Low-income groups.  

### **2. Economic Indicators**  
- **GDP per Capita**  
- **Urbanization Rate (%)**  
- **Fertility Rate**  

---

## **EXPLORATORY DATA ANALYSIS (EDA)**  
EDA was conducted to **identify patterns in population growth** across income groups.  

### **1. Data Overview**  
- **Checked dataset structure** using `.info()` and `.describe()`.  
- **Handled missing values** and outliers.  

### **2. Population Growth Trends**  
- **Line Plot:**  
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.lineplot(x=years, y=population_df.mean(), label="Global Average", color="red")
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Global Population Growth Trends (1960-2017)")
plt.show()
```
- **Key Insight:** Population growth rates differ significantly across income groups.  

### **3. Correlation Analysis**  
- **Heatmap to analyze feature relationships:**  
```python
plt.figure(figsize=(10,6))
sns.heatmap(population_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Population Features')
plt.show()
```
- **Key Finding:** Fertility rate and GDP per capita strongly influence population growth.  

---

## **RANDOM FOREST MODEL**  
A **Random Forest Regression Model** was trained to predict population growth rates.  

### **1. Model Implementation**  
- **Independent Variables (`X`)**: Economic & demographic indicators.  
- **Dependent Variable (`y`)**: Population Growth Rate.  
- **Model Used**: `sklearn.ensemble.RandomForestRegressor`  

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### **2. Model Evaluation**  
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² Score (Explained Variance)**  

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## **MODEL INTERPRETATION & VISUALIZATION**  
### **1. Residual Plot**  
Visualizing errors in predictions:  
```python
plt.figure(figsize=(8,6))
sns.residplot(x=y_pred, y=y_test - y_pred, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Predicted Population')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```
- **Key Finding:** Residuals are randomly distributed, confirming a well-fitted model.  

### **2. Feature Importance**  
Identifying the most **impactful predictors of population growth**:  
```python
tree_feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
tree_feature_importance = tree_feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=tree_feature_importance, palette='viridis')
plt.title('Feature Importance in Random Forest Model')
plt.show()
```
- **Key Finding:** GDP per capita and urbanization rate significantly influence population growth.  

---

## **CONCLUSION**  
1. **GDP per capita and fertility rate strongly influence population growth.**  
2. **Random Forest Regression effectively models global population trends.**  
3. Future improvements should include **additional socioeconomic factors** for better accuracy.


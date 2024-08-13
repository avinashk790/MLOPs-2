import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# import libraries

# Load dataset
df = pd.read_csv('ds_salaries.csv')

# Load the original dataset to get unique values for dropdowns
df_original = pd.read_csv('ds_salaries.csv')

# Load the best model
# best_model = joblib.load('best_model.pkl')

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Drop the salary_currency column as it's not needed for prediction
df = df.drop(columns=['salary_currency'])

# Encode categorical variables
categorical_columns = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define features and target variable
X = df.drop(['salary', 'salary_in_usd'], axis=1)
y = df['salary_in_usd']

# Split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and experiment tracking with MLflow
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)


# Streamlit app
st.title('*Salary Predictor*')

# Input features
experience_level = st.selectbox('Experience Level', df_original['experience_level'].unique())
employment_type = st.selectbox('Employment Type', df_original['employment_type'].unique())
job_title = st.selectbox('Job Title', df_original['job_title'].unique())
employee_residence = st.selectbox('Employee Residence', df_original['employee_residence'].unique())
remote_ratio = st.selectbox('Remote Ratio', df_original['remote_ratio'].unique())
company_location = st.selectbox('Company Location', df_original['company_location'].unique())
company_size = st.selectbox('Company Size', df_original['company_size'].unique())

# Predict salary
input_data = pd.DataFrame({
    'work_year': [2023],
    'experience_level': [experience_level],
    'employment_type': [employment_type],
    'job_title': [job_title],
    'employee_residence': [employee_residence],
    'remote_ratio': [remote_ratio],
    'company_location': [company_location],
    'company_size': [company_size]
})

# Encode categorical variables
categorical_columns = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Align input data with training data columns
input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

# joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
# Predict the salary
salary_prediction = grid_search.best_estimator_.predict(input_data)[0]
st.write(f'Predicted Salary: ${salary_prediction:.2f}')

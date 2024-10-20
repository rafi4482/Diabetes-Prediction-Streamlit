import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv('diabetes_prediction_dataset.csv')
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data['smoking_history'] = data['smoking_history'].astype('category').cat.codes

    X = data.drop('diabetes', axis=1)
    y = data['diabetes']

    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_cols = ['gender', 'smoking_history']

    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])
    X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

    return X, y

X, y = load_and_prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("🩺 Diabetes Prediction App")
st.markdown("---")

with st.form("input_form", clear_on_submit=False):
    st.subheader("Enter the following details:")
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        hypertension = st.selectbox("Hypertension", ["Yes", "No"], key="hypertension")
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key="heart_disease")

    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1, format="%d", key="age")
        bmi = st.number_input("BMI", min_value=10, max_value=95, value=27, step=1, format="%d", key="bmi")
        smoking_history = st.selectbox(
            "Smoking History", ["never", "current", "former", "not current"], key="smoking_history"
        )

    with col3:
        HbA1c_level = st.number_input("HbA1c Level", min_value=3.5, max_value=9.0, value=5.5, step=0.1, key="HbA1c")
        blood_glucose_level = st.number_input(
            "Blood Glucose Level", min_value=80, max_value=300, value=140, step=1, key="glucose"
        )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(
            """
            <style>
            div.stButton > button {
                margin-top: 20px;
                margin-bottom: 20px;
                margin-left: auto;
                margin-right: auto;
                display: block;
                width: 50%;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        submitted = st.form_submit_button("💡 Predict")

    if submitted:
        input_data = pd.DataFrame({
            "gender": [1 if gender == "Male" else 0],
            "age": [age],
            "hypertension": [1 if hypertension == "Yes" else 0],
            "heart_disease": [1 if heart_disease == "Yes" else 0],
            "smoking_history": [{
                "never": 0, "No Info": 1, "former": 2,
                "current": 3, "ever": 4, "not current": 5
            }[smoking_history]],
            "bmi": [bmi],
            "HbA1c_level": [HbA1c_level],
            "blood_glucose_level": [blood_glucose_level],
        })

        prediction = rf_model.predict(input_data)[0]
        prediction_proba = rf_model.predict_proba(input_data)[0]

        st.markdown("---")
        st.subheader("Prediction Result")
        result_text = "Diabetes" if prediction == 1 else "No Diabetes"
        st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{result_text}</h3>", unsafe_allow_html=True)

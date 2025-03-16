import pandas as pd
import re
import zipfile
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
import streamlit as st
def clean_total_square(total_square):
    if isinstance(total_square, (float, int)):
        return float(total_square)
    match = re.search(r'(d+([,.]d+)?)', total_square)
    if match:
        return float(match.group(0).replace(',', '.'))
    return None
def prepare_data():
    with zipfile.ZipFile('train_data.zip') as zf:
        data = pd.read_csv(zf.open('realty_data.csv'))
    data['total_square'] = data['total_square'].apply(clean_total_square)
    data['city'] = data['city'].str.strip()
    data = data.dropna(subset=['total_square'])
    encoder = OneHotEncoder(sparse_output=False)
    city_encoded = encoder.fit_transform(data[['city']])
    city_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(['city']))
    X = pd.concat([data[['total_square']], city_df], axis=1)
    y = data['price'].astype(float)
    joblib.dump(encoder, 'city_encoder.pkl')
    return X, y
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'real_estate_model.pkl')
def read_model(model_path):
    return joblib.load(model_path)
def load_encoder(encoder_path):
    return joblib.load(encoder_path)
st.set_page_config(page_title="Прогнозирование стоимости недвижимости")
model_path = 'real_estate_model.pkl'
encoder_path = 'city_encoder.pkl'
if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    X, y = prepare_data()
    train_model(X, y)
total_square_input = st.sidebar.text_input("Введите площадь (м²):")
city_input = st.sidebar.text_input("Введите название города Московской области:")
if total_square_input and city_input:
    try:
        total_square_value = float(total_square_input.replace('м²', '').replace(',', '.').strip())
        if total_square_value <= 0:
            raise ValueError("Площадь должна быть положительным числом.")
        inputDF = pd.DataFrame(
            {
                "total_square": [total_square_value],
                "city": [city_input]
            }
        )
        model = read_model(model_path)
        encoder = load_encoder(encoder_path)
        city_encoded = encoder.transform(inputDF[['city']])
        city_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(['city']))
        inputDF_encoded = pd.concat([inputDF[['total_square']], city_df], axis=1)
        prediction = model.predict(inputDF_encoded)[0]
        st.write(f"Предсказанная стоимость недвижимости: {prediction:.2f}")
    except ValueError as e:
        st.write(f"Ошибка: {str(e)}")
else:
    st.button("Предсказать")
    st.write("Пожалуйста, введите значения для всех полей.")











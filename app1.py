# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2024)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import pickle

import streamlit as st
from PIL import Image
import pandas as pd

# ====================== главная страница ============================

# параметры главной страницы
# https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config
st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto',
    page_title='Отток app',
    page_icon='🧊',
)


# ----------- функции -------------------------------------

# функция для загрузки картики с диска
# кэшируем иначе каждый раз будет загружатся заново
@st.cache_data
def load_image(image_path):
    image = Image.open(image_path)
    return image

# функция загрузки модели
# кэшируем иначе каждый раз будет загружатся заново
@st.cache_data
def load_model(model_path):
    # загрузка сериализованной модели
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# ------------- загрузка картинки для страницы и модели ---------

# путь до картинки
image_path = 'main_page_image.jpg'
image = load_image(image_path)

# путь до модели
model_path = 'cat_model.pkl'
churn_model = load_model(model_path)


# ---------- отрисовка текста и картинки ------------------------
st.write(
    """
    ### Диагностика оттока клиента с кредитной карты
    Введите ваши данные и получите результат
    """
)

# отрисовка картинки на странице
st.image(image, width=600)


# ====================== боковое меню для ввода данных ===============

st.sidebar.header('Входные данные пользователя')

# словарь с названиями признаков на русском для отображения в приложении
features = dict(
    Customer_Age = 'Возраст',
    Gender = 'Как давно клиент активировал кредитную карту (мес.)',
    Marital_Status = 'Семейный статус',
    Card_Category = 'Категория кредитной карты',
    Months_on_book = 'Как давно клиент активировал кредитную карту (мес.)',
    Total_Relationship_Count = 'Количество продуктов в банке',
    Months_Inactive_12_mon = 'Сколько месяцев клиент не активен в течение года',
    Contacts_Count_12_mon = 'Последняя коммуникация с клиентом',
    Credit_Limit = 'Кредитный лимит',
    Total_Revolving_Bal = 'Возобновляемый баланс',
    Total_Amt_Chng_Q4_Q1 = 'Изменение суммы кредита Q4/Q1',
    Total_Trans_Ct = 'Количество транзакций',
    Total_Ct_Chng_Q4_Q1 = 'Изменение количества транзакций Q4/Q1'
)

# словари с названиями признаков и соответствующими индексами
gender_label_to_index = {'Female': 0, 'Male': 1}
ms_label_to_index = {
    'Divorced': 0,
    'Married': 1,
    'Single': 2}
card_label_to_index = {
    'Blue': 0,
    'Gold': 1,
    'Platinum': 2,
    'Silver': 3}

# кнопки - слайдеры для ввода данных человека
Gender = st.sidebar.radio(features['Gender'], gender_label_to_index.keys(), horizontal=True)
Marital_Status = st.sidebar.radio(features['Marital_Status'], ms_label_to_index.keys(), horizontal=True)
Card_Category = st.sidebar.radio(features['Card_Category'], card_label_to_index.keys(), horizontal=True)
Customer_Age = st.sidebar.slider(features['Customer_Age'], min_value=26, max_value=73, value=42, step=1)
Months_on_book = st.sidebar.slider(features['Months_on_book'], min_value=26, max_value=73, value=5, step=1)
Total_Relationship_Count = st.sidebar.slider(features['Total_Relationship_Count'], min_value=1, max_value=6, value=2, step=1)
Months_Inactive_12_mon = st.sidebar.slider(features['Months_Inactive_12_mon'], min_value=0, max_value=6, value=2, step=1)
Contacts_Count_12_mon = st.sidebar.slider(features['Contacts_Count_12_mon'], min_value=0, max_value=6, value=2, step=1)
Credit_Limit = st.sidebar.slider(features['Credit_Limit'], min_value=1400, max_value=34510, value=3000, step=1)
Total_Revolving_Bal = st.sidebar.slider(features['Total_Revolving_Bal'], min_value=0, max_value=2517, value=1000, step=1)
Total_Amt_Chng_Q4_Q1 = st.sidebar.slider(features['Total_Amt_Chng_Q4_Q1'], min_value=0, max_value=4, value=2, step=1)
Total_Trans_Ct = st.sidebar.slider(features['Total_Trans_Ct'], min_value=10, max_value=139, value=100, step=1)
Total_Ct_Chng_Q4_Q1 = st.sidebar.slider(features['Total_Ct_Chng_Q4_Q1'], min_value=0, max_value=4, value=2, step=1)


# записать входные данные в словарь и в датафрейм
data_df = pd.DataFrame([dict(
    Gender=gender_label_to_index[Gender],
    Marital_Status=ms_label_to_index[Marital_Status],
    Card_Category=card_label_to_index[Card_Category],
    Customer_Age=Customer_Age,
    Months_on_book=Months_on_book,
    Total_Relationship_Count=Total_Relationship_Count,
    Months_Inactive_12_mon=Months_Inactive_12_mon,
    Contacts_Count_12_mon=Contacts_Count_12_mon,
    Credit_Limit=Credit_Limit,
    Total_Revolving_Bal=Total_Revolving_Bal,
    Total_Amt_Chng_Q4_Q1=Total_Amt_Chng_Q4_Q1,
    Total_Trans_Ct=Total_Trans_Ct,
    Total_Ct_Chng_Q4_Q1=Total_Ct_Chng_Q4_Q1,
)])


# =========== вывод входных данных и предсказания модели ==========

# вывести входные данные на страницу
st.write("##### Ваши данные")
st.write(data_df)


# предикт моделью входных данных, на выходе вероятность диабета
churn_prob = churn_model.predict_proba(data_df.values)[0, 1]


# вывести предсказание модели
st.write("##### Вероятность оттока")
st.write(f'{churn_prob:.2f}')
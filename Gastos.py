import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor


st.write(''' # Predicción de gasto de Joaquín Amaro ''')
st.image("gastos.jpg", caption="Cuánto gasta Joaquín en una compra según ciertos datos")

st.header('Datos del gasto')

def user_input_features():
  # Entrada
  presupuesto = st.number_input('Presupuesto:', min_value=0, max_value=10000, value = 0, step = 1)
  tiempo = st.number_input('Tiempo invertido (minutos)', min_value=0, max_value=300, value = 0, step = 1)
  tipo = st.number_input('Tipo',min_value=1, max_value=6, value = 0, step = 1)
  momento = st.number_input('Momento:', min_value=1, max_value=3, value = 0, step = 1)
  num_de_personas = st.number_input('No. de personas', min_value=1, max_value=50, value = 0, step = 1)

  user_input_data = {'presupuesto': presupuesto,
                     'tiempo': tiempo,
                     'tipo': tipo,
                     'momento': momento,
                     'num_de_personas': num_de_personas}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

gasto =  pd.read_csv('Gastos2.csv', encoding='latin-1')
X = gasto.drop(columns='Costo')
Y = gasto['Costo']

classifier = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, max_features=5, random_state=1613786)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción de Costo')
predicted_cost = prediction[0]

st.write(f'El costo de la compra es de: ${predicted_cost}')

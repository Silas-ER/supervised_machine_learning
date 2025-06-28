import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

from modules.load_data import load_data


#configuração da página
st.set_page_config(
    page_title="Previsão de Atrasos em Voos",
    page_icon="🛬",  
    layout="wide",   
    initial_sidebar_state="collapsed",
)

#Titulo da página
st.title("Previsão de Atrasos em Voos")

tab1, tab2, tab3 = st.tabs(["Dados", "Análise Exploratória", "Modelo"])

with tab1:
    st.subheader("Explore os datasets disponíveis")
    dataset = st.selectbox("Selecione o dataset", ["Flights", "Airlines", "Airports"])

    if dataset == "Flights":
        flights = load_data('data', 'flights.csv')

        st.write("### Visualização - Flights.csv")
        st.dataframe(flights.head())  
        st.write(f"Shape: {flights.shape}")

    elif dataset == "Airlines":
        airlines = load_data('data', 'airlines.csv')

        st.write("### Visualização - Airlines.csv")
        st.dataframe(airlines.head())
        st.write(f"Shape: {airlines.shape}")

    elif dataset == "Airports":
        airports = load_data('data', 'airports.csv')

        st.write("### Visualização - Airports.csv")
        st.dataframe(airports.head())
        st.write(f"Shape: {airports.shape}")
with tab2:
    pass
with tab3:
    pass
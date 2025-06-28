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

    # Caso o dataset selecionado seja Flights
    if dataset == "Flights":
        flights = load_data('data', 'flights.csv')

        st.subheader("Descrição das Colunas")
        colunas_flights = {
            "YEAR": "Ano em que o voo ocorreu.",
            "MONTH": "Mês do ano em que o voo ocorreu.",
            "DAY": "Dia do mês em que o voo ocorreu.",
            "DAY_OF_WEEK": "Dia da semana (1 = Segunda-feira, ..., 7 = Domingo).",
            "AIRLINE": "Código IATA da companhia aérea responsável pelo voo.",
            "FLIGHT_NUMBER": "Número único do voo.",
            "TAIL_NUMBER": "Código único da aeronave usada no voo.",
            "ORIGIN_AIRPORT": "Código IATA do aeroporto de origem.",
            "DESTINATION_AIRPORT": "Código IATA do aeroporto de destino.",
            "SCHEDULED_DEPARTURE": "Horário programado de partida (HHMM).",
            "DEPARTURE_TIME": "Horário real de partida (HHMM).",
            "DEPARTURE_DELAY": "Atraso na decolagem em minutos (negativo = adiantado).",
            "TAXI_OUT": "Tempo em solo antes da decolagem (minutos).",
            "WHEELS_OFF": "Horário (HHMM) em que o avião decolou.",
            "SCHEDULED_TIME": "Tempo total programado do voo (minutos).",
            "ELAPSED_TIME": "Tempo total real do voo, incluindo atrasos.",
            "AIR_TIME": "Tempo efetivo no ar (minutos).",
            "DISTANCE": "Distância percorrida (milhas).",
            "WHEELS_ON": "Horário (HHMM) que o avião pousou.",
            "TAXI_IN": "Tempo no solo ao chegar (minutos).",
            "SCHEDULED_ARRIVAL": "Horário programado da chegada (HHMM).",
            "ARRIVAL_TIME": "Horário real da chegada (HHMM).",
            "ARRIVAL_DELAY": "Atraso em minutos (negativo = adiantado).",
            "DIVERTED": "Desvio para outro aeroporto (1 = Sim, 0 = Não).",
            "CANCELLED": "Cancelamento (1 = Sim, 0 = Não).",
            "CANCELLATION_REASON": "Razão do cancelamento (A = Companhia, B = Clima, C = Sistema Aéreo).",
            "AIR_SYSTEM_DELAY": "Atraso no sistema aéreo (minutos).",
            "SECURITY_DELAY": "Atrasos devido à segurança (minutos).",
            "AIRLINE_DELAY": "Atraso causado pela companhia aérea (minutos).",
            "LATE_AIRCRAFT_DELAY": "Atraso por chegada tardia de outro avião.",
            "WEATHER_DELAY": "Atraso devido ao clima (minutos)."
        }
        st.table(pd.DataFrame(list(colunas_flights.items()), columns=["Colunas", "Descrição"]))
        
        st.subheader("Visualização - Flights.csv")
        st.write("5 primeiras linhas do dataset:")
        st.dataframe(flights.head())  

        st.subheader("Resumo Estatístico")
        st.write("Veja as principais estatísticas das variáveis numéricas:")
        st.write(flights.describe())

        st.subheader("Estrutura do Dataset")
        st.write("Tipos dos dados:")
        info_df = pd.DataFrame({
            'Coluna': flights.columns,
            'Tipo': flights.dtypes.values,
            'Não-Nulos': flights.count().values,
            'Total': len(flights)
        })
        st.dataframe(info_df)

        st.write(f"Shape: {flights.shape}")

    # Caso o dataset selecionado seja Airlines
    elif dataset == "Airlines":
        airlines = load_data('data', 'airlines.csv')

        st.subheader("Descrição das Colunas")
        colunas_airlines = {
            "IATA_CODE": "Código IATA único da companhia aérea (ex.: AA = American Airlines).",
            "AIRLINE": "Nome completo da companhia aérea (ex.: Delta Air Lines, United Airlines, etc.)."
        }
        st.table(pd.DataFrame(list(colunas_airlines.items()), columns=["Colunas", "Descrição"]))
        
        st.subheader("Visualização - airlines.csv")
        st.write("5 primeiras linhas do dataset:")
        st.dataframe(airlines.head())  

        st.subheader("Resumo Estatístico")
        st.write("Veja as principais estatísticas das variáveis numéricas:")
        st.write(airlines.describe())

        st.subheader("Estrutura do Dataset")
        st.write("Tipos dos dados:")
        info_df = pd.DataFrame({
            'Coluna': airlines.columns,
            'Tipo': airlines.dtypes.values,
            'Não-Nulos': airlines.count().values,
            'Total': len(airlines)
        })
        st.dataframe(info_df)

        st.write(f"Shape: {airlines.shape}")

    # Caso o dataset selecionado seja Airports
    elif dataset == "Airports":
        airports = load_data('data', 'airports.csv')

        st.subheader("Descrição das Colunas")
        colunas_airports = {
            "IATA_CODE": "Código IATA único do aeroporto (ex.: ATL = Atlanta International Airport).",
            "AIRPORT": "Nome oficial do aeroporto (ex.: Los Angeles International Airport).",
            "CITY": "Cidade onde o aeroporto está localizado.",
            "STATE": "Estado ou região onde o aeroporto está localizado.",
            "COUNTRY": "País onde o aeroporto está localizado.",
            "LATITUDE": "Latitude geográfica do aeroporto (coordenadas WGS84).",
            "LONGITUDE": "Longitude geográfica do aeroporto (coordenadas WGS84)."
        }
        st.table(pd.DataFrame(list(colunas_airports.items()), columns=["Colunas", "Descrição"]))
        
        st.subheader("Visualização - airports.csv")
        st.write("5 primeiras linhas do dataset:")
        st.dataframe(airports.head())  

        st.subheader("Resumo Estatístico")
        st.write("Veja as principais estatísticas das variáveis numéricas:")
        st.write(airports.describe())

        st.subheader("Estrutura do Dataset")
        st.write("Tipos dos dados:")
        info_df = pd.DataFrame({
            'Coluna': airports.columns,
            'Tipo': airports.dtypes.values,
            'Não-Nulos': airports.count().values,
            'Total': len(airports)
        })
        st.dataframe(info_df)

        st.write(f"Shape: {airports.shape}")
with tab2:
    pass
with tab3:
    pass
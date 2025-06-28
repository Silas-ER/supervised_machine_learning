import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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
    st.header("Análise Exploratória de Dados - EDA")

    col1, col2 = st.columns(2)

    with col1:
        # Atrasos na chegada
        st.subheader("Distribuição de Atrasos na Chegada")
        delayed_pct = flights['ARRIVAL_DELAY'].apply(lambda x: 1 if x > 15 else 0).mean() * 100
        st.write(f"Aproximadamente {delayed_pct:.2f}% dos voos sofreram atrasos superiores a 15 minutos.")

        fig, ax = plt.subplots()
        sns.histplot(flights['ARRIVAL_DELAY'], bins=50, kde=True, ax=ax, color='blue')
        ax.set_title('Distribuição dos Atrasos na Chegada (ARRIVAL_DELAY)')
        ax.set_xlabel('Minutos de Atraso')
        ax.set_ylabel('Frequência')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        st.pyplot(fig)

        st.markdown(""" 
            **Observações:**
            - O gráfico mostra que a maior parte dos atrasos na chegada está concentrada perto de 0 minutos (ou seja, muitos voos são pontuais ou chegam apenas alguns minutos atrasados).
            - A cauda do gráfico esticada para a direita indica haver voos com atrasos muito longos (ex.: voos que se atrasam por dezenas ou centenas de minutos).
        """)

    with col2:
        # Atrasos na decolagem
        st.subheader("Distribuição de Atrasos na Decolagem")
        delayed_departure_pct = flights['DEPARTURE_DELAY'].apply(lambda x: 1 if x > 15 else 0).mean() * 100
        st.write(f"Aproximadamente {delayed_departure_pct:.2f}% dos voos sofreram atrasos superiores a 15 minutos na decolagem.")

        fig, ax = plt.subplots()
        sns.histplot(flights['DEPARTURE_DELAY'], bins=50, kde=True, ax=ax, color='orange')
        ax.set_title('Distribuição dos Atrasos na Decolagem (DEPARTURE_DELAY)')
        ax.set_xlabel('Minutos de Atraso')
        ax.set_ylabel('Frequência')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        st.pyplot(fig)

        st.markdown("""
            **Observações:**
            - Há uma concentração de voos pontuais ou com pequenos atrasos ou adiantamentos (próximo de 0 minutos).
            - Assim como na chegada, existe uma cauda longa à direita nos atrasos de decolagem, indicando que voos atrasados na partida podem ter impacto significativo.
        """)
    
    # Como os atrasos na partida afetam os atrasos na chegada
    st.subheader("Impacto do Atraso na Partida no Atraso na Chegada")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=flights['DEPARTURE_DELAY'], y=flights['ARRIVAL_DELAY'], alpha=0.5, ax=ax)
    ax.set_title('Relação entre Atraso na Partida e na Chegada')
    ax.set_xlabel('Atraso na Partida (minutos)')
    ax.set_ylabel('Atraso na Chegada (minutos)')
    st.pyplot(fig)

    st.markdown("""
        **Observações:**
        - Quanto maior o atraso na partida, maior o atraso na chegada. A relação é quase linear, ou seja, se um voo atrasa 100 minutos na partida, ele tende a atrasar cerca de 100 minutos na chegada.
        - Há pouca ou nenhuma recuperação de tempo significativo: Os pontos não se desviam muito da linha diagonal, o que indica que, em geral, os voos não conseguem compensar grandes atrasos que ocorrem na partida.
        - Atrasos na partida são os principais impulsionadores dos atrasos na chegada. Raramente você vê um ponto com um grande atraso na partida e um pequeno atraso na chegada (o que significaria que o voo recuperou muito tempo).
    """)

    col1, col2 = st.columns(2)

    with col1:
        # Atrasos por cia aerea
        st.subheader("Taxa de Atrasos por Companhia Aérea")
        airline_delays = flights.groupby('AIRLINE')['ARRIVAL_DELAY'].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        airline_delays.sort_values().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Média de Atrasos por Companhia Aérea')
        ax.set_xlabel('Companhia Aérea')
        ax.set_ylabel('Atraso Médio na Chegada (minutos)')
        st.pyplot(fig)

    with col2:
        # Atrasos por aeroportos
        st.subheader("Top 10 Aeroportos com Mais Atrasos Médios")
        airport_delays = flights.groupby('ORIGIN_AIRPORT')['ARRIVAL_DELAY'].mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        airport_delays.plot(kind='bar', ax=ax, color='red')
        ax.set_title('Atraso Médio na Chegada por Aeroporto de Origem')
        ax.set_xlabel('Aeroporto')
        ax.set_ylabel('Atraso Médio (minutos)')
        st.pyplot(fig)

    with col1:
        # Atrasos por dia da semana
        st.subheader("Atrasos por Dia da Semana")
        flights['DAY_OF_WEEK'] = flights['DAY_OF_WEEK'].map({1: 'Segunda', 2: 'Terça', 3: 'Quarta', 4: 'Quinta',
                                                            5: 'Sexta', 6: 'Sábado', 7: 'Domingo'})

        week_day_delays = flights.groupby('DAY_OF_WEEK')['ARRIVAL_DELAY'].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        week_day_delays.plot(kind='bar', ax=ax, color='green')
        ax.set_title('Média de Atraso por Dia da Semana')
        ax.set_xlabel('Dia da Semana')
        ax.set_ylabel('Atraso Médio (minutos)')
        st.pyplot(fig)
    
    with col2:
        # Atrasos por hora do dia
        st.subheader("Atrasos por Hora do Dia")

        
        flights['HOUR'] = flights['SCHEDULED_DEPARTURE'] // 100
        hourly_delays = flights.groupby('HOUR')['ARRIVAL_DELAY'].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        hourly_delays.plot(kind='line', ax=ax, marker='o')
        ax.set_title('Atrasos Médios por Hora do Dia')
        ax.set_xlabel('Hora do Dia')
        ax.set_ylabel('Atraso Médio (minutos)')
        st.pyplot(fig)

    # Impacto da distancia nos atrasos
    st.subheader("Impacto da Distância no Atraso")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=flights['DISTANCE'], y=flights['ARRIVAL_DELAY'], alpha=0.6, ax=ax)
    ax.set_title('Impacto da Distância no Atraso na Chegada')
    ax.set_xlabel('Distância (milhas)')
    ax.set_ylabel('Atraso na Chegada (minutos)')
    st.pyplot(fig)

    st.markdown("""
        **Observações:**
        - Há uma densa concentração de pontos nas distâncias menores (até aproximadamente 2500-3000 milhas). Isso sugere que a maioria dos voos na base de dados é de curta a média distância.
        - Para todas as faixas de distância, observamos que os atrasos na chegada variam de 0 (ou muito próximos de 0) até valores bastante altos (por volta de 1500-2000 minutos). Isso significa que voos curtos, médios e longos podem sofrer atrasos significativos.
        - Não há uma linha ou curva óbvia que mostre que voos mais longos têm sistematicamente mais (ou menos) atrasos do que voos mais curtos, ou vice-versa. Os pontos estão espalhados verticalmente em todas as faixas de distância.
        - Este gráfico reforça a ideia de que os atrasos são provavelmente causados por outros fatores, como: Atrasos na partida (como vimos no gráfico anterior), Problemas operacionais no aeroporto de partida ou chegada, Condições meteorológicas, Problemas mecânicos da aeronave, Congestionamento do espaço aéreo.
    """)

    # Analise de correlações
    st.subheader("Correlação entre Variáveis Contínuas")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(flights[['ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'DISTANCE', 'TAXI_OUT', 'AIR_TIME']].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Mapa de Correlação')
    st.pyplot(fig)

    st.markdown("""
        **Maiores Correlações:**
        - ARRIVAL_DELAY vs. DEPARTURE_DELAY (Atraso na Chegada vs. Atraso na Partida):
            - Quanto maior o atraso na partida, maior tende a ser o atraso na chegada.
        - DISTANCE vs. AIR_TIME (Distância vs. Tempo de Voo):
            - É intuitivo: quanto maior a distância do voo, maior o tempo de voo.
        - TAXI_OUT vs. ARRIVAL_DELAY (Tempo de Taxiamento na Saída vs. Atraso na Chegada):
            - Há uma correlação positiva, mas relativamente fraca. Um tempo de taxiamento maior pode contribuir um pouco para o atraso na chegada, mas não é um fator tão dominante quanto o atraso na partida.
    """)

with tab3:
    pass
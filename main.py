import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

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

@st.cache_data
def load_cached_data(dir, filename):
    return load_data(dir, filename)

flights = load_cached_data('data', 'flights.csv')
flights.drop(
    columns=[
        'YEAR',
        'MONTH',
        'DAY',
        'FLIGHT_NUMBER',
        'TAIL_NUMBER',
        'CANCELLATION_REASON',
        'AIRLINE_DELAY',
        'LATE_AIRCRAFT_DELAY',
        'WEATHER_DELAY',
        'AIR_SYSTEM_DELAY',
        'SECURITY_DELAY',
    ],
    inplace=True
)
airlines = load_cached_data('data', 'airlines.csv')
airports = load_cached_data('data', 'airports.csv')

tab1, tab2, tab3 = st.tabs(["Dados", "Análise Exploratória", "Modelo"])

with tab1:
    st.subheader("Explore os datasets disponíveis")
    dataset = st.selectbox("Selecione o dataset", ["Flights", "Airlines", "Airports"])

    # Caso o dataset selecionado seja Flights
    if dataset == "Flights":
        st.subheader("Descrição das colunas utilizadas:")
        colunas_flights = {
            "DAY_OF_WEEK": "Dia da semana (1 = Segunda-feira, ..., 7 = Domingo).",
            "AIRLINE": "Código IATA da companhia aérea responsável pelo voo.",
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
        info_data = []
        for col in flights.columns:
            info_data.append({
                'Coluna': str(col),
                'Tipo': str(flights[col].dtype),
                'Não-Nulos': int(flights[col].count()),
                'Total': int(len(flights))
            })
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df)

        st.write(f"Shape: {flights.shape}")

    # Caso o dataset selecionado seja Airlines
    elif dataset == "Airlines":
        st.subheader("Descrição das Colunas")
        colunas_airlines = {
            "IATA_CODE": ["Código IATA único da companhia aérea (ex.: AA = American Airlines).", "object"],
            "AIRLINE": ["Nome completo da companhia aérea (ex.: Delta Air Lines, United Airlines, etc.).", "object"]
        }
        
        
        airlines_info = []
        for coluna, info in colunas_airlines.items():
            airlines_info.append({
                "Colunas": coluna,
                "Descrição": info[0],
                "Tipo": info[1]
            })
        
        st.table(pd.DataFrame(airlines_info))
        
        st.subheader("Visualização - airlines.csv")
        st.write("5 primeiras linhas do dataset:")
        st.dataframe(airlines.head())  

        st.subheader("Resumo Estatístico")
        st.write("Veja as principais estatísticas das variáveis numéricas:")
        st.write(airlines.describe())

        st.write(f"Shape: {airlines.shape}")

    # Caso o dataset selecionado seja Airports
    elif dataset == "Airports":
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
        info_data = []
        for col in airports.columns:
            info_data.append({
                'Coluna': str(col),
                'Tipo': str(airports[col].dtype),
                'Não-Nulos': int(airports[col].count()),
                'Total': int(len(airports))
            })
        info_df = pd.DataFrame(info_data)
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
        # Create a copy for visualization to avoid modifying original data
        flights_viz = flights.copy()
        flights_viz['DAY_OF_WEEK_NAME'] = flights_viz['DAY_OF_WEEK'].map({1: 'Segunda', 2: 'Terça', 3: 'Quarta', 4: 'Quinta',
                                                            5: 'Sexta', 6: 'Sábado', 7: 'Domingo'})

        week_day_delays = flights_viz.groupby('DAY_OF_WEEK_NAME')['ARRIVAL_DELAY'].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        week_day_delays.plot(kind='bar', ax=ax, color='green')
        ax.set_title('Média de Atraso por Dia da Semana')
        ax.set_xlabel('Dia da Semana')
        ax.set_ylabel('Atraso Médio (minutos)')
        st.pyplot(fig)
    
    with col2:
        # Atrasos por hora do dia
        st.subheader("Atrasos por Hora do Dia")

        # Create HOUR column if it doesn't exist
        if 'HOUR' not in flights.columns:
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
    st.header("Modelagem Preditiva para Atrasos em Voos ✈️")

    st.subheader("Objetivo")
    st.markdown("""
        Treinamento de modelos simples para prever atrasos superiores a 15 minutos.
    """)

    # Variavel target: Atrasos na chegada
    flights['DELAYED'] = flights['ARRIVAL_DELAY'].apply(lambda x: 1 if x > 15 else 0)

    # Criacao da coluna HOUR se não existir
    if 'HOUR' not in flights.columns:
        flights['HOUR'] = flights['SCHEDULED_DEPARTURE'] // 100
    
    # Selecionando as features relevantes
    features = ['DEPARTURE_DELAY', 'TAXI_OUT', 'DISTANCE', 'DAY_OF_WEEK', 'HOUR']
    X = flights[features]
    y = flights['DELAYED']
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Análise de Valores Ausentes")
        missing_data = X.isnull().sum()
        st.write("**Valores ausentes por feature:**")
        st.dataframe(missing_data.to_frame(name='Valores Ausentes'))
    
    # Limpeza dos dados
    complete_cases = X.notna().all(axis=1) & y.notna()
    X_clean = X[complete_cases]
    y_clean = y[complete_cases]

    # Divisao de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    
    with col2:
        st.write("### Informações dos Dados (após limpeza)")
        st.write(f"**Número original de amostras:** {X.shape[0]}")
        st.write(f"**Número de amostras após limpeza:** {X_clean.shape[0]}")
        st.write(f"**Número de features:** {X_clean.shape[1]}")
        st.write(f"**Amostras removidas:** {X.shape[0] - X_clean.shape[0]}")
    
    with col3:
        st.write("### Divisão dos Dados (80% para treinamento e 20% para teste)")
        st.write(f"**Tamanho dos dados de treino:** {X_train.shape[0]}")
        st.write(f"**Tamanho dos dados de teste:** {X_test.shape[0]}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Modelo 1: Regressão Logística")
        # Treinamento
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train, y_train)

        # Previsões
        y_pred = log_reg.predict(X_test)

        st.write("**Relatório de Classificação:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Matriz de Confusão: Regressão Logística")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(log_reg, X_test, y_test, ax=ax)
        st.pyplot(fig)

        st.markdown(""" 
            - True Label 0 (Não Atrasado Real):
                - 925.369 (TN): O modelo previu corretamente que 925.369 voos não atrasariam, e eles de fato não atrasaram.
                - 16.438 (FP): O modelo previu que 16.438 voos atrasariam, mas eles, na verdade, não atrasaram (falsos alarmes).
            - True Label 1 (Atrasado Real):
                - 47.728 (FN): O modelo previu que 47.728 voos não atrasariam, mas eles, na verdade, atrasaram (erros de previsão de atraso).
                - 156.472 (TP): O modelo previu corretamente que 156.472 voos atrasariam, e eles de fato atrasaram.
        """)

    with col2:
        st.subheader("Modelo 2: K-Nearest Neighbors (KNN)")
        # Treinamento
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        # Previsões
        y_pred_knn = knn.predict(X_test)
        st.write("**Relatório de Classificação:**")
        report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
        st.dataframe(pd.DataFrame(report_knn).transpose())

        st.subheader("Matriz de Confusão: KNN")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test, ax=ax)
        st.pyplot(fig)

        st.markdown(""" 
            - True Label 0 (Não Atrasado Real):
                - 920.295 (TN): O modelo previu corretamente que 920.295 voos não atrasariam, e eles de fato não atrasaram.
                - 21.512 (FP): O modelo previu que 21.512 voos atrasariam, mas eles, na verdade, não atrasaram (falsos alarmes).
            - True Label 1 (Atrasado Real):
                - 45.095 (FN): O modelo previu que 45.095 voos não atrasariam, mas eles, na verdade, atrasaram (erros de previsão de atraso).
                - 159.105 (TP): O modelo previu corretamente que 159.105 voos atrasariam, e eles de fato atrasaram.
        """)

    col1, col2 = st.columns(2)

    with col1:
        # Comparação dos Modelos
        st.subheader("Comparação dos Modelos")
        results = {
            "Modelo": ["Regressão Logística", "KNN"],
            "F1-Score": [report["1"]["f1-score"], report_knn["1"]["f1-score"]],
            "Acurácia": [report["accuracy"], report_knn["accuracy"]],
            "Precision": [report["1"]["precision"], report_knn["1"]["precision"]],
            "Recall": [report["1"]["recall"], report_knn["1"]["recall"]],
        }

        st.table(pd.DataFrame(results).set_index("Modelo"))

    with col2:
        st.subheader("Importância das Features (Regressão Logística)")
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': log_reg.coef_[0],
            'Abs_Coefficient': abs(log_reg.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        st.dataframe(feature_importance)
    
    with col1:
        st.subheader("Gráfico de Importância das Features")
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance.plot(x='Feature', y='Coefficient', kind='bar', ax=ax)
        ax.set_title('Coeficientes da Regressão Logística')
        ax.set_ylabel('Coeficiente')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Conclusões")
        st.markdown(""" 
            - DEPARTURE_DELAY e TAXI_OUT são, de longe, as variáveis mais importantes para prever se um voo vai atrasar na chegada, com DEPARTURE_DELAY sendo a mais impactante.
            - HOUR, DAY_OF_WEEK e DISTANCE têm uma importância marginal ou insignificante para este modelo de regressão logística na previsão de atrasos.
        """)
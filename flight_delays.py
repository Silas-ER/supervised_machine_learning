import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

#########################################################
#### MODELO DE PREDICAO DE ATRASOS DE VOO NA CHEGADA ####
#########################################################

st.header("✈️ Modelo para predição de atrasos de voos")
st.subheader("📖 Descrição do Modelo")
st.write("""
    O modelo visa prever se um voo chegará atrasado ao seu destino com base em fatores como:
    - DEP_HOUR: Hora de partida programada (0-23)
    - IS_WEEKEND: Indica se é fim de semana (1) ou não (0)
    - SEASON: Estação do ano (Winter, Spring, Summer, Fall)
    - TIME_PERIOD: Período do dia (Dawn, Morning, Afternoon, Night)
    - HOLIDAY_SEASON: Indica se é temporada de férias/feriados (1) ou não (0)
""")

# Criando variavel de target (objetivo)
flights = pd.read_csv('data/flights.csv', low_memory=False)
flights['IS_DELAYED'] = (flights['ARRIVAL_DELAY'] >= 15).astype(int)

# Convertendo variaveis categoricas em numericas
airline_encoder = LabelEncoder()
origin_encoder = LabelEncoder()
dest_encoder = LabelEncoder()

# Retirar colunas irrelevantes para o modelo
colunas_para_remover = [
    'TAIL_NUMBER', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY',
    'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
    'WEATHER_DELAY', 'DEPARTURE_TIME', 'WHEELS_OFF', 'WHEELS_ON',
    'TAXI_IN', 'TAXI_OUT', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
    'ELAPSED_TIME', 'AIR_TIME', 'YEAR', 'FLIGHT_NUMBER',
    'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_DELAY',
    'DIVERTED', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'
]

# Criação de novas variáveis
flights['DEP_HOUR'] = flights['SCHEDULED_DEPARTURE'] // 100
flights['DEP_HOUR'] = flights['DEP_HOUR'].apply(lambda x: min(x, 23)) 
flights['IS_WEEKEND'] = flights['DAY_OF_WEEK'].isin([6, 7]).astype(int)
flights['SEASON'] = pd.cut(flights['MONTH'], bins=[0,3,6,9,12], labels=['Winter','Spring','Summer','Fall'])
flights['TIME_PERIOD'] = pd.cut(flights['DEP_HOUR'], bins=[0,6,12,18,24], labels=['Dawn','Morning','Afternoon','Night'])
flights['HOLIDAY_SEASON'] = flights['MONTH'].isin([7,8,12]).astype(int)

flights['AIRLINE'] = airline_encoder.fit_transform(flights['AIRLINE'])
flights['ORIGIN_ENCODED'] = origin_encoder.fit_transform(flights['ORIGIN_AIRPORT'])
flights['DEST_ENCODED'] = dest_encoder.fit_transform(flights['DESTINATION_AIRPORT'])

# Limpeza dos dados
flights_cleaned = flights.drop(columns=colunas_para_remover, errors='ignore')

# Tratamento de valores ausentes
flights_cleaned = flights_cleaned.fillna(flights_cleaned.median(numeric_only=True))

#########################################################
### ANALISE INICIAL DOS DADOS APÓS REMOÇÃO DE COLUNAS ###
#########################################################

# Colunas que ficamos após o tratamento dos dados
st.subheader("📊 Análise do Dataset")
st.write("### Features Utilizadas")
flights_exibition = flights_cleaned.drop(columns=['IS_DELAYED'])
features_info = pd.DataFrame({
    'Feature': flights_exibition.columns.astype(str),
    'Tipo': [str(dtype) for dtype in flights_exibition.dtypes],
    'Valores Únicos': [flights_exibition[col].nunique() for col in flights_exibition.columns],
    'Valores Ausentes (%)': [flights_exibition[col].isna().mean() * 100 for col in flights_exibition.columns]
})

st.write("### Estatísticas das Features Numéricas")
numeric_stats = flights_exibition.describe().round(2)
st.dataframe(numeric_stats)

st.write("### Informações das Features")
st.dataframe(features_info.round(2))

st.write("### Primeiras linhas do Dataset")
st.dataframe(flights_exibition.head(5))

#########################################################
###### TREINAMENTO DO MODELO DE REGRESSAO LOGISTICA #####
#########################################################

st.subheader("🤖 Modelo utilizando regressão logística")
st.markdown("""
    ### Por que Regressão Logística?
    
    - **Simplicidade**: Modelo linear de fácil interpretação
    - **Eficiência**: Bom desempenho em classificação binária
    - **Rapidez**: Treinamento rápido mesmo com grandes volumes de dados
    - **Probabilidades**: Fornece probabilidades de atraso
    
    ### Como funciona?
    O modelo analisa as características do voo (horário, distância, aeroportos, etc.) 
    e calcula a probabilidade dele atrasar. Se essa probabilidade for maior que 50%, 
    o voo é classificado como "provável atraso".
    
    ### Vantagens para Previsão de Atrasos:
    - Captura relações lineares entre as features
    - Menos propenso a overfitting que modelos mais complexos
    - Resultados facilmente interpretáveis
    - Boa baseline para comparação com outros modelos
""")

# Separação das features (X) e target (y)            
X = flights_cleaned.drop(columns=['IS_DELAYED'])
y = flights_cleaned['IS_DELAYED']

# Definição das features categóricas e numéricas para preprocessamento
categorical_features = ['SEASON', 'TIME_PERIOD']
numeric_features = [
    'AIRLINE', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'DEP_HOUR', 
    'DISTANCE', 'CANCELLED', 'ORIGIN_ENCODED', 
    'DEST_ENCODED', 'IS_WEEKEND'
    ]

# Criação do preprocessador para tratamento das features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Divisão dos dados em conjuntos de treino (80%) e validação (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicação do preprocessamento
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

# Treinamento do modelo de Regressão Logística
with st.spinner('Treinando o modelo de Regressão Logística...'):
    model = LogisticRegression(random_state=42, max_iter=1000, solver='saga', n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

st.subheader("📊 Perfomance do modelo sem balanceamento de classes")

col1, col2 = st.columns(2)

with col1:
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    
    # Formatando a tabela
    df_report = df_report.round(3)  
    df_report = df_report.drop('support', axis=1)  
    
    # Renomeando os índices para mais legíveis
    df_report.index = df_report.index.map({
        '0': 'Não Atrasado',
        '1': 'Atrasado',
        'accuracy': 'Acurácia',
        'macro avg': 'Média Macro',
        'weighted avg': 'Média Ponderada'
    })
    
    st.write("### Classificação do Modelo:")
    st.dataframe(df_report.style.format("{:.2%}"))

with col2:
    st.subheader("🎯 Matriz de Confusão")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_val, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    plt.title("Logistic Regression - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# Balanceamento de classes com SMOTE
with st.spinner('Aplicando balanceamento de classes...'):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Treinamento do modelo com dados balanceados
    model_balanced = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='saga',
        n_jobs=-1
    )
    model_balanced.fit(X_train_balanced, y_train_balanced)
    
    # Previsões com threshold ajustado
    y_pred_proba = model_balanced.predict_proba(X_val)
    y_pred_balanced = (y_pred_proba[:, 1] > 0.3).astype(int)

st.subheader("📊 Performance do Modelo Balanceado")

# Exibição dos resultados
col1, col2 = st.columns(2)

with col1:
    report = classification_report(y_val, y_pred_balanced, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    
    # Formatando a tabela
    df_report = df_report.round(3)  
    df_report = df_report.drop('support', axis=1)  
    
    # Renomeando os índices
    df_report.index = df_report.index.map({
        '0': 'Não Atrasado',
        '1': 'Atrasado',
        'accuracy': 'Acurácia',
        'macro avg': 'Média Macro',
        'weighted avg': 'Média Ponderada'
    })
    
    st.write("### Classificação do Modelo:")
    st.dataframe(df_report.style.format("{:.2%}"))

with col2:
    st.subheader("🎯 Matriz de Confusão")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_val, y_pred_balanced),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    plt.title("Regressão Logística Balanceada")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    st.pyplot(fig)

#########################################################
############### TREINAMENTO DO MODELO KNN ###############
#########################################################

st.subheader("🎯 KNN Model")
st.markdown("""
    ### Por que KNN?
    
    - **Simplicidade**: Modelo intuitivo baseado em similaridade
    - **Não-paramétrico**: Não assume distribuição dos dados
    - **Versatilidade**: Eficaz em padrões locais
    - **Adaptabilidade**: Ajusta-se naturalmente à complexidade dos dados
    
    ### Como funciona?
    O modelo analisa os K voos mais similares ao voo em questão e decide
    com base na maioria. Por exemplo:
    - Se entre os 5 voos mais similares, 3 atrasaram
    - Então o modelo prevê que este voo também atrasará
    
    ### Vantagens para Previsão de Atrasos:
    - Captura padrões locais de atrasos
    - Considera similaridade entre rotas e condições
    - Adapta-se a diferentes regiões do espaço de features
    - Decisões baseadas em casos reais similares
""")

with st.spinner('Treinando modelo KNN...'):
    y_train_np = y_train.to_numpy()
    
    sample_size = min(100000, int(0.1 * len(X_train)))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train_np[indices]

    knn_param_grid = {
        'n_neighbors': [3, 5],
        'weights': ['uniform'],
        'p': [2]  
    }
    
    knn_grid_search = GridSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        knn_param_grid,
        cv=3,
        scoring={
            'recall': 'recall',
            'precision': 'precision',
            'f1': 'f1'
        },
        refit='recall',
        n_jobs=-1,
        verbose=2
    )
    
    try:
        # 5. Treinamento com dados reduzidos
        knn_grid_search.fit(X_train_sample, y_train_sample)
        knn_model = knn_grid_search.best_estimator_
        knn_pred = knn_model.predict(X_val)
        
    except Exception as e:
        st.error(f"Erro no treinamento: {str(e)}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        # Relatório de classificação
        report = classification_report(y_val, knn_pred, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).transpose()
        
        # Formatação da tabela
        df_report = df_report.round(3)
        df_report = df_report.drop('support', axis=1)
        
        # Renomeando índices
        df_report.index = df_report.index.map({
            '0': 'Não Atrasado',
            '1': 'Atrasado',
            'accuracy': 'Acurácia',
            'macro avg': 'Média Macro',
            'weighted avg': 'Média Ponderada'
        })
        
        st.write("### Classificação do Modelo:")
        st.dataframe(df_report.style.format("{:.2%}"))
        
        # Melhores parâmetros
        st.write("### Melhores Parâmetros:")
        st.json(knn_grid_search.best_params_)

    with col2:
        # Matriz de confusão
        st.subheader("🎯 Matriz de Confusão")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(y_val, knn_pred),
            annot=True,
            fmt="d",
            cmap="Purples",
            ax=ax
        )
        plt.title("KNN - Matriz de Confusão")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        st.pyplot(fig)

#########################################################
###### TREINAMENTO DO MODELO DE ARVORE DE DECISSAO ######
#########################################################

st.subheader("🌳 Modelo de árvore de decisão")
st.markdown("""
    ### Por que Árvore de Decisão?
    
    - **Interpretabilidade**: Fácil visualização das regras de decisão
    - **Não-linear**: Captura relações complexas entre features
    - **Robustez**: Lida bem com diferentes tipos de features
    - **Hierarquia**: Identifica features mais importantes
            
    ### Como funciona?
    O modelo cria uma estrutura em árvore onde cada nó representa uma decisão 
    baseada em uma feature específica. Por exemplo:
    - Se a hora de partida < 6h, vai para um caminho
    - Se a distância > 1000 milhas, vai para outro
    - E assim sucessivamente até chegar a uma previsão final

    ### Vantagens para Previsão de Atrasos:
    - Captura naturalmente padrões sazonais e horários
    - Identifica combinações críticas de fatores que levam a atrasos
    - Permite visualizar o processo de decisão
    - Lida bem com features numéricas e categóricas
    - Funciona mesmo com dados não balanceados
""")

with st.spinner('Treinando a árvore de decisão...'):
    # Convert to numpy array and sample data
    y_train_np = y_train.to_numpy()
    
    sample_size = min(100000, int(0.1 * len(X_train)))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train_np[indices]
    
    param_grid = {
        'max_depth': [5, 10],
        'min_samples_split': [50, 100],
        'min_samples_leaf': [20, 30],
        'criterion': ['gini'],
        'class_weight': [None, 'balanced']
    }

    # Criar e executar GridSearchCV com a amostra
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring={
            'accuracy': 'accuracy',
            'recall': 'recall',
            'precision': 'precision',
            'f1': 'f1'
        },
        refit='recall',  
        n_jobs=-1,
        verbose=1
    )
    
    try:
        # Apply SMOTE on the sample instead of full dataset
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_sample, y_train_sample)
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        dt_model = grid_search.best_estimator_
        dt_pred = dt_model.predict(X_val)
        dt_pred_proba = dt_model.predict_proba(X_val)
        
    except Exception as e:
        st.error(f"Erro no treinamento: {str(e)}")
        st.stop()

# Mostrar performance do Decision Tree
col1, col2 = st.columns(2)

with col1:
    # Métricas detalhadas
    report = classification_report(y_val, dt_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(3)
    df_report = df_report.drop('support', axis=1)
    
    df_report.index = df_report.index.map({
        '0': 'Não Atrasado',
        '1': 'Atrasado',
        'accuracy': 'Acurácia',
        'macro avg': 'Média Macro',
        'weighted avg': 'Média Ponderada'
    })
    
    st.write("### Classificação do Modelo:")
    st.dataframe(df_report.style.format("{:.2%}"))
    
    # Melhores parâmetros
    st.write("### Melhores Parâmetros:")
    st.json(grid_search.best_params_)

with col2:
    # Matriz de confusão
    st.subheader("🎯 Matriz de Confusão")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_val, dt_pred),
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax
    )
    plt.title("Árvore de Decisão - Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    st.pyplot(fig)
    
    # Feature Importance
    st.subheader("🔍 Importância das Features")
    feature_names = (
        numeric_features + 
        [f"{feat}_{val}" for feat, vals in 
         zip(categorical_features, 
             preprocessor.named_transformers_['cat'].categories_) 
         for val in vals[1:]]  # Skip first category due to drop='first'
    )
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot top 15 most important features
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=feature_importance.head(15),
        x='Importance',
        y='Feature'
    )
    plt.title("Top 15 Features mais Importantes")
    plt.tight_layout()
    st.pyplot(fig)

#########################################################
################## TENTATIVA DE PREVISAO ################
#########################################################

st.subheader("✈️ Predictor de Atrasos de Voos")

with st.form("flight_predictor"):
    col1, col2 = st.columns(2)
    
    with col1:
        month = st.number_input("Mês", min_value=1, max_value=12, value=6)
        day = st.number_input("Dia", min_value=1, max_value=31, value=15)
        day_of_week = st.number_input("Dia da Semana (1-7)", min_value=1, max_value=7, value=1)
        dep_hour = st.number_input("Hora de Partida (0-23)", min_value=0, max_value=23, value=14)
        distance = st.number_input("Distância (milhas)", min_value=0, max_value=5000, value=2475)
    
    with col2:
        airline = st.selectbox("Companhia Aérea", options=sorted(flights['AIRLINE'].unique()))
        origin = st.selectbox("Aeroporto de Origem", options=sorted(flights['ORIGIN_AIRPORT'].unique()))
        destination = st.selectbox("Aeroporto de Destino", options=sorted(flights['DESTINATION_AIRPORT'].unique()))
        cancelled = st.checkbox("Voo Cancelado")
    
    # Adicionar campos para as novas features
    is_weekend = 1 if day_of_week in [6, 7] else 0
    season = pd.cut([month], bins=[0,3,6,9,12], labels=['Winter','Spring','Summer','Fall'])[0]
    time_period = pd.cut([dep_hour], bins=[0,6,12,18,24], labels=['Dawn','Morning','Afternoon','Night'])[0]
    holiday_season = 1 if month in [7,8,12] else 0
    
    model_choice = st.radio(
        "Escolha o Modelo",
        ["Árvore de Decisão", "KNN"]
    )

    submitted = st.form_submit_button("Prever Atraso")

if submitted:
    # Criar DataFrame com todas as features necessárias
    input_data = pd.DataFrame([{
        'AIRLINE': airline_encoder.transform([airline])[0],
        'MONTH': month,
        'DAY': day,
        'DAY_OF_WEEK': day_of_week,
        'DEP_HOUR': dep_hour,
        'DISTANCE': distance,
        'CANCELLED': int(cancelled),
        'ORIGIN_ENCODED': origin_encoder.transform([origin])[0],
        'DEST_ENCODED': dest_encoder.transform([destination])[0],
        'IS_WEEKEND': is_weekend,
        'SEASON': season,
        'TIME_PERIOD': time_period,
        'HOLIDAY_SEASON': holiday_season
    }])

    # Preprocessar dados de entrada
    input_processed = preprocessor.transform(input_data)

    # Fazer predição com o modelo escolhido
    if model_choice == "Árvore de Decisão":
        prediction = dt_model.predict(input_processed)[0]
        probability = dt_model.predict_proba(input_processed)[0][1]
    else:  # KNN
        prediction = knn_model.predict(input_processed)[0]
        probability = knn_model.predict_proba(input_processed)[0][1]

    # Mostrar resultados
    result = '🟥 Atrasado' if prediction == 1 else '🟩 No Horário'
    
    st.markdown(f"### Previsão: {result}")
    st.markdown(f"### Probabilidade de Atraso: {probability:.2%}")
    
    if prediction == 1:
        st.warning("Este voo provavelmente irá atrasar!")
    else:
        st.success("Este voo provavelmente chegará no horário!")

    # Se for árvore de decisão, mostrar importância das features
    if model_choice == "Árvore de Decisão":
        st.subheader("🔍 Importância das Features para esta Previsão")
        feature_names = (
            numeric_features + 
            [f"{feat}_{val}" for feat, vals in 
             zip(categorical_features, 
                 preprocessor.named_transformers_['cat'].categories_) 
             for val in vals[1:]]
        )
        
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': dt_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(5)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=importance, x='Importance', y='Feature')
        plt.title("Top 5 Features Mais Importantes")
        plt.tight_layout()
        st.pyplot(fig)
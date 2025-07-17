# ✈️ Previsão de Atrasos de Voos Utilizando Aprendizado de Máquina Supervisionado

> Projeto da terceira unidade de Aprendizado de Máquina Supervisionado - IMD3002

## 📋 Descrição do Projeto

Este projeto tem como objetivo desenvolver modelos de machine learning para prever atrasos em voos superiores a 15 minutos. Utilizamos dados históricos de voos, companhias aéreas e aeroportos para treinar e avaliar diferentes algoritmos de classificação.

## 🎯 Objetivos

- 📊 **Análise Exploratória**: Identificar padrões e fatores que influenciam atrasos em voos
- 🤖 **Modelagem Preditiva**: Implementar e comparar modelos de machine learning (Regressão Logística, KNN e Random Forest)
- 📈 **Visualização Interativa**: Criar interface web com Streamlit para exploração dos dados e resultados
- 🔍 **Insights de Negócio**: Gerar insights acionáveis sobre fatores de atraso

## 🗂️ Estrutura do Projeto

```
📁 supervised_machine_learning/
├── 📄 flight_delays.py       # Aplicação principal Streamlit
├── 📄 requirements.txt       # Dependências do projeto
├── 📄 README.md             # Documentação do projeto
├── 📁 data/                 # Datasets (gerados automaticamente)
│   ├── flights.csv          # Dados de voos
│   ├── airlines.csv         # Dados de companhias aéreas
│   └── airports.csv         # Dados de aeroportos
└── 📁 modules/
    └── load_data.py         # Módulo para carregamento de dados
```

## 📊 Datasets Utilizados

- **🛫 Flights Dataset**: Dados detalhados de voos incluindo horários, atrasos, rotas e métricas operacionais
- **🏢 Airlines Dataset**: Informações sobre companhias aéreas e códigos IATA
- **🏢 Airports Dataset**: Dados de aeroportos com localização geográfica e códigos

*Os dados são automaticamente baixados do Kaggle na primeira execução.*

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- pip (gerenciador de pacotes Python)

### Passo a Passo

1. **Clone o repositório**
   ```bash
   git clone https://github.com/Silas-ER/supervised_machine_learning.git
   cd supervised_machine_learning
   ```

2. **Crie um ambiente virtual**
   ```bash
   virtualenv venv
   ```

3. **Ative o ambiente virtual**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

5. **Execute a aplicação**
   ```bash
   streamlit run main.py
   ```

6. **Acesse no navegador**
   ```
   🌐 Local URL: http://localhost:8501
   ```

## 🔧 Funcionalidades

### 📋 Aba "Dados"
- Exploração interativa dos 3 datasets
- Descrição detalhada de todas as colunas
- Estatísticas descritivas e informações estruturais

### 📊 Aba "Análise Exploratória"
- **Distribuições**: Histogramas de atrasos na chegada e decolagem
- **Correlações**: Análise da relação entre atraso na partida e chegada
- **Análises Segmentadas**: 
  - Atrasos por companhia aérea
  - Atrasos por aeroporto de origem
  - Atrasos por dia da semana
  - Atrasos por hora do dia
- **Impacto da Distância**: Análise do efeito da distância do voo nos atrasos
- **Mapa de Correlação**: Heatmap entre variáveis numéricas

### 🤖 Aba "Modelo"
- **Preparação dos Dados**: Criação da variável target e limpeza
- **Modelos Implementados**:
  - 📈 Regressão Logística
  - 🔍 K-Nearest Neighbors (KNN)
  - 🌲 Random Forest
- **Métricas de Avaliação**: Precision, Recall, F1-Score e Acurácia
- **Visualizações**: Matrizes de confusão e importância das features
- **Comparação**: Análise comparativa entre os modelos
- **Novas Features**: Inclusão de variáveis como `TAXI_IN`, `DIVERTED`, `ELAPSED_TIME` e `SCHEDULED_TIME` para melhorar a performance dos modelos

## 📈 Tecnologias Utilizadas

- **🐍 Python**: Linguagem principal
- **🌊 Streamlit**: Framework para interface web
- **🐼 Pandas**: Manipulação de dados
- **📊 Matplotlib/Seaborn**: Visualização de dados
- **🤖 Scikit-learn**: Machine learning
- **📥 KaggleHub**: Download automático de datasets

## 📊 Principais Insights

- 🚀 **Forte correlação** entre atraso na partida e chegada
- 📅 **Variação sazonal** nos atrasos por dia da semana
- 🕐 **Padrões horários** com maior probabilidade de atraso
- ✈️ **Diferenças significativas** entre companhias aéreas
- 🛫 **Aeroportos específicos** com maior tendência a atrasos
- 🌲 **Random Forest** como modelo mais eficaz para prever atrasos, com alta precisão e recall

## 👥 Contribuição

Projeto desenvolvido como parte do curso IMD3002 - Aprendizado de Máquina Supervisionado.

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos.

---

⚡ **Dica**: Execute `streamlit run main.py` e explore a aplicação interativa para uma experiência completa!
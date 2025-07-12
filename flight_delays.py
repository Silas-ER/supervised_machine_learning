import datetime, warnings, scipy
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from collections import OrderedDict
from matplotlib.gridspec import GridSpec

from sklearn import metrics, linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#########################################################
################## FUNÇÕES UTILIZADAS ###################
#########################################################

def formatar_hora(hora):
    """
    Converte um inteiro HHMM (ex.: 5 → 0005, 1230 → 12:30) em datetime.time.
    Se o valor for 2400, considera como 00:00. Valores nulos retornam NaN.
    """
    if pd.isnull(hora):
        return np.nan
    
    if hora == 2400:
        hora = 0

    hora_str = "{:04d}".format(int(hora))
    hora = datetime.time(int(hora_str[0:2]), int(hora_str[2:4]))
    return hora

def combinar_data_hora(x):
    """
    Combina uma data (datetime.date) e uma hora (datetime.time) em um datetime.datetime.
    """
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0],x[1])

def criar_horario_voo(df, col):
    """
    Cria uma coluna de horário de voo a partir de uma coluna de data e uma coluna de hora.
    """
    lista = []
    for index, cols in df[['DATE', col]].iterrows():
        if pd.isnull(cols[1]):
            lista.append(np.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0, 0)
            lista.append(combinar_data_hora(cols))
        else:
            cols[1] = formatar_hora(cols[1])
            lista.append(combinar_data_hora(cols))
    
    return pd.Series(lista)

#########################################################
### MODELO DE PREDICAO DE ATRASOS DE VOO NA DECOLAGEM ###
#########################################################

"""
O objetivo principal não é alcançar a maior precisão preditiva possível, 
mas ilustrar as etapas-chave envolvidas na construção de tal modelo
"""

flights = pd.read_csv('data/flights.csv', low_memory=False)
#print("flights shape:", flights.shape)
#print("flights columns:")
#print(flights.columns.tolist())
#print(flights.head())
#print(flights.info())
#print(flights.isnull().sum())

flights['IS_DELAYED'] = (flights['ARRIVAL_DELAY'] >= 15).astype(int)
flights['IS_DELAYED'].value_counts(normalize=True) * 100

# Retirar colunas irrelevantes para o modelo
flights_cleaned = flights.drop(columns=[
    'TAIL_NUMBER', 
    'CANCELLATION_REASON',  
    'AIR_SYSTEM_DELAY', 
    'SECURITY_DELAY', 
    'AIRLINE_DELAY', 
    'LATE_AIRCRAFT_DELAY', 
    'WEATHER_DELAY'  
])

flights_cleaned['DEP_HOUR'] = flights_cleaned['SCHEDULED_DEPARTURE'] // 100
flights_cleaned['DEP_HOUR'] = flights_cleaned['DEP_HOUR'].apply(lambda x: min(x, 23)) 
flights_cleaned['IS_WEEKEND'] = flights_cleaned['DAY_OF_WEEK'].isin([6, 7]).astype(int)

flights_cleaned = flights_cleaned.drop(columns=[
    'DEPARTURE_TIME', 
    'WHEELS_OFF', 
    'WHEELS_ON',
    'TAXI_IN', 
    'TAXI_OUT', 
    'ARRIVAL_TIME', 
    'ARRIVAL_DELAY',
    'ELAPSED_TIME', 
    'AIR_TIME',
    'YEAR', 
    'FLIGHT_NUMBER',
    'SCHEDULED_DEPARTURE',
    'SCHEDULED_ARRIVAL', 
    'DEPARTURE_DELAY'
], errors='ignore')

print(flights_cleaned.columns.tolist())

flights_encoded = pd.get_dummies(flights_cleaned, columns=['AIRLINE'], drop_first=True)

"""
# Select only numeric features
numeric_features = flights.select_dtypes(include=['int64', 'float64'])

# Compute correlation matrix
corr_matrix = numeric_features.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()"""

# Encode ORIGIN_AIRPORT based on mean delay
origin_delay_rate = flights_encoded.groupby('ORIGIN_AIRPORT')['IS_DELAYED'].mean()
flights_encoded['ORIGIN_ENCODED'] = flights_encoded['ORIGIN_AIRPORT'].map(origin_delay_rate)

# Encode DESTINATION_AIRPORT based on mean delay
dest_delay_rate = flights_encoded.groupby('DESTINATION_AIRPORT')['IS_DELAYED'].mean()
flights_encoded['DEST_ENCODED'] = flights_encoded['DESTINATION_AIRPORT'].map(dest_delay_rate)

# Drop original string columns
flights_encoded = flights_encoded.drop(columns=['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])

flights_encoded.isnull().sum().sort_values(ascending=False).head(10)

flights_encoded['ORIGIN_ENCODED'] = flights_encoded['ORIGIN_ENCODED'].fillna(flights_encoded['ORIGIN_ENCODED'].mean())
flights_encoded['DEST_ENCODED'] = flights_encoded['DEST_ENCODED'].fillna(flights_encoded['DEST_ENCODED'].mean())
# Fill any remaining numeric nulls (e.g., SCHEDULED_TIME) with median
flights_encoded = flights_encoded.fillna(flights_encoded.median(numeric_only=True))

#########################################################
################ TREINANDO O MODELO BASE ################
#########################################################

X = flights_encoded.drop(columns=['IS_DELAYED'])
y = flights_encoded['IS_DELAYED']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lgbm_model = lgb.LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)
y_pred = lgbm_model.predict(X_val)
print("✅ Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("LightGBM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#########################################################
#################### OTIMIZANDO MODELO ##################
#########################################################

lgbm_model = lgb.LGBMClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, -1],  # -1 means no limit
    'learning_rate': [0.1, 0.05],
    'num_leaves': [31, 50],
    'min_child_samples': [20, 50]
}
grid_search = GridSearchCV(
    estimator=lgbm_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,  # Use 3-fold CV due to dataset size
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_lgbm = LGBMClassifier(
    learning_rate=0.1,
    max_depth=-1,
    min_child_samples=20,
   n_estimators=200,
    num_leaves=50,
    random_state=42
)

best_lgbm.fit(X_train, y_train)

# Predict and evaluate
y_pred = best_lgbm.predict(X_val)



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Accuracy
print("✅ Final Accuracy:", accuracy_score(y_val, y_pred))

# Detailed report
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Final LightGBM Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create and train XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict on validation set
y_pred_xgb = xgb_model.predict(X_val)

# Evaluate
print("✅ XGBoost Accuracy:", accuracy_score(y_val, y_pred_xgb))
print("\nXGBoost Classification Report:\n", classification_report(y_val, y_pred_xgb))

# Confusion matrix
cm_xgb = confusion_matrix(y_val, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Purples")
plt.title("Confusion Matrix – XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Predict using the validation set
test_preds = grid_search.best_estimator_.predict(X_val)

# 2. Accuracy score
print("✅ Final Validation Accuracy:", accuracy_score(y_val, test_preds))

# 3. Classification report
print("\n📋 Classification Report:\n")
print(classification_report(y_val, test_preds))

# 4. Confusion matrix
cm = confusion_matrix(y_val, test_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – LightGBM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



import pandas as pd

def predict_delay(model, airline, origin, dest, month, day, day_of_week, dep_hour, distance, cancelled, diverted):
    # Build input dataframe (1 row with feature names)
    input_data = pd.DataFrame([{
        'AIRLINE': airline,
        'ORIGIN_AIRPORT': origin,
        'DESTINATION_AIRPORT': dest,
        'MONTH': month,
        'DAY': day,
        'DAY_OF_WEEK': day_of_week,
        'DEP_HOUR': dep_hour,
        'DISTANCE': distance,
        'CANCELLED': cancelled,
        'DIVERTED': diverted
    }])

    # Apply same encoding (match your preprocessing!)
    input_data['AIRLINE'] = airline_encoder.transform(input_data['AIRLINE'])
    input_data['ORIGIN_ENCODED'] = origin_encoder.transform(input_data['ORIGIN_AIRPORT'])
    input_data['DEST_ENCODED'] = dest_encoder.transform(input_data['DESTINATION_AIRPORT'])

    # Drop unused categorical columns (same as model input)
    input_data = input_data.drop(columns=['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])

    # Match model columns
    input_data = input_data[X_train.columns]  # Ensures order is correct

    # Make prediction
    prediction = model.predict(input_data)[0]

    return '🟥 Delayed' if prediction == 1 else '🟩 On-Time'

airline_encoder = LabelEncoder()
airline_encoder.fit(flights_cleaned['AIRLINE'])

origin_encoder = LabelEncoder()
origin_encoder.fit(flights_cleaned['ORIGIN_AIRPORT'])

dest_encoder = LabelEncoder()
dest_encoder.fit(flights_cleaned['DESTINATION_AIRPORT'])



sample = X_train.sample(1).copy()

# Simulate a new flight by changing values
sample['MONTH'] = 6
sample['DAY'] = 15
sample['DAY_OF_WEEK'] = 1
sample['DEP_HOUR'] = 14
sample['DISTANCE'] = 2475
sample['AIRLINE_AS'] = 1
sample['AIRLINE_DL'] = 0
sample['CANCELLED'] = 0
sample['DIVERTED'] = 0

# Predict using final model
pred = grid_search.best_estimator_.predict(sample)[0]
print("🟥 Delayed" if pred == 1 else "🟩 On-Time")
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import dump, load
import streamlit as st
from skimage import io

def pred_knn(X_test, X_test_norm, X_test_stand):
    model_Knn0 = load('joblib_model_Knn0.joblib')
    model_Knn1 = load('joblib_model_Knn1.joblib')
    model_Knn2 = load('joblib_model_Knn2.joblib')

    acc = []
    f1_samples = []
    precision = []
    recall = []

    # Prediccion
    pred = model_Knn0.predict(X_test)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))
    # Prediccion
    pred = model_Knn1.predict(X_test_norm)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))
    # Prediccion
    pred = model_Knn2.predict(X_test_stand)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))

    accuracy = pd.DataFrame({'Accuracy':acc},index=['Original','Normalized','Standardized'])
    f1 = pd.DataFrame({'F1 samples':f1_samples},index=['Original','Normalized','Standardized'])
    pre = pd.DataFrame({'Precision':precision},index=['Original','Normalized','Standardized'])
    rec = pd.DataFrame({'Recall':recall},index=['Original','Normalized','Standardized'])

    scores = accuracy.join(f1, lsuffix='_caller', rsuffix='_other')
    scores = scores.join(pre, lsuffix='_caller', rsuffix='_other')
    scores = scores.join(rec, lsuffix='_caller', rsuffix='_other')
    return scores

def pred_dt(X_test, X_test_norm, X_test_stand):
    model_DT0 = load('joblib_model_DT0.joblib')
    model_DT1 = load('joblib_model_DT1.joblib')
    model_DT2 = load('joblib_model_DT2.joblib')

    acc = []
    f1_samples = []
    precision = []
    recall = []

    # Prediccion
    pred = model_DT0.predict(X_test)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))
    # Prediccion
    pred = model_DT1.predict(X_test_norm)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))
    # Prediccion
    pred = model_DT2.predict(X_test_stand)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))

    accuracy = pd.DataFrame({'Accuracy':acc},index=['Original','Normalized','Standardized'])
    f1 = pd.DataFrame({'F1 samples':f1_samples},index=['Original','Normalized','Standardized'])
    pre = pd.DataFrame({'Precision':precision},index=['Original','Normalized','Standardized'])
    rec = pd.DataFrame({'Recall':recall},index=['Original','Normalized','Standardized'])

    scores = accuracy.join(f1, lsuffix='_caller', rsuffix='_other')
    scores = scores.join(pre, lsuffix='_caller', rsuffix='_other')
    scores = scores.join(rec, lsuffix='_caller', rsuffix='_other')
    return scores

def pred_rf(X_test, X_test_norm, X_test_stand):
    model_RF0 = load('joblib_model_RF0.joblib')
    model_RF1 = load('joblib_model_RF1.joblib')
    model_RF2 = load('joblib_model_RF2.joblib')

    acc = []
    f1_samples = []
    precision = []
    recall = []

    # Prediccion
    pred = model_RF0.predict(X_test)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))
    # Prediccion
    pred = model_RF1.predict(X_test_norm)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))
    # Prediccion
    pred = model_RF2.predict(X_test_stand)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))

    accuracy = pd.DataFrame({'Accuracy':acc},index=['Original','Normalized','Standardized'])
    f1 = pd.DataFrame({'F1 samples':f1_samples},index=['Original','Normalized','Standardized'])
    pre = pd.DataFrame({'Precision':precision},index=['Original','Normalized','Standardized'])
    rec = pd.DataFrame({'Recall':recall},index=['Original','Normalized','Standardized'])

    scores = accuracy.join(f1, lsuffix='_caller', rsuffix='_other')
    scores = scores.join(pre, lsuffix='_caller', rsuffix='_other')
    scores = scores.join(rec, lsuffix='_caller', rsuffix='_other')
    return scores

def pred_xgb(X_test, X_test_norm, X_test_stand):
    model_XGB0 = load('joblib_model_XGB0.joblib')
    model_XGB1 = load('joblib_model_XGB1.joblib')
    model_XGB2 = load('joblib_model_XGB2.joblib')

    acc = []
    f1_samples = []
    precision = []
    recall = []

    # Prediccion
    pred = model_XGB0.predict(X_test)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))
    # Prediccion
    pred = model_XGB1.predict(X_test_norm)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))
    # Prediccion
    pred = model_XGB2.predict(X_test_stand)
    # Accuracy
    acc.append(np.sqrt(accuracy_score(y_test,pred)))
    # F1 samples
    f1_samples.append(np.sqrt(f1_score(y_test,pred,average='micro')))
    # Precision
    precision.append(np.sqrt(precision_score(y_test,pred,average='micro')))
    # Recall
    recall.append(np.sqrt(recall_score(y_test,pred,average='micro')))

    accuracy = pd.DataFrame({'Accuracy':acc},index=['Original','Normalized','Standardized'])
    f1 = pd.DataFrame({'F1 samples':f1_samples},index=['Original','Normalized','Standardized'])
    pre = pd.DataFrame({'Precision':precision},index=['Original','Normalized','Standardized'])
    rec = pd.DataFrame({'Recall':recall},index=['Original','Normalized','Standardized'])

    scores = accuracy.join(f1, lsuffix='_caller', rsuffix='_other')
    scores = scores.join(pre, lsuffix='_caller', rsuffix='_other')
    scores = scores.join(rec, lsuffix='_caller', rsuffix='_other')
    return scores

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: justify; color: Darkblue; font-size:40px'>PREDICCION DEL GENERO DE CANCIONES DE SPOTIFY</h1>", unsafe_allow_html=True)
intro =  """
 Este proyecto consiste en el la implementacion de algoritmos de clasificacion para encontrar el mejor modelo que realice la prediccion del genero de una cancion dadas sus caracteristicas musicales.\n
 """
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:20px'>{intro}</p>", unsafe_allow_html=True)
box = st.columns(3)
with box[0]:
    img1 = io.imread("Spotify_logo.png")
    st.image(img1, width = 300)
with box[1]:
    st.header("**Librerias**")
    dataframeinfo ="""
        - Pandas\n
        - Numpy\n
        - Seaborn\n
        - Matplotlib\n
        - Joblib\n
        """
    st.markdown(dataframeinfo, unsafe_allow_html=True,)
with box[2]:
    st.header("**Modelos**")
    dataframeinfo ="""
        - KNeighborsClassifier\n
        - DecisionTreeClassifier\n
        - RandomForestClassifier\n
        - XGBClassifier\n
        """
    st.markdown(dataframeinfo, unsafe_allow_html=True,)

st.header("**Dataset**")
dataframeinfo ="""
        Dataset de canciones de Spotify obtenido de Kaggle
        """
st.markdown(dataframeinfo, unsafe_allow_html=True,)

df_dep = pd.read_csv('spotify_all_genres_tracks_raw.csv')
st.dataframe(df_dep.head(5))

st.header("**Limpieza de columnas irrelevantes**")
del_col = ['track_id','track_name','artist_name','duration_ms','playlist_url','album','album_cover','artist_genres','playlist_name']
df_dep.drop(del_col, axis=1, inplace=True)
st.dataframe(df_dep.head(5))
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:15px'>Columnas eliminadas: {del_col}</p>", unsafe_allow_html=True)

st.header("**Feature Engineering**")
feature = """
Tecnicas de Feature Engineeing:\n
- Encoding\n
- Sacaling\n
    - Normalization\n
    - Standarization\n
"""
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:20px'>{feature} </p>", unsafe_allow_html=True)
df_dep['genre'] = OrdinalEncoder().fit_transform(df_dep['genre'].to_numpy().reshape(-1,1)).astype(int)

st.header("**Feature Selection**")
sel = VarianceThreshold(threshold=(.95 * (1 - .95)))
sel.fit_transform(df_dep)
sel_col = sel.get_feature_names_out()
seleccion = """
Se utilizo VarianceThreshold al 95% para descartar los features cuya varianza es muy baja \n
"""
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:20px'>{seleccion}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:20px'>Columnas seleccionadas: {del_col}</p>", unsafe_allow_html=True)
df_dep.drop(['danceability','speechiness','liveness'], axis=1, inplace=True)
st.dataframe(df_dep.head(5))

cmap = sns.diverging_palette(230, 20, as_cmap=True)
plt.figure(figsize=(15, 10))
p = sns.heatmap(df_dep.corr(), annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap=cmap)
plt.title("Correlation Matrix", fontweight='bold', fontsize='large')
st.pyplot(p.get_figure())

correlacion = df_dep.corr()["genre"]
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:20px'>Correlacion de los features con el genero:</p>", unsafe_allow_html=True)
st.table(correlacion)

img2 = io.imread("df_hist.png")
st.image(img2, width = 950)

y = df_dep['genre'].to_numpy().reshape(-1,1)
X = df_dep.drop('genre',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
norm = """Features normalizados"""
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:20px'>{norm} </p>", unsafe_allow_html=True)
st.dataframe(X_test_norm)

X_train_stand = X_train.copy()
X_test_stand = X_test.copy()
num_cols = ['track_popularity','artist_popularity','key','loudness','tempo','time_signature']
for i in num_cols:
    scale = StandardScaler().fit(X_train_stand[[i]])
    X_train_stand[i] = scale.transform(X_train_stand[[i]])
    X_test_stand[i] = scale.transform(X_test_stand[[i]])
stand = """Features estandarizados y escalados"""
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:20px'>{stand} </p>", unsafe_allow_html=True)
st.dataframe(X_test_stand)

st.header("**KNeighborsClassifier**")
knn = pred_knn(X_test, X_test_norm, X_test_stand)
st.dataframe(knn)

st.header("**DecisionTreeClassifier**")
dt = pred_dt(X_test, X_test_norm, X_test_stand)
st.dataframe(dt)

st.header("**RandomForestClassifier**")
rf = pred_rf(X_test, X_test_norm, X_test_stand)
st.dataframe(rf)

st.header("**XGBClassifier**")
xgb_c = pred_xgb(X_test, X_test_norm, X_test_stand)
st.dataframe(xgb_c)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import scipy.cluster.hierarchy as sch
import warnings
import os
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Spotify ML Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema personalizado
def set_custom_theme():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1DB954;
        border-bottom: 2px solid #1DB954;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #191414;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1DB954;
        margin: 0.5rem 0;
    }
    .stButton button {
        background-color: #1DB954;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1ed760;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

set_custom_theme()

# Título principal
st.markdown('<h1 class="main-header">🎵 Spotify Music Analysis</h1>', unsafe_allow_html=True)
st.markdown("### Machine Learning con Dataset Spotify - KNN & Clustering Jerárquico")

# Función mejorada para cargar datos
@st.cache_data
def load_data():
    """Cargar datos buscando en múltiples ubicaciones"""
    csv_files = []
    
    # Buscar archivos CSV en el directorio actual
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            csv_files.append(file)
    
    # Buscar en la carpeta archive
    archive_path = './archive'
    if os.path.exists(archive_path):
        for file in os.listdir(archive_path):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(archive_path, file))
    
    # Mostrar archivos encontrados
    if csv_files:
        st.sidebar.info(f"📁 Archivos CSV encontrados: {len(csv_files)}")
        for csv_file in csv_files:
            st.sidebar.write(f"   📄 {csv_file}")
    
    # Intentar cargar archivos en orden de preferencia
    preferred_files = [
        'spotify_data_clean.csv',
        'track_data_final.csv', 
        'spotify_songs.csv',
        'tracks.csv',
        'songs.csv',
        'data.csv'
    ]
    
    for preferred_file in preferred_files:
        # Buscar en directorio raíz
        if os.path.exists(preferred_file):
            try:
                df = pd.read_csv(preferred_file)
                st.sidebar.success(f"✅ Cargado: {preferred_file}")
                return df
            except Exception as e:
                st.sidebar.error(f"❌ Error cargando {preferred_file}: {e}")
        
        # Buscar en archive
        archive_file = os.path.join(archive_path, preferred_file)
        if os.path.exists(archive_file):
            try:
                df = pd.read_csv(archive_file)
                st.sidebar.success(f"✅ Cargado: {archive_file}")
                return df
            except Exception as e:
                st.sidebar.error(f"❌ Error cargando {archive_file}: {e}")
    
    # Si hay archivos CSV pero no los preferidos, cargar el primero
    if csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            st.sidebar.success(f"✅ Cargado: {csv_files[0]}")
            return df
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando {csv_files[0]}: {e}")
    
    # Si no hay archivos CSV
    st.sidebar.error("❌ No se encontraron archivos CSV")
    return None

@st.cache_data
def preprocess_data(df):
    """Preprocesamiento MÍNIMO - mantener todos los datos posibles"""
    df_clean = df.copy()
    
    # Mostrar información inicial
    st.sidebar.write(f"📊 Datos originales: {df_clean.shape}")
    
    # ESTRATEGIA: Usar SOLO columnas que ya son numéricas
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    st.sidebar.write(f"🔢 Columnas numéricas encontradas: {len(numeric_cols)}")
    
    # Si no hay suficientes columnas numéricas, mostrar advertencia pero continuar
    if len(numeric_cols) < 2:
        st.sidebar.warning("⚠️ Pocas columnas numéricas. Usando todas las disponibles.")
        # Intentar convertir algunas columnas comunes
        potential_numeric = ['popularity', 'duration_ms', 'tempo', 'danceability', 'energy', 
                           'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                           'liveness', 'valence', 'key', 'mode', 'time_signature']
        
        for col in potential_numeric:
            if col in df_clean.columns and col not in numeric_cols:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    if df_clean[col].notna().sum() > 0:
                        numeric_cols.append(col)
                except:
                    pass
    
    # Usar SOLO las columnas numéricas
    df_clean = df_clean[numeric_cols]
    
    st.sidebar.write(f"🎯 Usando {len(numeric_cols)} características numéricas")
    
    # ESTRATEGIA CONSERVADORA: Imputar NaN en lugar de eliminar
    initial_nans = df_clean.isnull().sum().sum()
    
    if initial_nans > 0:
        st.sidebar.info(f"🔄 Imputando {initial_nans} valores NaN...")
        # Imputar valores faltantes con la mediana (más robusta que la media)
        imputer = SimpleImputer(strategy='median')
        df_imputed = imputer.fit_transform(df_clean)
        df_clean = pd.DataFrame(df_imputed, columns=df_clean.columns)
    
    # Verificar que tenemos datos
    if len(df_clean) == 0:
        st.sidebar.error("❌ No hay datos después del procesamiento")
        return None
    
    st.sidebar.success(f"✅ Datos finales: {df_clean.shape}")
    st.sidebar.info(f"🎯 Características: {list(df_clean.columns)}")
    
    return df_clean

# Sidebar
st.sidebar.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png", 
                 width=200)
st.sidebar.title("Configuración")

# Cargar datos
df_raw = load_data()

if df_raw is not None:
    st.sidebar.success(f"✅ Dataset cargado: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")
    
    # Mostrar información detallada del dataset
    with st.sidebar.expander("📊 Info Detallada del Dataset"):
        st.write(f"**Filas:** {df_raw.shape[0]}")
        st.write(f"**Columnas:** {df_raw.shape[1]}")
        st.write("**Todas las columnas:**")
        for col in df_raw.columns:
            dtype = str(df_raw[col].dtype)
            null_count = df_raw[col].isnull().sum()
            st.write(f"- {col} ({dtype}) - Nulos: {null_count}")
        
        st.write("**Columnas numéricas:**")
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            st.write(f"- {col}")
else:
    st.error("No se pudo cargar el dataset.")
    st.stop()

# Preprocesar datos (mínimo)
df_processed = preprocess_data(df_raw)

if df_processed is None or len(df_processed) == 0:
    st.error("No se pudieron procesar los datos. Usando datos originales...")
    # Usar solo columnas numéricas del dataset original
    df_processed = df_raw.select_dtypes(include=[np.number])
    if len(df_processed) == 0:
        st.error("No hay columnas numéricas en el dataset.")
        st.stop()

# Sidebar - Selección de características
st.sidebar.subheader("🔧 Configuración de Modelos")

# Navegación principal
app_mode = st.sidebar.radio(
    "Selecciona el modo:",
    ["🏠 Inicio", "📊 Análisis Exploratorio", "🎯 Modelo Supervisado", "🔍 Modelo No Supervisado", "📤 Exportación"]
)

# Página de Inicio
if app_mode == "🏠 Inicio":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">Bienvenido al Analizador de Música Spotify</h2>', unsafe_allow_html=True)
        st.markdown("""
        Esta aplicación utiliza machine learning para analizar tu dataset de Spotify y descubrir patrones interesantes en la música.
        
        ### 🎯 **Modelo Supervisado: K-Vecinos Cercanos (KNN)**
        - Clasificación de canciones basada en características musicales
        - Predicción de popularidad o género
        - Métricas: Accuracy, Precision, Recall, F1-Score
        
        ### 🔍 **Modelo No Supervisado: Clustering Jerárquico**
        - Agrupamiento de canciones similares
        - Descubrimiento de patrones ocultos
        - Métricas: Silhouette Score, Davies-Bouldin Index
        """)
        
        # Mostrar estadísticas básicas
        st.info(f"""
        ### 📊 **Características del Dataset**
        - **Total de canciones:** {df_raw.shape[0]}
        - **Características para ML:** {df_processed.shape[1]}
        - **Dataset original:** {df_raw.shape[1]} columnas
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/174/174872.png", width=200)
        st.markdown("### 🚀 Comienza explorando:")
        st.markdown("""
        1. 📊 **Análisis Exploratorio**: Visualiza los datos
        2. 🎯 **Modelo Supervisado**: Entrena KNN
        3. 🔍 **Modelo No Supervisado**: Descubre clusters
        4. 📤 **Exportación**: Descarga resultados
        """)
    
    # Vista previa de datos
    st.markdown('<h3 class="section-header">Vista Previa del Dataset</h3>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Datos Originales", "Datos Procesados", "Estadísticas"])
    
    with tab1:
        st.write("**Dataset Original Completo:**")
        st.dataframe(df_raw, use_container_width=True)
        
        st.write("**Información del Dataset:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Filas:** {df_raw.shape[0]}")
            st.write(f"**Columnas:** {df_raw.shape[1]}")
            st.write(f"**Memoria:** {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col2:
            st.write(f"**Valores nulos:** {df_raw.isnull().sum().sum()}")
            st.write(f"**Columnas numéricas:** {len(df_raw.select_dtypes(include=[np.number]).columns)}")
            st.write(f"**Columnas texto:** {len(df_raw.select_dtypes(include=['object']).columns)}")
    
    with tab2:
        st.write("**Datos para Machine Learning (solo columnas numéricas):**")
        st.dataframe(df_processed, use_container_width=True)
        st.success(f"✅ {df_processed.shape[0]} filas listas para ML con {df_processed.shape[1]} características")
        
        with st.expander("🔍 Ver detalles de características"):
            for col in df_processed.columns:
                st.write(f"**{col}**")
                st.write(f"  - Tipo: {df_processed[col].dtype}")
                st.write(f"  - Rango: {df_processed[col].min():.2f} a {df_processed[col].max():.2f}")
                st.write(f"  - Media: {df_processed[col].mean():.2f}")
                st.write("---")
    
    with tab3:
        st.write("**Estadísticas Descriptivas:**")
        st.dataframe(df_processed.describe(), use_container_width=True)

# Análisis Exploratorio
elif app_mode == "📊 Análisis Exploratorio":
    st.markdown('<h2 class="section-header">📊 Análisis Exploratorio de Datos</h2>', unsafe_allow_html=True)
    
    if len(df_processed.columns) < 2:
        st.error("Se necesitan al menos 2 características para el análisis exploratorio")
        st.stop()
    
    # Selector de características para visualización
    numeric_cols = df_processed.columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("Característica para eje X:", numeric_cols, index=0)
    with col2:
        y_feature = st.selectbox("Característica para eje Y:", numeric_cols, index=min(1, len(numeric_cols)-1))
    
    # Gráficos
    tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Distribuciones", "Correlaciones"])
    
    with tab1:
        fig = px.scatter(df_processed, x=x_feature, y=y_feature, 
                        title=f'{y_feature} vs {x_feature}',
                        opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            feature_hist = st.selectbox("Selecciona característica:", numeric_cols, index=0)
            fig_hist = px.histogram(df_processed, x=feature_hist, 
                                  title=f'Distribución de {feature_hist}',
                                  nbins=30)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(df_processed, y=feature_hist, 
                           title=f'Box Plot de {feature_hist}')
            st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        # Matriz de correlación
        corr_matrix = df_processed.corr()
        fig_corr = px.imshow(corr_matrix, 
                           title='Matriz de Correlación',
                           color_continuous_scale='RdBu_r',
                           aspect='auto')
        st.plotly_chart(fig_corr, use_container_width=True)

# Modelo Supervisado - KNN
elif app_mode == "🎯 Modelo Supervisado":
    st.markdown('<h2 class="section-header">🎯 Modelo Supervisado - K-Vecinos Cercanos</h2>', unsafe_allow_html=True)
    
    if len(df_processed) < 10:
        st.error("Se necesitan al menos 10 filas para entrenar el modelo KNN")
        st.stop()
    
    # Configuración del modelo
    st.sidebar.subheader("⚙️ Configuración KNN")
    
    # Crear variable target automáticamente
    df_supervised = df_processed.copy()
    
    # Estrategia para crear target: usar la primera columna dividida en 2 categorías
    if len(df_supervised.columns) > 0:
        target_feature = df_supervised.columns[0]
        threshold = df_supervised[target_feature].median()
        df_supervised['target'] = (df_supervised[target_feature] > threshold).astype(int)
        class_names = [f'Bajo {target_feature}', f'Alto {target_feature}']
        st.sidebar.info(f"🎯 Target creado desde: {target_feature}")
    else:
        st.error("No hay características disponibles para crear el target")
        st.stop()
    
    # Selección de características (excluyendo el target)
    available_features = [col for col in df_supervised.columns if col != 'target']
    
    if not available_features:
        st.error("No hay características disponibles para el modelo")
        st.stop()
    
    selected_features = st.sidebar.multiselect(
        "Selecciona características para el modelo:",
        available_features,
        default=available_features[:min(3, len(available_features))]
    )
    
    if not selected_features:
        st.error("Selecciona al menos una característica para el modelo")
        st.stop()
    
    n_neighbors = st.sidebar.slider("Número de vecinos (k):", 1, 15, 5)
    test_size = st.sidebar.slider("Tamaño conjunto de prueba:", 0.1, 0.4, 0.2, 0.05)
    
    # Preparar datos
    X = df_supervised[selected_features]
    y = df_supervised['target']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Entrenar modelo
    try:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        
        # Predicciones
        y_pred = knn.predict(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Mostrar métricas
        st.subheader("📊 Métricas de Evaluación")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        
        # Matriz de confusión
        st.subheader("📋 Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, 
                          text_auto=True,
                          color_continuous_scale='Blues',
                          title='Matriz de Confusión',
                          labels=dict(x="Predicho", y="Real", color="Cantidad"))
        fig_cm.update_layout(xaxis=dict(tickvals=[0,1], ticktext=class_names),
                           yaxis=dict(tickvals=[0,1], ticktext=class_names))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Prueba interactiva
        st.subheader("🧪 Prueba Interactiva")
        
        st.write("Ingresa valores para predecir la clase:")
        
        # Crear sliders dinámicamente
        input_values = {}
        cols = st.columns(3)
        
        for i, feature in enumerate(selected_features):
            with cols[i % 3]:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                mean_val = float(X[feature].mean())
                
                input_values[feature] = st.slider(
                    f"{feature}",
                    min_val, max_val, mean_val,
                    help=f"Rango: {min_val:.2f} - {max_val:.2f}"
                )
        
        # Predicción
        if st.button("🎵 Predecir Clase", type="primary"):
            input_array = np.array([[input_values[feat] for feat in selected_features]])
            prediction = knn.predict(input_array)[0]
            prediction_proba = knn.predict_proba(input_array)[0]
            
            st.success(f"**Clase predicha:** {class_names[prediction]}")
            
            # Mostrar probabilidades
            prob_df = pd.DataFrame({
                'Clase': class_names,
                'Probabilidad': prediction_proba
            })
            
            fig_proba = px.bar(prob_df, x='Clase', y='Probabilidad', 
                              title='Probabilidades de Predicción',
                              color='Probabilidad',
                              color_continuous_scale='Viridis')
            st.plotly_chart(fig_proba, use_container_width=True)
        
        # Guardar para exportación
        st.session_state['knn_model'] = knn
        st.session_state['knn_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        st.session_state['knn_features'] = selected_features
        st.session_state['class_names'] = class_names
        
        st.success("✅ Modelo KNN entrenado exitosamente!")
        
    except Exception as e:
        st.error(f"Error entrenando el modelo: {e}")

# Modelo No Supervisado - Clustering Jerárquico
elif app_mode == "🔍 Modelo No Supervisado":
    st.markdown('<h2 class="section-header">🔍 Modelo No Supervisado - Clustering Jerárquico</h2>', unsafe_allow_html=True)
    
    if len(df_processed) < 5:
        st.error("Se necesitan al menos 5 filas para clustering")
        st.stop()
    
    if len(df_processed.columns) < 2:
        st.error("Se necesitan al menos 2 características para clustering")
        st.stop()
    
    # Configuración
    st.sidebar.subheader("⚙️ Configuración Clustering")
    
    # Selección de características
    available_features = df_processed.columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Características para clustering:",
        available_features,
        default=available_features[:min(4, len(available_features))]
    )
    
    if not selected_features:
        st.error("Selecciona al menos una característica para clustering")
        st.stop()
    
    n_clusters = st.sidebar.slider("Número de clusters:", 2, 8, 3)
    linkage_method = st.sidebar.selectbox(
        "Método de linkage:",
        ["ward", "complete", "average", "single"]
    )
    
    # Preparar datos
    X = df_processed[selected_features].values
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar clustering jerárquico
    try:
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric='euclidean'
        )
        cluster_labels = hierarchical.fit_predict(X_scaled)
        
        # Métricas de calidad
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        db_index = davies_bouldin_score(X_scaled, cluster_labels)
        
        # Mostrar métricas
        st.subheader("📊 Métricas de Calidad de Clustering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
        with col2:
            st.metric("Davies-Bouldin Index", f"{db_index:.4f}")
        
        # Interpretación de métricas
        st.info(f"""
        **Interpretación:**
        - **Silhouette Score:** {'✅ Buena estructura' if silhouette_avg > 0.5 else '⚠️ Estructura débil' if silhouette_avg > 0.2 else '❌ Sin estructura clara'}
        - **Davies-Bouldin:** {'✅ Buen clustering' if db_index < 1.0 else '⚠️ Clustering aceptable' if db_index < 1.5 else '❌ Clustering pobre'}
        """)
        
        # Visualizaciones
        st.subheader("📈 Visualización de Clusters")
        
        tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Dendrograma", "Análisis de Clusters"])
        
        with tab1:
            # Reducción dimensional para visualización
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Crear DataFrame para plotting
            plot_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': cluster_labels,
                'Tamaño': [30] * len(cluster_labels)
            })
            
            fig_clusters = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                                    size='Tamaño', opacity=0.7,
                                    title='Visualización de Clusters (PCA)',
                                    color_continuous_scale='viridis')
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            st.write(f"**Varianza explicada por PCA:** {sum(pca.explained_variance_ratio_):.2%}")
        
        with tab2:
            # Dendrograma
            st.subheader("Dendrograma")
            
            # Calcular linkage matrix para muestras más pequeñas
            sample_size = min(50, len(X_scaled))
            if len(X_scaled) > 100:
                st.info("Mostrando dendrograma para las primeras 50 muestras por rendimiento")
                X_sample = X_scaled[:50]
            else:
                X_sample = X_scaled
            
            linkage_matrix = sch.linkage(X_sample, method=linkage_method)
            
            # Crear dendrograma
            fig, ax = plt.subplots(figsize=(12, 6))
            dendrogram = sch.dendrogram(linkage_matrix, 
                                      truncate_mode='lastp',
                                      p=min(20, len(X_sample)),
                                      show_leaf_counts=True,
                                      ax=ax)
            plt.title(f'Dendrograma - {linkage_method.title()} Linkage')
            plt.xlabel('Índice de Muestra')
            plt.ylabel('Distancia')
            plt.axhline(y=linkage_matrix[-n_clusters+1, 2] if len(linkage_matrix) >= n_clusters else np.mean(linkage_matrix[:, 2]), 
                       color='r', linestyle='--', label='Límite de clusters')
            plt.legend()
            st.pyplot(fig)
        
        with tab3:
            # Análisis de clusters
            st.subheader("Análisis Detallado por Cluster")
            
            # Añadir labels al dataframe original
            df_clustered = df_processed.copy()
            df_clustered['Cluster'] = cluster_labels
            
            # Estadísticas por cluster
            cluster_stats = df_clustered.groupby('Cluster').mean().round(3)
            
            st.write("**Promedio de Características por Cluster:**")
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Distribución de clusters
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            
            fig_dist = px.bar(x=cluster_counts.index.astype(str), y=cluster_counts.values,
                             title='Distribución de Canciones por Cluster',
                             labels={'x': 'Cluster', 'y': 'Número de Canciones'},
                             color=cluster_counts.index.astype(str))
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Características más importantes por cluster
            st.write("**Características Promedio por Cluster (Normalizado):**")
            
            # Normalizar para heatmap
            normalized_means = (cluster_stats - cluster_stats.mean()) / cluster_stats.std()
            
            fig_heatmap = px.imshow(normalized_means.T,
                                   title='Características Normalizadas por Cluster',
                                   color_continuous_scale='RdBu_r',
                                   aspect='auto')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Guardar para exportación
        st.session_state['hierarchical_model'] = hierarchical
        st.session_state['hierarchical_metrics'] = {
            'silhouette_score': silhouette_avg,
            'davies_bouldin': db_index
        }
        st.session_state['cluster_labels'] = cluster_labels.tolist()
        st.session_state['hierarchical_params'] = {
            'n_clusters': n_clusters,
            'linkage_method': linkage_method,
            'features_used': selected_features
        }
        
        st.success("✅ Clustering Jerárquico completado exitosamente!")
        
    except Exception as e:
        st.error(f"Error en clustering: {e}")

# Zona de Exportación
elif app_mode == "📤 Exportación":
    st.markdown('<h2 class="section-header">📤 Zona de Exportación</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Exporta los resultados de tus modelos para usar en:**
    - 📱 Aplicaciones React
    - 🚀 APIs
    - 📊 Dashboards externos
    - 🔬 Análisis posteriores
    """)
    
    col1, col2 = st.columns(2)
    
    # Exportación KNN
    with col1:
        st.subheader("🎯 Modelo KNN")
        
        if 'knn_model' in st.session_state:
            # Crear JSON para KNN
            knn_json = {
                "model_type": "Supervised",
                "model_name": "K-Nearest Neighbors",
                "parameters": {
                    "n_neighbors": st.session_state['knn_model'].n_neighbors,
                    "algorithm": "auto",
                    "features_used": st.session_state.get('knn_features', [])
                },
                "metrics": st.session_state['knn_metrics'],
                "class_names": st.session_state.get('class_names', ['Class_0', 'Class_1']),
                "timestamp": pd.Timestamp.now().isoformat(),
                "dataset_info": {
                    "total_samples": len(df_processed),
                    "features_count": len(st.session_state.get('knn_features', []))
                }
            }
            
            json_knn = json.dumps(knn_json, indent=2)
            
            st.download_button(
                label="📥 Descargar JSON KNN",
                data=json_knn,
                file_name="knn_spotify_results.json",
                mime="application/json",
                type="primary"
            )
            
            # Exportar modelo
            knn_pkl = pickle.dumps(st.session_state['knn_model'])
            st.download_button(
                label="📥 Descargar Modelo KNN (.pkl)",
                data=knn_pkl,
                file_name="knn_spotify_model.pkl",
                mime="application/octet-stream"
            )
            
            # Vista previa
            with st.expander("👁️ Vista Previa JSON KNN"):
                st.code(json_knn, language='json')
        else:
            st.warning("⚠️ Entrena primero el modelo KNN en la pestaña correspondiente.")
    
    # Exportación Clustering
    with col2:
        st.subheader("🔍 Clustering Jerárquico")
        
        if 'hierarchical_model' in st.session_state:
            # Crear JSON para clustering
            clustering_json = {
                "model_type": "Unsupervised",
                "algorithm": "Hierarchical Clustering",
                "parameters": st.session_state['hierarchical_params'],
                "metrics": st.session_state['hierarchical_metrics'],
                "cluster_distribution": pd.Series(st.session_state['cluster_labels']).value_counts().to_dict(),
                "total_samples": len(st.session_state['cluster_labels']),
                "timestamp": pd.Timestamp.now().isoformat(),
                "dataset_info": {
                    "total_samples": len(df_processed),
                    "features_used": st.session_state['hierarchical_params']['features_used']
                }
            }
            
            json_clustering = json.dumps(clustering_json, indent=2)
            
            st.download_button(
                label="📥 Descargar JSON Clustering",
                data=json_clustering,
                file_name="hierarchical_clustering_spotify_results.json",
                mime="application/json",
                type="primary"
            )
            
            # Exportar modelo
            hierarchical_pkl = pickle.dumps(st.session_state['hierarchical_model'])
            st.download_button(
                label="📥 Descargar Modelo Clustering (.pkl)",
                data=hierarchical_pkl,
                file_name="hierarchical_clustering_spotify_model.pkl",
                mime="application/octet-stream"
            )
            
            # Vista previa
            with st.expander("👁️ Vista Previa JSON Clustering"):
                st.code(json_clustering, language='json')
        else:
            st.warning("⚠️ Ejecuta primero el clustering en la pestaña correspondiente.")
    
    # Ejemplo de consumo en React
    st.markdown("---")
    st.subheader("💡 Ejemplo de Consumo en React")
    
    st.markdown("""
    ```jsx
    // Ejemplo de componente React para mostrar resultados KNN
    import React, { useState, useEffect } from 'react';
    
    function SpotifyKNNResults() {
        const [modelData, setModelData] = useState(null);
        
        useEffect(() => {
            // En una aplicación real, esto vendría de una API
            fetch('/api/spotify-knn-results')
                .then(response => response.json())
                .then(data => setModelData(data));
        }, []);
        
        if (!modelData) return <div>Cargando...</div>;
        
        return (
            <div className="model-results">
                <h2>🎵 Spotify KNN Analysis</h2>
                <div className="metrics-grid">
                    <div className="metric-card">
                        <h3>Accuracy</h3>
                        <p>{(modelData.metrics.accuracy * 100).toFixed(2)}%</p>
                    </div>
                    <div className="metric-card">
                        <h3>Precision</h3>
                        <p>{(modelData.metrics.precision * 100).toFixed(2)}%</p>
                    </div>
                    <div className="metric-card">
                        <h3>Recall</h3>
                        <p>{(modelData.metrics.recall * 100).toFixed(2)}%</p>
                    </div>
                    <div className="metric-card">
                        <h3>F1-Score</h3>
                        <p>{(modelData.metrics.f1_score * 100).toFixed(2)}%</p>
                    </div>
                </div>
                
                <div className="model-info">
                    <h4>Configuración del Modelo</h4>
                    <p><strong>Vecinos:</strong> {modelData.parameters.n_neighbors}</p>
                    <p><strong>Características:</strong> {modelData.parameters.features_used.join(', ')}</p>
                    <p><strong>Muestras:</strong> {modelData.dataset_info.total_samples}</p>
                </div>
            </div>
        );
    }
    
    export default SpotifyKNNResults;
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🎵 Spotify ML Analysis | KNN & Clustering Jerárquico | "
    "Hecho con Streamlit 🤖"
    "</div>",
    unsafe_allow_html=True
)
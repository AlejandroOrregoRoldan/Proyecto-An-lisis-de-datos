import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de Internet en Colombia",
    page_icon="🇨🇴",
    layout="wide"
)

# --- Carga de Datos y Entrenamiento del Modelo (cacheado para eficiencia) ---
@st.cache_resource
def load_and_train():
    """Carga los datos, los limpia y entrena un modelo de Machine Learning."""
    # Cargar y limpiar datos
    try:
        df = pd.read_csv('Internet_Fijo_Penetraci_n_Municipio_20250804.csv')
    except FileNotFoundError:
        st.error("Error: El archivo 'Internet_Fijo_Penetraci_n_Municipio_20250804.csv' no se encontró.")
        st.stop()
        
    df['INDICE'] = df['INDICE'].str.replace(',', '.').astype(float)
    df['FECHA'] = pd.to_datetime(df['AÑO'].astype(str) + 'Q' + df['TRIMESTRE'].astype(str))

    # Preparar datos para el modelo
    df_model = df.copy()
    le_depto = LabelEncoder()
    le_municipio = LabelEncoder()
    df_model['DEPARTAMENTO_COD'] = le_depto.fit_transform(df_model['DEPARTAMENTO'])
    df_model['MUNICIPIO_COD'] = le_municipio.fit_transform(df_model['MUNICIPIO'])

    features = ['AÑO', 'TRIMESTRE', 'POBLACIÓN DANE', 'DEPARTAMENTO_COD', 'MUNICIPIO_COD']
    target = 'INDICE'
    X = df_model[features]
    y = df_model[target]

    # Entrenar el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    return model, df, le_depto, le_municipio

# Cargar todo
model, df, le_depto, le_municipio = load_and_train()

# --- Interfaz de la Aplicación ---
st.title('📊 Análisis de Penetración de Internet Fijo en Colombia')
st.markdown("Una herramienta interactiva para explorar datos y predecir el índice de penetración de internet.")

# --- Pestañas de Navegación ---
tab1, tab2 = st.tabs(["🚀 Análisis Exploratorio", "🧠 Modelo Predictivo"])

with tab1:
    st.header("Exploración Visual de los Datos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('evolucion_nacional.png', caption='Evolución del Índice de Penetración Nacional.')
    with col2:
        st.image('top_10_departamentos.png', caption='Top 10 Departamentos con Mayor Penetración.')
    
    st.image('distribucion_top5_deptos.png', caption='Distribución del Índice en los 5 Departamentos con mayor promedio.')
    
    st.markdown("---")
    st.subheader('Análisis Interactivo por Departamento')
    depto_seleccionado = st.selectbox('Selecciona un Departamento:', df['DEPARTAMENTO'].unique())
    
    df_depto = df[df['DEPARTAMENTO'] == depto_seleccionado]
    
    st.write(f"#### Evolución para {depto_seleccionado}")
    st.line_chart(df_depto.set_index('FECHA')['INDICE'])

with tab2:
    st.header("Predicción del Índice de Penetración")
    st.markdown("Utiliza nuestro modelo de Machine Learning para obtener una predicción del índice de penetración para una ubicación y fecha específicas.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            año = st.selectbox('Año', sorted(df['AÑO'].unique(), reverse=True))
            departamento = st.selectbox('Departamento', df['DEPARTAMENTO'].unique())
        with col2:
            trimestre = st.selectbox('Trimestre', df['TRIMESTRE'].unique())
            municipios_depto = df[df['DEPARTAMENTO'] == departamento]['MUNICIPIO'].unique()
            municipio = st.selectbox('Municipio', municipios_depto)
        with col3:
             # Obtener la población más reciente para el municipio seleccionado
            poblacion_reciente = df[df['MUNICIPIO'] == municipio]['POBLACIÓN DANE'].iloc[-1]
            st.metric(label="Población Aprox. (DANE)", value=f"{poblacion_reciente:,}")

        submitted = st.form_submit_button("🔮 Predecir")

        if submitted:
            # Codificar las entradas del usuario
            depto_cod = le_depto.transform([departamento])[0]
            municipio_cod = le_municipio.transform([municipio])[0]

            # Crear DataFrame para la predicción
            input_data = pd.DataFrame([[año, trimestre, poblacion_reciente, depto_cod, municipio_cod]], columns=['AÑO', 'TRIMESTRE', 'POBLACIÓN DANE', 'DEPARTAMENTO_COD', 'MUNICIPIO_COD'])
            
            # Realizar predicción
            prediction = model.predict(input_data)
            
            st.success(f"El índice de penetración predicho para **{municipio}** en el **T{trimestre} de {año}** es: **{prediction[0]:.2f}**")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 0. Configuración de la Página y Estilo ---
st.set_page_config(
    page_title="Análisis de Internet en Colombia",
    page_icon="📊",
    layout="wide"
)
sns.set_theme(style="whitegrid")

# --- 1. Carga de Datos (Cacheada) ---
@st.cache_data
def cargar_datos():
    """Carga y prepara el DataFrame una sola vez."""
    df = pd.read_csv('Internet_Fijo_Penetraci_n_Municipio_20250804.csv')
    df['No. ACCESOS FIJOS A INTERNET'] = df['No. ACCESOS FIJOS A INTERNET'].astype(int)
    return df

# --- 2. Definiciones de las Funciones para Gráficas ---

def generar_grafico_evolucion(df, departamento_seleccionado):
    """Genera un gráfico de líneas para la evolución de accesos en un departamento."""
    df_dep = df[df['DEPARTAMENTO'].str.upper() == departamento_seleccionado.upper()].copy()
    df_grouped = df_dep.groupby(['AÑO', 'TRIMESTRE'])['No. ACCESOS FIJOS A INTERNET'].sum().reset_index()
    df_grouped.sort_values(by=['AÑO', 'TRIMESTRE'], inplace=True)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    x = range(len(df_grouped))
    y = df_grouped['No. ACCESOS FIJOS A INTERNET']
    trimestres_labels = [f"T{t}" for t in df_grouped['TRIMESTRE']]
    ax.plot(x, y, marker='o', linestyle='-', color='#007ACC', linewidth=2, markersize=8, label='No. de Accesos')
    
    ax.set_title(f'Evolución de Accesos Fijos a Internet en {departamento_seleccionado.title()}', fontsize=18, weight='bold')
    ax.set_xlabel('Año y Trimestre', fontsize=12)
    ax.set_ylabel('Número de Accesos Fijos', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(trimestres_labels, rotation=45, ha="right")
    
    y_pos_anio = ax.get_ylim()[0]
    for i, row in df_grouped.iterrows():
        if row['TRIMESTRE'] == 1:
            ax.text(i, y_pos_anio, row['AÑO'], ha='center', va='bottom', fontsize=11, weight='bold')
            
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
    fig.tight_layout()
    return fig, df_grouped

def crear_grafico_ranking(df, año, trimestre, nivel_geografico, tipo_ranking, n=10):
    """Función flexible para generar gráficos de ranking."""
    df_periodo = df[(df['AÑO'] == año) & (df['TRIMESTRE'] == trimestre)].copy()
    
    if nivel_geografico == 'DEPARTAMENTO':
        data_proc = df_periodo.groupby('DEPARTAMENTO').agg({'No. ACCESOS FIJOS A INTERNET': 'sum', 'POBLACIÓN DANE': 'sum'}).reset_index()
    else:
        data_proc = df_periodo.copy()
        
    data_proc = data_proc[data_proc['POBLACIÓN DANE'] > 0]
    data_proc['INDICE_CALCULADO'] = (data_proc['No. ACCESOS FIJOS A INTERNET'] / data_proc['POBLACIÓN DANE']) * 100
    
    ascendente = (tipo_ranking == 'peores')
    if ascendente:
        data_proc = data_proc[data_proc['No. ACCESOS FIJOS A INTERNET'] > 0]
    
    data_ranked = data_proc.sort_values('INDICE_CALCULADO', ascending=ascendente).head(n)
    
    if data_ranked.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No hay datos para esta selección", ha='center')
        return fig, pd.DataFrame()
        
    titulo_ranking = "Mejores" if tipo_ranking == 'mejores' else "Peores"
    titulo_geo = "Departamentos" if nivel_geografico == 'DEPARTAMENTO' else "Municipios"
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=data_ranked['INDICE_CALCULADO'], y=data_ranked[nivel_geografico], palette='viridis' if tipo_ranking == 'mejores' else 'rocket_r', ax=ax)
    ax.set_title(f'{titulo_ranking} {n} {titulo_geo} por Penetración ({año}-T{trimestre})', fontsize=16, weight='bold')
    ax.set_xlabel('Accesos por cada 100 Habitantes (%)', fontsize=12)
    ax.set_ylabel(titulo_geo[:-1] if nivel_geografico.endswith('s') else titulo_geo, fontsize=12)
    fig.tight_layout()
    return fig, data_ranked

def crear_torta_accesos_por_departamento(df, top_n=10):
    """Genera un gráfico de torta con la distribución de accesos por departamento."""
    accesos_por_depto = df.groupby('DEPARTAMENTO')['No. ACCESOS FIJOS A INTERNET'].sum().sort_values(ascending=False)
    
    if len(accesos_por_depto) > top_n:
        top_deptos = accesos_por_depto.head(top_n)
        otros_sum = accesos_por_depto.iloc[top_n:].sum()
        data_to_plot = pd.concat([top_deptos, pd.Series({'Otros': otros_sum})])
    else:
        data_to_plot = accesos_por_depto
        
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = sns.color_palette('viridis_r', len(data_to_plot))
    wedges, texts, autotexts = ax.pie(data_to_plot, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85, wedgeprops=dict(width=0.4))
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.set_title('Distribución Porcentual de Accesos a Internet por Departamento (Histórico)', fontsize=16, weight='bold')
    ax.legend(wedges, data_to_plot.index, title="Departamentos", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    return fig, data_to_plot.reset_index(name='Total Accesos')

# --- 3. Construcción de la Aplicación en Streamlit ---
df_main = cargar_datos()

st.title("Análisis de Acceso a Internet Fijo en Colombia 🇨🇴")

# Crear las pestañas
tab1, tab2 = st.tabs(["📊 Análisis Exploratorio", "🧠 Modelo Predictivo"])

# --- Pestaña 1: Análisis Exploratorio ---
with tab1:
    st.header("Análisis de la Distribución y Ranking de Conectividad")
    st.markdown("""
    Esta sección te permite explorar la conectividad a internet en Colombia desde diferentes perspectivas. 
    Puedes analizar la evolución histórica de un departamento, comparar el rendimiento entre diferentes 
    municipios o departamentos en un periodo específico, y ver la distribución general del mercado.
    """)

    st.divider()

    # --- Sección de Ranking (Mejores/Peores) ---
    st.subheader("Ranking de Conectividad por Periodo")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        año_sel = st.selectbox("Año", sorted(df_main['AÑO'].unique(), reverse=True), key='ranking_año')
    with col2:
        trim_sel = st.selectbox("Trimestre", sorted(df_main['TRIMESTRE'].unique()), key='ranking_trim')
    with col3:
        geo_sel = st.selectbox("Nivel Geográfico", ["DEPARTAMENTO", "MUNICIPIO"], key='ranking_geo')
    with col4:
        ranking_sel = st.selectbox("Tipo de Ranking", ["Mejores 10", "Peores 10"], key='ranking_tipo')
        
    tipo_ranking_map = 'mejores' if ranking_sel == "Mejores 10" else 'peores'
    
    fig_ranking, datos_ranking = crear_grafico_ranking(df_main, año_sel, trim_sel, geo_sel, tipo_ranking_map)
    st.pyplot(fig_ranking)
    with st.expander(f"Ver datos de la tabla: {ranking_sel} de {geo_sel}s"):
        st.dataframe(datos_ranking)

    st.divider()

    # --- Sección de Evolución por Departamento ---
    st.subheader("Evolución Histórica por Departamento")
    depto_sel_evolucion = st.selectbox("Selecciona un Departamento para ver su evolución", sorted(df_main['DEPARTAMENTO'].unique()))
    
    fig_evolucion, datos_evolucion = generar_grafico_evolucion(df_main, depto_sel_evolucion)
    st.pyplot(fig_evolucion)
    with st.expander("Ver datos de la tabla de evolución"):
        st.dataframe(datos_evolucion)

    st.divider()

    # --- Sección de Distribución Total (Torta) ---
    st.subheader("Distribución General de Accesos por Departamento")
    fig_torta, datos_torta = crear_torta_accesos_por_departamento(df_main)
    st.pyplot(fig_torta)
    with st.expander("Ver datos de la distribución total"):
        st.dataframe(datos_torta)


# --- Pestaña 2: Modelo Predictivo ---
with tab2:
    st.header("Predicción de Índice de Penetración")
    st.markdown("---")
    st.warning("🚧 Esta sección se encuentra en construcción. ¡Próximamente podrás interactuar con el modelo predictivo!", icon="⚠️")
    
    # Aquí irá el código para tu modelo predictivo en el futuro
    # Por ejemplo:
    # st.subheader("Ingresa los datos para la predicción")
    # input_año = st.number_input("Año", min_value=2024)
    # ... etc.
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA E CORES DA MARCA ---
st.set_page_config(layout="wide")

# MUDANÇA: Paleta de cores definida para ser usada em todo o dashboard
CORES_SICOOB = {
    "turquesa": "#00AE9D",
    "verde_escuro": "#003641",
    "verde_claro": "#C9D200",
    "verde_medio": "#7DB61C",
    "roxo": "#49479D"
}

# --- TÍTULO E DESCRIÇÃO ---
st.title("🚜 Análise de Propensos")
st.markdown("Utilize os filtros na barra lateral para explorar os dados por localidade e categoria.")

# --- FUNÇÕES AUXILIARES ---
@st.cache_data
def carregar_dados():
    # Seu código para carregar e preparar os dados permanece o mesmo
    df_rural = pd.read_parquet('LISTA_RURAL_202505.parquet', engine='pyarrow')
    df_municipios = pd.read_csv('brasil.csv', encoding='utf-8')
    
    # ... (todo o seu código de merge e limpeza de dados) ...
    df_rural['Código_IBGE'] = pd.to_numeric(df_rural['Código_IBGE'], errors='coerce')
    df_municipios['ibge'] = pd.to_numeric(df_municipios['ibge'], errors='coerce')
    df_rural.dropna(subset=['Número_CPF_CNPJ'], inplace=True)
    df_rural['Código_IBGE'] = df_rural['Código_IBGE'].astype(str)
    df_rural['ibge_6dig'] = df_rural['Código_IBGE'].str.slice(0, 6)
    df_rural['ibge_6dig'] = pd.to_numeric(df_rural['ibge_6dig'], errors='coerce')
    df_rural.dropna(subset=['ibge_6dig'], inplace=True)
    df_municipios['ibge'] = df_municipios['ibge'].astype(str)
    df_municipios['ibge_6dig'] = df_municipios['ibge'].str.slice(0, 6)
    df_municipios['ibge_6dig'] = pd.to_numeric(df_municipios['ibge_6dig'], errors='coerce')
    df_municipios.dropna(subset=['ibge_6dig'], inplace=True)
    df_completo = pd.merge(df_rural, df_municipios[['ibge_6dig', 'latitude', 'longitude']], on='ibge_6dig', how='left')
    colunas_para_converter = ['VLR_VENCER_FORA_TOTAL', 'QT_PRODUTOS', 'SLD_DVD_RURAL_SCB', 'SLD_DVD_RURAL_SFN_CRG']
    for coluna in colunas_para_converter:
        if coluna in df_completo.columns:
            df_completo[coluna] = pd.to_numeric(df_completo[coluna].astype(str).str.replace(',', '.'), errors='coerce')
    df_completo['PRONAF'] = np.where(df_completo['Numero_CAF'].notna(), 1, 0)
    df_completo['PRONAMP'] = np.where(df_completo['Porte'] == 'médio produtor', 1, 0)
    df_completo['DEMAIS'] = np.where(df_completo['Porte'] == 'grande produtor', 1, 0)
    
    return df_completo

def criar_mapa_folium(_data):
    if _data.empty:
        return folium.Map(location=[-15.788494, -47.882569], zoom_start=4, tiles="OpenStreetMap")

    map_center = [_data['latitude'].mean(), _data['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=5, tiles="OpenStreetMap")
    marker_cluster = MarkerCluster().add_to(m)
    max_contagem = _data['contagem'].max()
    min_radius, max_radius = 5, 30

    for idx, row in _data.iterrows():
        radius = min_radius
        if max_contagem > 0:
            radius = min_radius + (row['contagem'] / max_contagem) * (max_radius - min_radius)
        
        popup_text = f"<b>Município:</b> {row['Município']}<br><b>Produtores:</b> {row['contagem']}"
        
        # MUDANÇA: Cor do mapa alterada para o turquesa da marca
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']], radius=radius, popup=popup_text,
            color=CORES_SICOOB["turquesa"], fill=True, fill_color=CORES_SICOOB["turquesa"], fill_opacity=0.6
        ).add_to(marker_cluster)
    return m

# Carrega os dados
df = carregar_dados()

# --- BARRA LATERAL COM FILTROS HIERÁRQUICOS AVANÇADOS ---

st.sidebar.header("Filtros de Seleção")

# MUDANÇA: Lógica de filtros reescrita com st.session_state para permitir filtragem reversa
if 'uf' not in st.session_state:
    st.session_state.uf = 'Todos'
if 'municipio' not in st.session_state:
    st.session_state.municipio = 'Todos'
if 'cooperativa' not in st.session_state:
    st.session_state.cooperativa = 'Todos'
if 'pa' not in st.session_state:
    st.session_state.pa = 'Todos'

# Filtros são aplicados sequencialmente
df_filtrado = df.copy()

# 1. Filtro de UF
lista_ufs = ['Todos'] + sorted(df['UF'].unique().tolist())
uf_selecionada = st.sidebar.selectbox("UF", lista_ufs, index=lista_ufs.index(st.session_state.uf))
if uf_selecionada != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['UF'] == uf_selecionada]

# 2. Filtro de Município
lista_municipios = ['Todos'] + sorted(df_filtrado['Município'].unique().tolist())
municipio_selecionado = st.sidebar.selectbox("Município", lista_municipios, index=lista_municipios.index(st.session_state.municipio if st.session_state.municipio in lista_municipios else 'Todos'))
if municipio_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Município'] == municipio_selecionado]

# 3. Filtro de Cooperativa
lista_cooperativas = ['Todos'] + sorted(df_filtrado['Número_Cooperativa'].unique().tolist())
cooperativa_selecionada = st.sidebar.selectbox("Cooperativa", lista_cooperativas, index=lista_cooperativas.index(st.session_state.cooperativa if st.session_state.cooperativa in lista_cooperativas else 'Todos'))
if cooperativa_selecionada != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Número_Cooperativa'] == cooperativa_selecionada]

# 4. Filtro de PA
lista_pas = ['Todos'] + sorted(df_filtrado['Número_PA'].unique().tolist())
pa_selecionado = st.sidebar.selectbox("Posto de Atendimento (PA)", lista_pas, index=lista_pas.index(st.session_state.pa if st.session_state.pa in lista_pas else 'Todos'))

# Lógica de filtragem reversa
if pa_selecionado != 'Todos' and pa_selecionado != st.session_state.pa:
    pa_info = df[df['Número_PA'] == pa_selecionado].iloc[0]
    st.session_state.uf = pa_info['UF']
    st.session_state.municipio = pa_info['Município']
    st.session_state.cooperativa = pa_info['Número_Cooperativa']
    st.session_state.pa = pa_selecionado
    st.rerun()

# Atualiza o estado da sessão com as seleções atuais
st.session_state.uf, st.session_state.municipio, st.session_state.cooperativa, st.session_state.pa = uf_selecionada, municipio_selecionado, cooperativa_selecionada, pa_selecionado

# Aplica o filtro final de PA
if pa_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Número_PA'] == pa_selecionado]

df_final_filtrado = df_filtrado

# --- PAINEL PRINCIPAL ---
st.header("Visão Geral por Categoria")

# MUDANÇA 1: Adicionada a opção 'Todos' na lista do st.radio
opcoes_categoria = ['Todos', 'PRONAF', 'PRONAMP', 'DEMAIS']
tipo_selecionado = st.radio(
    "Selecione a categoria para visualizar:",
    options=opcoes_categoria, # Usando a nova lista de opções
    horizontal=True,
    key='filtro_principal'
)

# MUDANÇA 2: Adicionada a lógica para a opção 'Todos'
if tipo_selecionado == 'Todos':
    # Se 'Todos' for selecionado, usamos o dataframe filtrado pela sidebar, sem filtro de categoria
    df_kpi_e_mapa = df_final_filtrado
elif tipo_selecionado == 'PRONAF':
    df_kpi_e_mapa = df_final_filtrado[df_final_filtrado['PRONAF'] == 1]
elif tipo_selecionado == 'PRONAMP':
    df_kpi_e_mapa = df_final_filtrado[df_final_filtrado['PRONAMP'] == 1]
else: # DEMAIS
    df_kpi_e_mapa = df_final_filtrado[df_final_filtrado['DEMAIS'] == 1]


st.markdown("---")

# A lógica dos KPIs e o resto do código não precisam de nenhuma alteração
col1, col2, col3, col4 = st.columns(4)

# Os títulos dos KPIs agora refletem a seleção, incluindo 'Todos'
total_produtores = len(df_kpi_e_mapa)
col1.metric(f"Total de Produtores ({tipo_selecionado})", f"{total_produtores:,}".replace(",", "."))

media_iap = df_kpi_e_mapa['QT_PRODUTOS'].mean() if not df_kpi_e_mapa.empty else 0
col2.metric(f"Média IAP ({tipo_selecionado})", f"{media_iap:.2f}")

total_iap = int(df_kpi_e_mapa['QT_PRODUTOS'].sum()) if not df_kpi_e_mapa.empty else 0
col3.metric(f"Total IAP ({tipo_selecionado})", f"{total_iap:,}".replace(",", "."))

total_valor_vencer = df_kpi_e_mapa['VLR_VENCER_FORA_TOTAL'].sum()
col4.metric(f"Total a Vencer ({tipo_selecionado})", f"R$ {total_valor_vencer:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

st.markdown("---")
st.header("🗺 Mapa de Distribuição")
if tipo_selecionado == 'PRONAF':
    map_data = df_kpi_e_mapa.groupby(['Município', 'latitude', 'longitude']).agg(contagem=('Numero_CAF', 'count')).reset_index()
else:
    map_data = df_kpi_e_mapa.groupby(['Município', 'latitude', 'longitude']).agg(contagem=('Número_CPF_CNPJ', 'count')).reset_index()

if map_data.empty:
    st.warning("Nenhum dado encontrado para a seleção atual.")
else:
    mapa_gerado = criar_mapa_folium(map_data)
    st_folium(mapa_gerado, returned_objects=[], width=1200, height=500, key="mapa_otimizado")

st.markdown("---")
st.header(f"Análises Adicionais para {tipo_selecionado}")
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    # MUDANÇA: Gráfico refeito com Plotly para customização
    st.subheader("Top 10 Municípios por Produtores")
    top_municipios = df_kpi_e_mapa['Município'].value_counts().nlargest(10).reset_index()
    top_municipios.columns = ['Município', 'Contagem']
    
    fig_municipios = px.bar(top_municipios, 
                            x='Município', 
                            y='Contagem',
                            title="Top 5 Municípios",
                            labels={'Contagem': f'Qtd {tipo_selecionado}', 'Município': 'Município'},
                            color_discrete_sequence=[CORES_SICOOB["turquesa"]])
    fig_municipios.update_layout(xaxis_tickangle=-45, title_x=0.5)
    st.plotly_chart(fig_municipios, use_container_width=True)

# MUDANÇA: Novo gráfico de Saldo Devedor por Cooperativa
with col_graf2:
    st.subheader("Top 5 Cooperativas por Saldo Devedor")

    dados_coops = df_kpi_e_mapa[df_kpi_e_mapa['SLD_DVD_RURAL_SFN_CRG'] > 0]
    top_coops = dados_coops.groupby('Número_Cooperativa')['SLD_DVD_RURAL_SFN_CRG'].sum().nlargest(5).reset_index()

    if top_coops.empty:
        st.info("Não há dados de saldo devedor para a seleção atual.")
    else:
        top_coops['Número_Cooperativa'] = top_coops['Número_Cooperativa'].astype(str)
        top_coops['Saldo Formatado'] = top_coops['SLD_DVD_RURAL_SFN_CRG'].apply(
            lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )

        fig_coops = px.bar(top_coops,
                           x='Número_Cooperativa',
                           y='SLD_DVD_RURAL_SFN_CRG',
                           title="Top 5 Cooperativas por Saldo Devedor",
                           labels={'SLD_DVD_RURAL_SFN_CRG': 'Saldo Devedor (R$)', 'Número_Cooperativa': 'Cooperativa'},
                           color_discrete_sequence=[CORES_SICOOB["verde_medio"]],
                           text=top_coops['Saldo Formatado']
                           )

        fig_coops.update_traces(
            hovertemplate="<b>Cooperativa:</b> %{x}<br><b>Saldo Devedor:</b> %{text}",
            texttemplate='%{text}',
            textposition='outside'
        )
        
        # MUDANÇA: Simplificada a configuração do eixo para garantir a ordenação correta pelo valor
        fig_coops.update_layout(
            xaxis_tickangle=-45,
            title_x=0.5,
            uniformtext_minsize=8, 
            uniformtext_mode='hide',
            xaxis_categoryorder='total descending' # <--- ESTA É A FORMA CORRETA E SIMPLES
        )

        st.plotly_chart(fig_coops, use_container_width=True)

with st.expander(f"Clique aqui para ver os dados detalhados (primeiras 500 linhas de {tipo_selecionado})"):
    st.dataframe(df_kpi_e_mapa.head(500))
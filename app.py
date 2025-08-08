import os
import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

# Inicializar la app
app = dash.Dash(__name__)
server = app.server
app.title = "Dashboard de Aprendizaje Automático"
BASE = os.path.dirname(__file__)
print("🔄 Cargando modelos...")

# -----------------------
# 🔹 Cargar modelos
# -----------------------
try:
    modelo_regresion = joblib.load(os.path.join(BASE, "RegresionSa.pkl"))
    print("✅ Modelo de regresión cargado")
except:
    print("❌ Error cargando modelo de regresión")
    modelo_regresion = None

try:
    modelo_clasificacion = joblib.load(os.path.join(BASE, "ClasificacionDe.pkl"))
    print("✅ Modelo de clasificación cargado")
except:
    print("❌ Error cargando modelo de clasificación")
    modelo_clasificacion = None

try:
    modelo_agrupamiento = joblib.load(os.path.join(BASE, "AgrupamientoSa.pkl"))
    print("✅ Modelo de agrupamiento cargado")
except:
    print("❌ Error cargando modelo de agrupamiento")
    modelo_agrupamiento = None

try:
    label_encoders = joblib.load(os.path.join(BASE, "label_encoders.pkl"))
    income_encoder = joblib.load(os.path.join(BASE, "income_encoder.pkl"))
    print("✅ Encoders cargados")
except:
    print("❌ Error cargando encoders")
    label_encoders = None
    income_encoder = None

# Crear scaler para clustering (simulado)
scaler_cluster = StandardScaler()
# Datos de ejemplo para ajustar el scaler
ejemplo_datos = np.array([[100, 5, 10], [500, 15, 25], [1000, 30, 50]])
scaler_cluster.fit(ejemplo_datos)

# -----------------------
# 🔹 LAYOUT DEL DASHBOARD
# -----------------------
app.layout = html.Div([
    # HEADER CON TÍTULO Y ENLACES
    html.Div([
        html.H1("🤖 Dashboard Interactivo de Machine Learning", 
               style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        
        # SECCIÓN DE ENLACES
        html.Div([
            html.H4("📚 Enlaces de Colab:", style={'textAlign': 'center', 'color': '#34495e', 'marginBottom': 15}),
            html.Div([
                html.A([
                    html.Div([
                        "📊 Notebook 1: Análisis Principal",
                        html.Br(),
                        html.Small("Modelo de regresión y clustering", style={'color': '#000'})
                    ], style={
                        'padding': '12px 20px',
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'borderRadius': '8px',
                        'textAlign': 'center',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'transition': 'all 0.3s ease'
                    })
                ], href="https://colab.research.google.com/drive/1pLL7kBPkB7PMZ571qkIYD7lodG7M3Gn2?authuser=0#scrollTo=-EgOj-ivB62J",
                   target="_blank", style={'textDecoration': 'none', 'margin': '0 10px'}),
                
                html.A([
                    html.Div([
                        "🔬 Notebook 2: Análisis Secundario",
                        html.Br(),
                        html.Small("Modelos de clasificación y reglas de asosiación", style={'color': '#000'})
                    ], style={
                        'padding': '12px 20px',
                        'backgroundColor': '#27ae60',
                        'color': 'white',
                        'borderRadius': '8px',
                        'textAlign': 'center',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'transition': 'all 0.3s ease'
                    })
                ], href="https://colab.research.google.com/drive/1uMtgGDv6_T0phNQ7hXtlSK_ZMd2GAEXu",
                   target="_blank", style={'textDecoration': 'none', 'margin': '0 10px'})
            ], style={
                'display': 'flex', 
                'justifyContent': 'center', 
                'alignItems': 'center',
                'flexWrap': 'wrap',
                'marginBottom': 30
            })
        ], style={
            'backgroundColor': '#ecf0f1',
            'padding': '20px',
            'borderRadius': '10px',
            'marginBottom': '30px'
        })
    ]),

    dcc.Tabs(id="tabs-ml", value='tab-1', children=[
        
        # TAB 1: REGRESIÓN - Criterio 1: Predicción Individual
        dcc.Tab(label='📈 Regresión - Individual', value='tab-1', children=[
            html.Div([
                html.H2("🎯 Criterio 1: Predicción de Felicidad Individual", style={'color': '#e74c3c'}),
                html.P("Ingresa los datos de un país para predecir su índice de felicidad"),
                
                html.Div([
                    html.Div([
                        html.Label("PIB per cápita:", style={'fontWeight': 'bold'}),
                        dcc.Slider(id='gdp-slider', min=0.1, max=2.0, step=0.1, value=1.0,
                                  marks={i/10: f'{i/10}' for i in range(1, 21, 3)},
                                  tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Label("Apoyo Social:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                        dcc.Slider(id='social-slider', min=0.0, max=1.0, step=0.1, value=0.7,
                                  marks={i/10: f'{i/10}' for i in range(0, 11, 2)},
                                  tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Label("Expectativa de Vida Saludable:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                        dcc.Slider(id='health-slider', min=0.0, max=1.0, step=0.1, value=0.6,
                                  marks={i/10: f'{i/10}' for i in range(0, 11, 2)},
                                  tooltip={"placement": "bottom", "always_visible": True}),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.Label("Libertad de Elección:", style={'fontWeight': 'bold'}),
                        dcc.Slider(id='freedom-slider', min=0.0, max=0.8, step=0.1, value=0.4,
                                  marks={i/10: f'{i/10}' for i in range(0, 9, 2)},
                                  tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Label("Generosidad:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                        dcc.Slider(id='generosity-slider', min=0.0, max=0.5, step=0.1, value=0.2,
                                  marks={i/10: f'{i/10}' for i in range(0, 6)},
                                  tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Label("Percepción de Corrupción:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                        dcc.Slider(id='corruption-slider', min=0.0, max=1.0, step=0.1, value=0.5,
                                  marks={i/10: f'{i/10}' for i in range(0, 11, 2)},
                                  tooltip={"placement": "bottom", "always_visible": True}),
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                ], style={'marginBottom': '30px'}),
                
                html.Div(id='regression-result', style={'fontSize': '20px', 'textAlign': 'center',
                        'backgroundColor': '#d4edda', 'padding': '20px', 'borderRadius': '10px'})
            ], style={'padding': '20px'})
        ]),

        # TAB 2: REGRESIÓN - Criterio 2: Comparación de Países
        dcc.Tab(label='📊 Regresión - Comparación', value='tab-2', children=[
            html.Div([
                html.H2("⚖️ Criterio 2: Comparación entre Países", style={'color': '#8e44ad'}),
                html.P("Compara hasta 3 países diferentes"),
                
                html.Div([
                    # País 1
                    html.Div([
                        html.H4("🇦 País A", style={'color': '#e74c3c'}),
                        html.Label("Nombre del País:"),
                        dcc.Input(id='pais1-nombre', type='text', value='País A', 
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("PIB per cápita:"),
                        dcc.Input(id='pais1-gdp', type='number', value=1.0, min=0.1, max=2.0, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Apoyo Social:"),
                        dcc.Input(id='pais1-social', type='number', value=0.7, min=0, max=1, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Expectativa de Vida:"),
                        dcc.Input(id='pais1-health', type='number', value=0.6, min=0, max=1, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Libertad:"),
                        dcc.Input(id='pais1-freedom', type='number', value=0.4, min=0, max=0.8, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Generosidad:"),
                        dcc.Input(id='pais1-generosity', type='number', value=0.2, min=0, max=0.5, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Percepción Corrupción:"),
                        dcc.Input(id='pais1-corruption', type='number', value=0.5, min=0, max=1, step=0.1,
                                style={'width': '100%'})
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                    
                    # País 2
                    html.Div([
                        html.H4("🇧 País B", style={'color': '#3498db'}),
                        html.Label("Nombre del País:"),
                        dcc.Input(id='pais2-nombre', type='text', value='País B',
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("PIB per cápita:"),
                        dcc.Input(id='pais2-gdp', type='number', value=1.5, min=0.1, max=2.0, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Apoyo Social:"),
                        dcc.Input(id='pais2-social', type='number', value=0.8, min=0, max=1, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Expectativa de Vida:"),
                        dcc.Input(id='pais2-health', type='number', value=0.8, min=0, max=1, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Libertad:"),
                        dcc.Input(id='pais2-freedom', type='number', value=0.6, min=0, max=0.8, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Generosidad:"),
                        dcc.Input(id='pais2-generosity', type='number', value=0.3, min=0, max=0.5, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Percepción Corrupción:"),
                        dcc.Input(id='pais2-corruption', type='number', value=0.3, min=0, max=1, step=0.1,
                                style={'width': '100%'})
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                    
                    # País 3
                    html.Div([
                        html.H4("🇨 País C", style={'color': '#27ae60'}),
                        html.Label("Nombre del País:"),
                        dcc.Input(id='pais3-nombre', type='text', value='País C',
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("PIB per cápita:"),
                        dcc.Input(id='pais3-gdp', type='number', value=0.8, min=0.1, max=2.0, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Apoyo Social:"),
                        dcc.Input(id='pais3-social', type='number', value=0.5, min=0, max=1, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Expectativa de Vida:"),
                        dcc.Input(id='pais3-health', type='number', value=0.5, min=0, max=1, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Libertad:"),
                        dcc.Input(id='pais3-freedom', type='number', value=0.3, min=0, max=0.8, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Generosidad:"),
                        dcc.Input(id='pais3-generosity', type='number', value=0.1, min=0, max=0.5, step=0.1,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Percepción Corrupción:"),
                        dcc.Input(id='pais3-corruption', type='number', value=0.7, min=0, max=1, step=0.1,
                                style={'width': '100%'})
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
                ]),
                
                html.Div(id='comparison-result', style={'marginTop': '20px'})
            ], style={'padding': '20px'})
        ]),

        # TAB 3: CLUSTERING - Criterio 2
        dcc.Tab(label='🎭 Clustering - Individual', value='tab-4', children=[
            html.Div([
                html.H2("🛒 Criterio 1: Análisis de Cliente Individual", style={'color': '#9b59b6'}),
                html.P("Ingresa los datos de un cliente para determinar su segmento"),
                
                html.Div([
                    html.Div([
                        html.Label("Gasto Total ($):", style={'fontWeight': 'bold'}),
                        dcc.Input(id='gasto-input', type='number', value=500, min=1, max=5000,
                                style={'width': '100%', 'marginBottom': '20px'}),
                        
                        html.Label("Número de Transacciones:", style={'fontWeight': 'bold'}),
                        dcc.Input(id='transacciones-input', type='number', value=10, min=1, max=100,
                                style={'width': '100%', 'marginBottom': '20px'}),
                        
                        html.Label("Productos Comprados:", style={'fontWeight': 'bold'}),
                        dcc.Input(id='productos-input', type='number', value=20, min=1, max=200,
                                style={'width': '100%', 'marginBottom': '20px'})
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div(id='clustering-individual-result', 
                            style={'width': '45%', 'float': 'right', 'backgroundColor': '#f8f9fa', 
                                   'padding': '20px', 'borderRadius': '10px'})
                ])
            ], style={'padding': '20px'})
        ]),

        # TAB 4: CLUSTERING - Criterio 2
        dcc.Tab(label='📊 Clustering - Múltiple', value='tab-5', children=[
            html.Div([
                html.H2("👥 Criterio 2: Análisis de Múltiples Clientes", style={'color': '#16a085'}),
                html.P("Ingresa datos de varios clientes para compararlos"),
                
                html.Div([
                    # Cliente 1
                    html.Div([
                        html.H4("🛍️ Cliente 1", style={'color': '#e74c3c'}),
                        html.Label("Nombre:"),
                        dcc.Input(id='cliente1-nombre', type='text', value='Cliente A',
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Gasto Total:"),
                        dcc.Input(id='cliente1-gasto', type='number', value=300,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Transacciones:"),
                        dcc.Input(id='cliente1-trans', type='number', value=8,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Productos:"),
                        dcc.Input(id='cliente1-prod', type='number', value=15,
                                style={'width': '100%'})
                    ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                    
                    # Cliente 2
                    html.Div([
                        html.H4("🛍️ Cliente 2", style={'color': '#3498db'}),
                        html.Label("Nombre:"),
                        dcc.Input(id='cliente2-nombre', type='text', value='Cliente B',
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Gasto Total:"),
                        dcc.Input(id='cliente2-gasto', type='number', value=800,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Transacciones:"),
                        dcc.Input(id='cliente2-trans', type='number', value=20,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Productos:"),
                        dcc.Input(id='cliente2-prod', type='number', value=35,
                                style={'width': '100%'})
                    ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                    
                    # Cliente 3
                    html.Div([
                        html.H4("🛍️ Cliente 3", style={'color': '#27ae60'}),
                        html.Label("Nombre:"),
                        dcc.Input(id='cliente3-nombre', type='text', value='Cliente C',
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Gasto Total:"),
                        dcc.Input(id='cliente3-gasto', type='number', value=150,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Transacciones:"),
                        dcc.Input(id='cliente3-trans', type='number', value=5,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Productos:"),
                        dcc.Input(id='cliente3-prod', type='number', value=8,
                                style={'width': '100%'})
                    ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                    
                    # Cliente 4
                    html.Div([
                        html.H4("🛍️ Cliente 4", style={'color': '#f39c12'}),
                        html.Label("Nombre:"),
                        dcc.Input(id='cliente4-nombre', type='text', value='Cliente D',
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Gasto Total:"),
                        dcc.Input(id='cliente4-gasto', type='number', value=600,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Transacciones:"),
                        dcc.Input(id='cliente4-trans', type='number', value=15,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        html.Label("Productos:"),
                        dcc.Input(id='cliente4-prod', type='number', value=28,
                                style={'width': '100%'})
                    ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
                ]),
                
                html.Div(id='clustering-multiple-result', style={'marginTop': '20px'})
            ], style={'padding': '20px'})
        ]),

        # TAB 5: CLASIFICACIÓN
        dcc.Tab(label='🎯 Clasificación', value='tab-3', children=[
            html.Div([
                html.H2("👤 Análisis de Perfil Individual", style={'color': '#3498db'}),
                html.P("Ingresa los datos de una persona para predecir sus ingresos"),
                
                html.Div([
                    html.Div([
                        html.Label("Edad:", style={'fontWeight': 'bold'}),
                        dcc.Input(id='edad-input', type='number', value=39, min=17, max=90,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("Clase de Trabajo:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='workclass-dropdown',
                                   options=[
                                       {'label': 'Private', 'value': 'Private'},
                                       {'label': 'Self-emp-not-inc', 'value': 'Self-emp-not-inc'},
                                       {'label': 'Self-emp-inc', 'value': 'Self-emp-inc'},
                                       {'label': 'Federal-gov', 'value': 'Federal-gov'},
                                       {'label': 'Local-gov', 'value': 'Local-gov'},
                                       {'label': 'State-gov', 'value': 'State-gov'},
                                       {'label': 'Without-pay', 'value': 'Without-pay'},
                                       {'label': 'Never-worked', 'value': 'Never-worked'}
                                   ], value='Private', style={'marginBottom': '10px'}),
                        
                        html.Label("Educación:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='education-dropdown',
                                   options=[
                                       {'label': 'Bachelors', 'value': 'Bachelors'},
                                       {'label': 'Some-college', 'value': 'Some-college'},
                                       {'label': 'HS-grad', 'value': 'HS-grad'},
                                       {'label': 'Masters', 'value': 'Masters'},
                                       {'label': 'Assoc-voc', 'value': 'Assoc-voc'},
                                       {'label': 'Doctorate', 'value': 'Doctorate'},
                                       {'label': 'Prof-school', 'value': 'Prof-school'},
                                       {'label': '11th', 'value': '11th'},
                                       {'label': '10th', 'value': '10th'},
                                       {'label': '7th-8th', 'value': '7th-8th'},
                                       {'label': '12th', 'value': '12th'},
                                       {'label': '1st-4th', 'value': '1st-4th'},
                                       {'label': '5th-6th', 'value': '5th-6th'},
                                       {'label': '9th', 'value': '9th'},
                                       {'label': 'Preschool', 'value': 'Preschool'}
                                   ], value='Bachelors', style={'marginBottom': '10px'}),
                        
                        html.Label("Estado Civil:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='marital-dropdown',
                                   options=[
                                       {'label': 'Never-married', 'value': 'Never-married'},
                                       {'label': 'Married-civ-spouse', 'value': 'Married-civ-spouse'},
                                       {'label': 'Divorced', 'value': 'Divorced'},
                                       {'label': 'Married-spouse-absent', 'value': 'Married-spouse-absent'},
                                       {'label': 'Separated', 'value': 'Separated'},
                                       {'label': 'Married-AF-spouse', 'value': 'Married-AF-spouse'},
                                       {'label': 'Widowed', 'value': 'Widowed'}
                                   ], value='Never-married', style={'marginBottom': '10px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.Label("Ocupación:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='occupation-dropdown',
                                   options=[
                                       {'label': 'Exec-managerial', 'value': 'Exec-managerial'},
                                       {'label': 'Prof-specialty', 'value': 'Prof-specialty'},
                                       {'label': 'Craft-repair', 'value': 'Craft-repair'},
                                       {'label': 'Adm-clerical', 'value': 'Adm-clerical'},
                                       {'label': 'Sales', 'value': 'Sales'},
                                       {'label': 'Other-service', 'value': 'Other-service'},
                                       {'label': 'Machine-op-inspct', 'value': 'Machine-op-inspct'},
                                       {'label': 'Transport-moving', 'value': 'Transport-moving'},
                                       {'label': 'Handlers-cleaners', 'value': 'Handlers-cleaners'},
                                       {'label': 'Farming-fishing', 'value': 'Farming-fishing'},
                                       {'label': 'Tech-support', 'value': 'Tech-support'},
                                       {'label': 'Protective-serv', 'value': 'Protective-serv'},
                                       {'label': 'Priv-house-serv', 'value': 'Priv-house-serv'},
                                       {'label': 'Armed-Forces', 'value': 'Armed-Forces'}
                                   ], value='Exec-managerial', style={'marginBottom': '10px'}),
                        
                        html.Label("Sexo:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='sex-dropdown',
                                   options=[
                                       {'label': 'Male', 'value': 'Male'},
                                       {'label': 'Female', 'value': 'Female'}
                                   ], value='Male', style={'marginBottom': '10px'}),
                        
                        html.Label("Horas por Semana:", style={'fontWeight': 'bold'}),
                        dcc.Input(id='hours-input', type='number', value=40, min=1, max=99,
                                style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label("País de Origen:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='country-dropdown',
                                   options=[
                                       {'label': 'United-States', 'value': 'United-States'},
                                       {'label': 'Mexico', 'value': 'Mexico'},
                                       {'label': 'Philippines', 'value': 'Philippines'},
                                       {'label': 'Germany', 'value': 'Germany'},
                                       {'label': 'Puerto-Rico', 'value': 'Puerto-Rico'},
                                       {'label': 'Canada', 'value': 'Canada'},
                                       {'label': 'India', 'value': 'India'},
                                       {'label': 'Japan', 'value': 'Japan'},
                                       {'label': 'China', 'value': 'China'},
                                       {'label': 'United-Kingdom', 'value': 'United-Kingdom'}
                                   ], value='United-States', style={'marginBottom': '10px'})
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                ]),
                
                html.Div(id='classification-result', style={'marginTop': '30px', 'fontSize': '18px',
                        'textAlign': 'center', 'backgroundColor': '#fff3cd', 'padding': '20px', 'borderRadius': '10px'})
            ], style={'padding': '20px'})
        ]),

        # TAB 6: REGLAS DE ASOCIACIÓN
        dcc.Tab(label='🔗 Reglas de Asociación', value='tab-6', children=[
            html.Div([
                html.H2("🌬️ Análisis de Calidad del Aire", style={'color': '#f39c12'}),
                html.P("Ingresa condiciones ambientales para encontrar reglas de asociación"),
                
                html.Div([
                    html.Div([
                        html.Label("Nivel de CO:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='co-dropdown',
                                   options=[
                                       {'label': 'Bajo', 'value': 'bajo'},
                                       {'label': 'Medio', 'value': 'medio'},
                                       {'label': 'Alto', 'value': 'alto'}
                                   ], value='medio', style={'marginBottom': '15px'}),
                        
                        html.Label("Nivel de NOx:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='nox-dropdown',
                                   options=[
                                       {'label': 'Bajo', 'value': 'bajo'},
                                       {'label': 'Medio', 'value': 'medio'},
                                       {'label': 'Alto', 'value': 'alto'}
                                   ], value='medio', style={'marginBottom': '15px'}),
                        
                        html.Label("Nivel de NO2:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='no2-dropdown',
                                   options=[
                                       {'label': 'Bajo', 'value': 'bajo'},
                                       {'label': 'Medio', 'value': 'medio'},
                                       {'label': 'Alto', 'value': 'alto'}
                                   ], value='medio', style={'marginBottom': '15px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.Label("Temperatura:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='temp-dropdown',
                                   options=[
                                       {'label': 'Baja', 'value': 'bajo'},
                                       {'label': 'Media', 'value': 'medio'},
                                       {'label': 'Alta', 'value': 'alto'}
                                   ], value='medio', style={'marginBottom': '15px'}),
                        
                        html.Label("Humedad Relativa:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='humedad-dropdown',
                                   options=[
                                       {'label': 'Baja', 'value': 'bajo'},
                                       {'label': 'Media', 'value': 'medio'},
                                       {'label': 'Alta', 'value': 'alto'}
                                   ], value='medio', style={'marginBottom': '15px'}),
                        
                        html.Label("Nivel de C6H6 (Benceno):", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='benceno-dropdown',
                                   options=[
                                       {'label': 'Bajo', 'value': 'bajo'},
                                       {'label': 'Medio', 'value': 'medio'},
                                       {'label': 'Alto', 'value': 'alto'}
                                   ], value='medio', style={'marginBottom': '15px'})
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                ]),
                
                html.Div(id='association-result', style={'marginTop': '30px'})
            ], style={'padding': '20px'})
        ])
    ])
], style={'fontFamily': 'Arial, sans-serif', 'margin': '0 auto', 'maxWidth': '1200px'})

# -----------------------
# 🔹 CALLBACKS
# -----------------------

# Callback para regresión individual
@app.callback(
    Output('regression-result', 'children'),
    [Input('gdp-slider', 'value'),
     Input('social-slider', 'value'),
     Input('health-slider', 'value'),
     Input('freedom-slider', 'value'),
     Input('generosity-slider', 'value'),
     Input('corruption-slider', 'value')]
)
def update_regression_individual(gdp, social, health, freedom, generosity, corruption):
    if modelo_regresion is None:
        return "❌ Modelo no disponible"
    
    try:
        # Crear DataFrame con los datos ingresados
        X_input = pd.DataFrame([[gdp, social, health, freedom, generosity, corruption]], 
                              columns=['GDP per capita', 'Social support', 'Healthy life expectancy',
                                     'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])
        
        # Hacer predicción
        prediccion = modelo_regresion.predict(X_input)[0]
        
        # Interpretar resultado
        if prediccion >= 7.0:
            categoria = "🌟 MUY FELIZ"
            color = "#27ae60"
        elif prediccion >= 5.5:
            categoria = "😊 FELIZ"
            color = "#f39c12"
        elif prediccion >= 4.0:
            categoria = "😐 MODERADO"
            color = "#e67e22"
        else:
            categoria = "😞 POCO FELIZ"
            color = "#e74c3c"
        
        return html.Div([
            html.H3(f"🎯 Índice de Felicidad Predicho: {prediccion:.3f}", 
                   style={'color': color, 'marginBottom': '10px'}),
            html.H4(f"Categoría: {categoria}", style={'color': color}),
            html.P("Escala: 0 (muy infeliz) - 10 (muy feliz)", style={'fontSize': '14px', 'color': '#7f8c8d'})
        ])
        
    except Exception as e:
        return f"❌ Error en predicción: {str(e)}"

# Callback para comparación de países
@app.callback(
    Output('comparison-result', 'children'),
    [Input('pais1-nombre', 'value'), Input('pais1-gdp', 'value'), Input('pais1-social', 'value'),
     Input('pais1-health', 'value'), Input('pais1-freedom', 'value'), Input('pais1-generosity', 'value'),
     Input('pais1-corruption', 'value'), Input('pais2-nombre', 'value'), Input('pais2-gdp', 'value'),
     Input('pais2-social', 'value'), Input('pais2-health', 'value'), Input('pais2-freedom', 'value'),
     Input('pais2-generosity', 'value'), Input('pais2-corruption', 'value'), Input('pais3-nombre', 'value'),
     Input('pais3-gdp', 'value'), Input('pais3-social', 'value'), Input('pais3-health', 'value'),
     Input('pais3-freedom', 'value'), Input('pais3-generosity', 'value'), Input('pais3-corruption', 'value')]
)
def update_country_comparison(*args):
    if modelo_regresion is None:
        return "❌ Modelo no disponible"
    
    try:
        # Organizar datos de los 3 países
        paises_data = []
        nombres = []
        
        for i in range(3):
            base_idx = i * 7
            nombre = args[base_idx]
            datos = list(args[base_idx + 1:base_idx + 7])
            paises_data.append(datos)
            nombres.append(nombre)
        
        # Crear DataFrame
        df_paises = pd.DataFrame(paises_data, 
                               columns=['GDP per capita', 'Social support', 'Healthy life expectancy',
                                      'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])
        df_paises['País'] = nombres
        
        # Hacer predicciones
        predicciones = modelo_regresion.predict(df_paises.drop('País', axis=1))
        df_paises['Felicidad'] = predicciones
        
        # Crear gráfico de comparación
        fig = px.bar(df_paises, x='País', y='Felicidad', 
                    title="Comparación de Índices de Felicidad",
                    color='Felicidad', color_continuous_scale='viridis')
        fig.update_layout(plot_bgcolor='white')
        
        # Ranking
        df_ranking = df_paises.sort_values('Felicidad', ascending=False).reset_index(drop=True)
        df_ranking['Posición'] = range(1, len(df_ranking) + 1)
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.H4("🏆 Ranking de Felicidad:"),
            html.Div([
                html.P(f"{row['Posición']}. {row['País']}: {row['Felicidad']:.3f} puntos", 
                      style={'fontSize': '16px', 'margin': '5px'})
                for _, row in df_ranking.iterrows()
            ])
        ])
        
    except Exception as e:
        return f"❌ Error en comparación: {str(e)}"

# Callback para clasificación
@app.callback(
    Output('classification-result', 'children'),
    [Input('edad-input', 'value'), Input('workclass-dropdown', 'value'),
     Input('education-dropdown', 'value'), Input('marital-dropdown', 'value'),
     Input('occupation-dropdown', 'value'), Input('sex-dropdown', 'value'),
     Input('hours-input', 'value'), Input('country-dropdown', 'value')]
)
def update_classification(edad, workclass, education, marital, occupation, sex, hours, country):
    if modelo_clasificacion is None or label_encoders is None or income_encoder is None:
        return "❌ Modelo no disponible"
    
    try:
        # Crear DataFrame con datos completos (incluyendo valores por defecto)
        X_input = pd.DataFrame([{
            'age': edad,
            'workclass': workclass,
            'fnlwgt': 77516,  # Valor por defecto
            'education': education,
            'education-num': 13,  # Valor por defecto
            'marital-status': marital,
            'occupation': occupation,
            'relationship': 'Not-in-family',  # Valor por defecto
            'race': 'White',  # Valor por defecto
            'sex': sex,
            'capital-gain': 0,  # Valor por defecto
            'capital-loss': 0,  # Valor por defecto
            'hours-per-week': hours,
            'native-country': country
        }])
        
        # Codificar variables categóricas
        X_encoded = X_input.copy()
        for col, le in label_encoders.items():
            if col in X_encoded.columns:
                try:
                    X_encoded[col] = le.transform(X_encoded[col])
                except ValueError:
                    # Si el valor no existe en el encoder, usar 0
                    X_encoded[col] = 0
        
        # Predicción
        y_pred = modelo_clasificacion.predict(X_encoded)
        y_pred_proba = modelo_clasificacion.predict_proba(X_encoded)
        resultado = income_encoder.inverse_transform(y_pred)[0]
        confianza = y_pred_proba[0].max()
        
        # Interpretar resultado
        color = "#27ae60" if resultado == ">50K" else "#e74c3c"
        emoji = "💰" if resultado == ">50K" else "💼"
        
        return html.Div([
            html.H3(f"{emoji} Predicción: {resultado}", style={'color': color}),
            html.P(f"🎯 Confianza del modelo: {confianza:.1%}", style={'fontSize': '16px'}),
            html.P(f"📊 Probabilidad >50K: {y_pred_proba[0][1]:.1%}", style={'fontSize': '14px'}),
            html.P(f"📊 Probabilidad ≤50K: {y_pred_proba[0][0]:.1%}", style={'fontSize': '14px'})
        ])
        
    except Exception as e:
        return f"❌ Error en clasificación: {str(e)}"

# Callback para clustering individual
@app.callback(
    Output('clustering-individual-result', 'children'),
    [Input('gasto-input', 'value'),
     Input('transacciones-input', 'value'),
     Input('productos-input', 'value')]
)
def update_clustering_individual(gasto, transacciones, productos):
    try:
        # Determinar cluster basado en reglas simples
        if gasto < 250:
            cluster = 0
            label = "Básico"
            color = "#e74c3c"
            emoji = "🔵"
            descripcion = "Cliente con gastos moderados y poca frecuencia de compra"
        elif gasto < 600:
            cluster = 1
            label = "Regular"
            color = "#f39c12"
            emoji = "🟡"
            descripcion = "Cliente con gastos medios y frecuencia moderada"
        else:
            cluster = 2
            label = "Premium"
            color = "#27ae60"
            emoji = "🟢"
            descripcion = "Cliente de alto valor con gastos elevados"
        
        # Calcular métricas adicionales
        gasto_promedio_transaccion = gasto / transacciones if transacciones > 0 else 0
        productos_por_transaccion = productos / transacciones if transacciones > 0 else 0
        
        return html.Div([
            html.H3(f"{emoji} Segmento: {label}", style={'color': color}),
            html.P(descripcion, style={'fontSize': '14px', 'marginBottom': '15px'}),
            html.Hr(),
            html.P(f"💰 Gasto promedio por transacción: ${gasto_promedio_transaccion:.2f}"),
            html.P(f"📦 Productos por transacción: {productos_por_transaccion:.1f}"),
            html.P(f"🔄 Frecuencia de compra: {transacciones} transacciones"),
            html.P(f"📊 Total de productos: {productos} productos")
        ])
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Callback para clustering múltiple
@app.callback(
    Output('clustering-multiple-result', 'children'),
    [Input('cliente1-nombre', 'value'), Input('cliente1-gasto', 'value'),
     Input('cliente1-trans', 'value'), Input('cliente1-prod', 'value'),
     Input('cliente2-nombre', 'value'), Input('cliente2-gasto', 'value'),
     Input('cliente2-trans', 'value'), Input('cliente2-prod', 'value'),
     Input('cliente3-nombre', 'value'), Input('cliente3-gasto', 'value'),
     Input('cliente3-trans', 'value'), Input('cliente3-prod', 'value'),
     Input('cliente4-nombre', 'value'), Input('cliente4-gasto', 'value'),
     Input('cliente4-trans', 'value'), Input('cliente4-prod', 'value')]
)
def update_clustering_multiple(*args):
    try:
        # Organizar datos de los 4 clientes
        clientes_data = []
        nombres = []
        
        for i in range(4):
            base_idx = i * 4
            nombre = args[base_idx]
            gasto = args[base_idx + 1]
            transacciones = args[base_idx + 2]
            productos = args[base_idx + 3]
            
            clientes_data.append([gasto, transacciones, productos])
            nombres.append(nombre)
        
        # Crear DataFrame
        df_clientes = pd.DataFrame(clientes_data, columns=['Gasto', 'Transacciones', 'Productos'])
        df_clientes['Cliente'] = nombres
        
        # Asignar clusters
        def asignar_cluster(gasto):
            if gasto < 250:
                return "Básico"
            elif gasto < 600:
                return "Regular"
            else:
                return "Premium"
        
        df_clientes['Segmento'] = df_clientes['Gasto'].apply(asignar_cluster)
        
        # Crear gráfico
        fig = px.scatter(df_clientes, x='Gasto', y='Transacciones', 
                        size='Productos', color='Segmento',
                        hover_data=['Cliente'],
                        title="Comparación de Clientes por Segmento",
                        color_discrete_map={'Básico': '#e74c3c', 'Regular': '#f39c12', 'Premium': '#27ae60'})
        fig.update_layout(plot_bgcolor='white')
        
        # Crear tabla resumen
        resumen = df_clientes.groupby('Segmento').agg({
            'Gasto': 'mean',
            'Transacciones': 'mean',
            'Productos': 'mean',
            'Cliente': 'count'
        }).round(2)
        resumen.columns = ['Gasto Promedio', 'Transacciones Promedio', 'Productos Promedio', 'Cantidad']
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.H4("📊 Resumen por Segmento:"),
            html.Pre(resumen.to_string(), style={'backgroundColor': '#f8f9fa', 'padding': '15px'})
        ])
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Callback para reglas de asociación
@app.callback(
    Output('association-result', 'children'),
    [Input('co-dropdown', 'value'), Input('nox-dropdown', 'value'),
     Input('no2-dropdown', 'value'), Input('temp-dropdown', 'value'),
     Input('humedad-dropdown', 'value'), Input('benceno-dropdown', 'value')]
)
def update_association_rules(co, nox, no2, temp, humedad, benceno):
    try:
        # Simular base de reglas de asociación
        reglas_conocidas = {
            ('co_alto', 'nox_alto'): {'confianza': 0.85, 'lift': 2.3, 'descripcion': 'CO alto está fuertemente asociado con NOx alto'},
            ('temp_alto', 'humedad_bajo'): {'confianza': 0.75, 'lift': 1.8, 'descripcion': 'Temperatura alta tiende a correlacionarse con baja humedad'},
            ('nox_alto', 'no2_alto'): {'confianza': 0.88, 'lift': 2.5, 'descripcion': 'NOx alto predice fuertemente NO2 alto'},
            ('benceno_alto', 'co_alto'): {'confianza': 0.70, 'lift': 1.9, 'descripcion': 'Benceno alto se asocia con CO alto'},
            ('humedad_alto', 'temp_bajo'): {'confianza': 0.82, 'lift': 1.7, 'descripcion': 'Alta humedad se asocia con temperatura baja'}
        }
        
        # Condiciones actuales
        condiciones = [f'co_{co}', f'nox_{nox}', f'no2_{no2}', f'temp_{temp}', f'humedad_{humedad}', f'benceno_{benceno}']
        
        # Buscar reglas aplicables
        reglas_aplicables = []
        
        for (antecedente, consecuente), info in reglas_conocidas.items():
            if antecedente in condiciones:
                reglas_aplicables.append({
                    'regla': f'{antecedente} → {consecuente}',
                    'confianza': info['confianza'],
                    'lift': info['lift'],
                    'descripcion': info['descripcion'],
                    'aplicable': True
                })
        
        # Si no hay reglas aplicables, mostrar algunas reglas generales
        if not reglas_aplicables:
            reglas_aplicables = [
                {'regla': 'co_medio → nox_medio', 'confianza': 0.65, 'lift': 1.4, 
                 'descripcion': 'Condiciones moderadas de CO tienden a asociarse con NOx moderado', 'aplicable': False}
            ]
        
        # Crear visualización
        resultado = html.Div([
            html.H3("🔍 Reglas de Asociación Encontradas:", style={'color': '#f39c12'}),
            html.P(f"📊 Condiciones actuales: {', '.join(condiciones)}", 
                  style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px'}),
            html.Hr()
        ])
        
        for regla in reglas_aplicables:
            color = "#27ae60" if regla['aplicable'] else "#95a5a6"
            icono = "✅" if regla['aplicable'] else "ℹ️"
            
            resultado.children.append(
                html.Div([
                    html.H4(f"{icono} {regla['regla']}", style={'color': color}),
                    html.P(f"🎯 Confianza: {regla['confianza']:.1%}"),
                    html.P(f"📈 Lift: {regla['lift']:.1f}"),
                    html.P(f"💡 {regla['descripcion']}", style={'fontStyle': 'italic'}),
                    html.Hr()
                ], style={'backgroundColor': '#fff3cd' if regla['aplicable'] else '#f8f9fa', 
                         'padding': '15px', 'borderRadius': '10px', 'marginBottom': '10px'})
            )
        
        return resultado
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    import os
    print("🚀 Iniciando dashboard...")
    print("🌐 Dashboard disponible en: http://127.0.0.1:8050")
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=True
    )

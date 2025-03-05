print('\n-----------------------')
import numpy as np
import streamlit as st
import requests
import plotly.express as px
# import plotly.graph_objects as go
from plotly.subplots import make_subplots

from SensorData import SensorDataProcessor

import pandas as pd

def load_a_file(file_path):
    df = pd.read_csv(file_path, skiprows=6, delimiter=';')
    df.columns = ['Time', 'Wind_speed', 'Power_sensor',
                        'Pitch_angle_1', 'Pitch_angle_2', 'Pitch_angle_3',
                        'Rotor_speed_sensor', 'Generator_speed_sensor', 'Torque_sensor', 'Unnamed']
    df.drop(columns=['Unnamed'], inplace=True)
    df = df.set_index('Time')
    group = df.groupby(np.arange(len(df)) // 10) # Agrupar e calcular mean e std
    df = group.agg(['mean', 'max', 'min', 'std'])
    df.columns = ['{}_{}'.format(col, stat) for col, stat in df.columns]
    return df

def load_train():
    csv_files = [f for f in os.listdir("Train_data") if f.endswith(".csv")]
    dfi = []
    for csv_file in csv_files:
        file_path = os.path.join("Train_data", csv_file)
        sensor_data = SensorDataProcessor(file_path, 10)
        sensor_data.load_and_process()
        sensor_data.feature_extraction()
        sensor_data.labelling()
        dfi.append(sensor_data.get_data())
    Train_df = pd.concat(dfi, ignore_index=True)
    Train_df.dropna(inplace=True)
    return Train_df

# ==============================================
# === Functions to add faults to the dataset ===
# ==============================================
def pitch_fixed(df, duration):
    fault_start = 1000 # np.random.randint(0, len(df) - duration) # Random start
    df.loc[fault_start : fault_start + duration, "Pitch_angle_1_mean"] = 5 # = df.loc[fault_start - 1, 'Pitch_angle_1_mean'] # Ultimo valor antes da falha
    df.loc[fault_start : fault_start + duration, "Pitch_angle_1_std"] = 0
    df["y_true"] = 0
    df.loc[fault_start : fault_start + duration-1, "y_true"] = 1
    return df

def pitch_gain(df, duration):
    fault_start = 1000 # np.random.randint(0, len(df) - duration) # Random start
    df.loc[fault_start:fault_start+duration, "Pitch_angle_2_mean"] *= 1.2 # Aumenta o ganho em 20%
    # Adicionar o efeito na Potência e nas velocidades angulares.
    df["y_true"] = 0
    df.loc[fault_start : fault_start + duration-1, "y_true"] = 1
    return df

def pitch_drift(df, duration):
    fault_start = 1000 #np.random.randint(0, len(df) - duration)
    df.loc[fault_start : fault_start + duration, "Pitch_angle_3_mean"] += np.linspace(2, 4, duration + 1) # Adiciona uma tendência crescente
    # Adicionar o efeito na Potência e nas velocidades angulares.
    df["y_true"] = 0
    df.loc[fault_start : fault_start + duration-1, "y_true"] = 1
    return df

def omega_fixed(df, duration):
    fault_start = 1600 # np.random.randint(0, len(df) - duration)
    df.loc[fault_start : fault_start + duration-1, "Generator_speed_sensor_mean"] = 50 # Define um valor fixo no RPM
    df.loc[fault_start : fault_start + duration-1, "Generator_speed_sensor_std"] = 0
    # Generator_speed_sensor_mean
    # Adicionar o efeito na Potência e nas velocidades angulares.
    df["y_true"] = 0
    df.loc[fault_start : fault_start + duration-1, "y_true"] = 1
    return df 

def omega_gain(df, duration):
    fault_start = 1500 # np.random.randint(0, len(df) - duration)
    df.loc[fault_start : fault_start + duration, "Rotor_speed_sensor_mean"] *= 1.2 # Gain no RPM
    # Adicionar o efeito na Potência e nas velocidades angulares.
    df["y_true"] = 0
    df.loc[fault_start : fault_start + duration-1, "y_true"] = 1
    return df

# Fault selector to insert it in the dataframe of the plotting
def fault_selector(add_select_fault, df, add_duration):
    if add_select_fault == 'Encoder with a fixed value':
        df = pitch_fixed(df, add_duration)
    elif add_select_fault == 'Encoder with gain':
        df = pitch_gain(df, add_duration)
    elif add_select_fault == 'Encoder with a drift':
        df = pitch_drift(df, add_duration)
    elif add_select_fault == 'Tachometer with a fixed value':
        df = omega_fixed(df, add_duration)
    elif add_select_fault == 'Tachometer with a gain':
        df = omega_gain(df, add_duration)
    elif add_select_fault == 'No fault: healthy condition':
        df["y_true"] = 0
    else:
        pass
    ## Add as many faults as you want
    return df

# ==============================================
# ============ Page configurations =============
# ==============================================
st.set_page_config(
    page_title="WT-KBDT app",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''This is an application resulting from a doctoral thesis in development.
        If you have any questions, please contact the author.'''
    }
)

# ==============================================
# =============== Sidebar window ===============
# ==============================================
url = "https://raw.githubusercontent.com/cesartadeub/kbdt/main/users_guide.pdf"# User's document download:
response = requests.get(url)

if response.status_code == 200:
    with st.sidebar:
        st.download_button(
            label="Download user's guide",
            data=response.content,
            file_name="users_guide.pdf",
            mime="application/pdf"
        )
else:
    st.sidebar.error("Failed to fetch the file.")

# Add a text to the sidebar:
add_sidebar_title = st.sidebar.write('''
                                     # Parameters
                                     ''')
# Add a select box:
add_select_turbine = st.sidebar.radio(
    'Select a wind turbine dataset',
    ('2MW', '4.8MW', '10MW'), disabled = False
)

add_select_fault = st.sidebar.selectbox(
    'Select a condition',
    ('No fault: healthy condition',
     'Encoder with a fixed value', 'Encoder with gain', 'Encoder with a drift',
     'Tachometer with a fixed value', 'Tachometer with a gain'), placeholder='Chose an option'
)

add_duration = st.sidebar.slider("Fault duration (s):",
                                     min_value=2, max_value=10, value=6, step=1)

# ==============================================
# ================= Main window ================
# ==============================================
st.title("""
         Wind turbine knowledge-based digital twin sensor faults diagnosis
         *_Cesar Tadeu NM Branco v0.25.03.1_*
         """)

if add_select_turbine == '4.8MW':
    df = load_a_file(file_path = 'https://raw.githubusercontent.com/cesartadeub/kbdt/refs/heads/main/turbine_dataset/dataset_4800.csv')
    # Fault selector acts below
    df = fault_selector(add_select_fault, df, add_duration)
    
    # Plotting power curve
    WsxP = px.scatter(df, x="Wind_speed_mean", y="Power_sensor_mean",
    labels={"Wind_speed_mean": "Wind speed (m/s)", "Power_sensor_mean": "Power generated (MW)"})
    WsxP.update_layout(title=f"{add_select_turbine} wind turbine power curve")
    WsxP.update_traces(hovertemplate='<b>P:</b> %{y:.1f} MW<br>'+
                       '<b>Ws: </b>%{x:.1f} m/s')
    st.plotly_chart(WsxP, on_select="rerun", color = [77, 183, 211])

    # Plotting blade angles
    pit = make_subplots(rows=1, cols=3,
                        subplot_titles=("Blade A (deg)", "Blade B (deg)", "Blade C (deg)"))
    pit1 = px.scatter(df, y="Pitch_angle_1_mean")
    pit2 = px.scatter(df, y="Pitch_angle_2_mean")
    pit3 = px.scatter(df, y="Pitch_angle_3_mean")
    pit.add_trace(pit1.data[0], row=1, col=1)
    pit.add_trace(pit2.data[0], row=1, col=2)
    pit.add_trace(pit3.data[0], row=1, col=3)
    pit.update_traces(hovertemplate=
                      '<b>Angle:</b> %{y:.2f} deg<br>'+
                      '<b>Time:</b> %{x:.0f} s')

    pit.update_layout(title=f"{add_select_turbine} wind turbine pitch angle",
        showlegend=False,
        xaxis_title="Time (s)",
        xaxis2_title="Time (s)",
        xaxis3_title="Time (s)"
        )
    st.plotly_chart(pit, on_select="rerun", use_container_width=True, color = [77, 183, 211])

    # Plotting turbine speed
    omega = make_subplots(rows=1, cols=2,
                          subplot_titles=("Rotor speed (rpm)", "Generator speed (rpm)"))
    om1 = px.scatter(df, y="Rotor_speed_sensor_mean")
    om2 = px.scatter(df, y="Generator_speed_sensor_mean")
    omega.add_trace(om1.data[0], row=1, col=1)
    omega.add_trace(om2.data[0], row=1, col=2)
    omega.update_traces(hovertemplate=
                        '<b>Speed:</b> %{y:.2f} rpm<br>'+
                        '<b>Time:</b> %{x:.0f} s')
    omega.update_layout(showlegend=False)
    omega.update_layout(title=f"{add_select_turbine} wind turbine speeds",
        showlegend=False,
        xaxis_title="Time (s)",
        xaxis2_title="Time (s)",
        )
    st.plotly_chart(omega, on_select="rerun", use_container_width=True, color = [77, 183, 211])
else:
    st.text('To be released soon')

# ==============================================
# =========== Running the application ==========
# ==============================================
# Action to trigger the button
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sensor_inference import SensorInferenceSystem

if st.sidebar.button('Run analysis'): # A sidebar button to trigger KB and ML analysis
    st.markdown('# Carrying knowledge-based and machine learning analysis')
    # st.write('...Loading data files for training')
    # Train_df = load_train()
    # Test_df = df

    # Criar o detector de anomalias
    detector = SensorInferenceSystem(df)

    # Executar o diagnóstico
    description, cm, accuracy, pod, pofa = detector.run()

    # Exibir no Streamlit
    st.write("### Wind turbine diagnosis")
    st.write(description)

    # Exibir a matriz de confusão
    st.write("### Metrics")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_labels = ["Healthy", "Faulty"]
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "jet",
                xticklabels = cm_labels, yticklabels = cm_labels, ax = ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    col1, col2, col3, col4 = st.columns(4, vertical_alignment = "center")
    col1.pyplot(fig)
    col2.metric("Accuracy", f"{accuracy:.2%}")
    col3.metric("Probability of detection", f"{pod:.2%}")
    col4.metric("Probability of a false alarm", f"{pofa:.2%}")

    st.write("### Key performance indicators")
    a, b, = st.columns(2); c, d, = st.columns(2)

    a.metric("Mean Sensor Deviation", "77 deg", "5%", border=True) # Devio médio do sinal do sensor real para o sensor virtual.
    b.metric("Sensor Ratio Efficiency", "77%", "5%", border=True)
    c.metric("Efficiency Loss Factor", "4 mph", "2 mph", border=True)
    d.metric("Energy Production Loss", "30°F", "-9°F", border=True)

    # # PUT THE SHOW ON THE ROAD
    # # Separando features e rótulos do dataset de treino
    # X = Train_df.drop(columns=['Label'])  # Removendo a coluna alvo
    # y = Train_df['Label']  # Definindo a variável alvo
    # # Dividir os dados em treino (80%) e validação (20%) para avaliação
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # # Criando o pipeline do modelo Random Forest
    # rf_model = make_pipeline(
    #     StandardScaler(),
    #     RandomForestClassifier(
    #         max_depth = 12,
    #         min_samples_split = 8),
    # )
    # st.write("Training a Random Forest model")
    # # Treinando o modelo
    # rf_model.fit(X_train, y_train)
    # train_score = rf_model.score(X_train, y_train)
    # val_score = rf_model.score(X_val, y_val)
    # st.write(f'Train accuracy: {train_score:.2%}')
    # st.write(f'Val accuracy: {val_score:.2%}')

# ==============================================
# ================ Link to rate ================
# ==============================================

link_to_validate = "https://docs.google.com/forms/d/e/1FAIpQLSfFn0pFWvAkbig2Oo3gOkTo0WeRRey4-Q1ymBEJvs_c1H_iwg/viewform?usp=dialog"
st.sidebar.page_link(link_to_validate, label="Rate this app", icon="✅")
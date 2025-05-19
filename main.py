print('\n-----------------------')
import numpy as np
import pandas as pd
import streamlit as st
import requests

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# User module
from pitch_controller import PitchControllerComponent, Sensor
from real_data import WindTurbineTest, WindTurbine2000, WindTurbine4800
from inference_engine import SensorInferenceSystem, SensorFaultDetector

# ==============================================
# ==== Question and likert scale functions =====
# ==============================================

def query(question, key=None):
    # Asking a question
    user_response = st.text_input(question, key=key)    
    return user_response

def likert(features):
    st.subheader("KBDT system evaluation - Likert Scale")
    
    st.markdown("""
    For each item below, rate your level of agreement using the star scale on the left.
    You can also add optional comments in the field next to each rating.
    """)
    
    st.markdown("Use the scale below to guide your responses:")
    st.markdown("""
    - 1 = Strongly disagree  
    - 2 = Disagree  
    - 3 = Neutral  
    - 4 = Agree  
    - 5 = Strongly agree
    """)
    
    results = {}

    for feature in features:
        st.markdown(f"**{feature}**")
        col1, col2 = st.columns([1, 2]) # Column 1: rating | Column 2: comment

        with col1:
            selected = st.feedback("stars", key=feature)
        
        with col2:
            comment = st.text_input(
                f"Comment on '{feature}'",
                key=f"{feature}_comment",
                label_visibility="collapsed"
            )

        score = selected + 1 if selected is not None else None
        results[feature] = {"score": score, "comment": comment}

    return results


# Store the results
user_response = {}

# ==============================================
# === Functions to add faults to the dataset ===
# ==============================================
def pitch_fixed(df, duration):
    fault_start = 3000 #np.random.randint(0, len(df) - duration) # Random start
    df.loc[fault_start : fault_start + duration, "Pitch_angle_1_mean"] = 5 # = df.loc[fault_start - 1, 'Pitch_angle_1_mean'] # Ultimo valor antes da falha
    df.loc[fault_start : fault_start + duration, "Pitch_angle_1_std"] = 0
    df.loc[fault_start : fault_start + duration, "Power_sensor_mean"] /= 1.2 # Adiciona o efeito na Potência
    df.loc[fault_start : fault_start + duration, "Rotor_speed_sensor_mean"] /= 1.2 # Adiciona o efeito no gerador
    df.loc[fault_start : fault_start + duration, "Generator_speed_sensor_mean"] /= 1.2 # Adiciona o efeito no rotor de pás
    df["y_true"] = 0
    df.loc[fault_start : fault_start + duration-1, "y_true"] = 1
    return df 
 
def pitch_gain(df, duration):
    fault_start = 1000 # np.random.randint(0, len(df) - duration) # Random start
    df.loc[fault_start:fault_start+duration, "Pitch_angle_2_mean"] *= 1.2 # Aumenta o ganho em 20%
    df.loc[fault_start : fault_start + duration, "Power_sensor_mean"] /= 1.2 # Adiciona o efeito na Potência
    # Adicionar o efeito na Potência e nas velocidades angulares.
    df["y_true"] = 0
    df.loc[fault_start : fault_start + duration-1, "y_true"] = 1
    return df

def pitch_drift(df, duration):
    fault_start = 1000 #np.random.randint(0, len(df) - duration)
    df.loc[fault_start : fault_start + duration, "Pitch_angle_3_mean"] += np.linspace(2, 4, duration + 1) # Adiciona uma tendência crescente
    # Adicionar o efeito na Potência e nas velocidades angulares.
    # Potência: diminui de forma decrescente
    # Omega: diminui de forma decrescente
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
    df = df.copy() # Create a copy of the DataFrame to avoid SettingWithCopyWarning
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
url = "https://raw.githubusercontent.com/cesartadeub/kbdt/main/guia_usuario.pdf"# User's document download:
response = requests.get(url)

if response.status_code == 200:
    with st.sidebar:
        st.download_button(
            label="Download user's guide",
            data=response.content,
            file_name="guia_usuario.pdf",
            mime="application/pdf"
        )
else:
    st.sidebar.error("Failed to fetch the file.")

# Criando os campos de entrada
name = st.sidebar.text_input("Enter your name:")
role = st.sidebar.text_input("Enter your role:")

# Add a text to the sidebar:
add_sidebar_title = st.sidebar.write('''
                                     # Parameters
                                     ''')
# Add a select box:
add_select_turbine = st.sidebar.radio(
    'Select a wind turbine dataset',
    ('Data test','4.8MW (Benchmarking)', '2.0MW (Kelmarsh)'), disabled = False
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
         Wind turbine knowledge-based digital twin sensor fault diagnosis
         *_Cesar Tadeu NM Branco v0.25.05.3_*
         """)
st.text(" Welcome to the KBDT prototype for conditional monitoring of wind turbines. In the window on the left, download the user guide by clicking the button and see what can be done with the prototype in question.")
if add_select_turbine == '2.0MW (Kelmarsh)':
    #data_link = 'https://raw.githubusercontent.com/cesartadeub/kbdt/refs/heads/main/turbine_dataset/dataset_2100_Kelmarsh.csv'
    data_link = 'turbine_dataset/dataset_2100_Kelmarsh.csv'
    pc = PitchControllerComponent(data_link, 10)
    pc.load_and_preprocess() # Carrega e limpa os dados
    pc.extract_features() # Extrai as estatísticas dos dados

    df = pc.get_data() # Obtendo os dados processados
    df = fault_selector(add_select_fault, df, add_duration) # Inject a fault

    # Creating the wind turbine visualization
    data_analysis = WindTurbine2000(df, add_select_turbine, add_select_fault, add_duration)
    data_analysis.display_plots()

    # Storing the outliers to analysis
    dv = data_analysis.generate_dataframe()
    st.write('''###### The digital twin was uploaded successfully!''')

    # Proceed to fault analysis
    detector = SensorInferenceSystem(df, 0.01, 3, 119.6, dv)

elif add_select_turbine == '4.8MW (Benchmarking)':
    data_link = 'https://raw.githubusercontent.com/cesartadeub/kbdt/refs/heads/main/turbine_dataset/dataset_4800.csv'
    # Criando uma instância da classe
    pc = PitchControllerComponent(data_link, 10)
    pc.load_and_preprocess() # Carrega e limpa os dados
    pc.extract_features() # Extrai as estatísticas dos dados
    df = pc.get_data() # Obtendo os dados processados
    df = fault_selector(add_select_fault, df, add_duration) # Inject a fault
    
    # Creating the wind turbine visualization
    data_analysis = WindTurbine4800(df, add_select_turbine, add_select_fault, add_duration)
    data_analysis.display_plots()

    # Storing the outliers to analysis
    dv = data_analysis.generate_dataframe()
    st.write('''###### The digital twin was uploaded successfully!''')

    # Proceed to fault analysis
    detector = SensorInferenceSystem(df, 0.01, 3, 95, dv)
    
elif add_select_turbine == 'Data test':
    st.write('''# Data analysis''')
    intro_text = (
        f'Here are results from a sensitivity analysis, considering constant wind speed with different intensities and in different regions of wind turbine operation.\n\n'
        'Therefore, choose a dataset below considering the wind speed and the turbine operating zone.'
        )
    with st.container():
        select_test_dataset = st.radio(intro_text,
            ["Wind speed: 5m/s (under rated speed)",
            "Wind speed: 6m/s (under rated speed)",
            "Wind speed: 7m/s (under rated speed)",
            "Wind speed: 12m/s (above rated speed)",
            "Wind speed: 13m/s (above rated speed)",
            "Wind speed: 14m/s (above rated speed)"], index=0, horizontal=True)

    if select_test_dataset is not None:
        speed_mapping = {
            "Wind speed: 5m/s (under rated speed)": "Train_data/set6.csv",
            "Wind speed: 6m/s (under rated speed)": "Train_data/set5.csv",
            "Wind speed: 7m/s (under rated speed)": "Train_data/set4.csv",
            "Wind speed: 12m/s (above rated speed)": "Train_data/set3.csv",
            "Wind speed: 13m/s (above rated speed)": "Train_data/set2.csv",
            "Wind speed: 14m/s (above rated speed)": "Train_data/set1.csv"}
        
        data_link = speed_mapping[select_test_dataset]
        @st.cache_data
        def load_data(data_link):
            sensor = Sensor(data_link, 10)
            sensor.load_and_preprocess()
            sensor.extract_features()
            sensor.label_data()
            return sensor.get_data()
        
        df_labeled = load_data(data_link)
    
    data_analysis = WindTurbineTest(df_labeled, add_select_turbine, None, None)
    data_analysis.display_plots()

    st.write("""### User's insights and comments""")
    q01 = query(
        "Do the statistical analyses and comments reflect the real operational behavior of wind turbines? Do the simulated faults in encoders and tachometers resemble current failure patterns observed in wind farms? Please elaborate on your assessment.", key='q01')
    user_response['Q1'] = q01 # Calling query function for user feedback

    # 2) Binary analysis
    st.write('''# Threshold tuning for rule-based diagnosis''')
    st.markdown("""
    In this section, you can explore and validate the **fault detection system parameters** implemented as part of the **wind turbine digital twin**.

    Although the system is pre-calibrated with default settings, you may interact with three key parameters:

    - **Window size:** the number of seconds used to verify signal constancy;
    - **Threshold:** the acceptable deviation to consider a value as constant;
    - **Consecutive points:** how many consecutive points must meet the constancy criterion to trigger the alarm.

    These controls allow you to observe:

    1. **How variations in these parameters affect the system's detection capability**, reflected by *Precision* and *Recall* metrics;
    2. **How the confusion matrix changes**, highlighting the impact of false detections and undetected faults.

    By experimenting with these settings, domain specialists can better understand the trade-offs between sensitivity and specificity in the detection process and validate whether the system's behavior aligns with realistic operational expectations.
    """)
    ################################
    ######## Falha 1 - Fixa ########
    ################################
    st.write("## 1) Encoder with a fixed fault")
    falha_b_fixa = df_labeled.iloc[1950:2150]
    col1, col2, col3 = st.columns([1, 2, 2])

    with col1: # -- COLUNA 1: sliders
        st.markdown("#### Parameters")
        window_size = st.selectbox("Window size (s)", options=[2, 6, 10], index=0, key="enc_fixed_window")
        consecutive_points = st.slider("Consecutive points", min_value=1, max_value=10, value=6, key="enc_fixed_points")
        threshold = st.slider("Threshold", min_value=0.01, max_value=0.1, step=0.005, value=0.035, key="enc_fixed_threshold")

    with col2: # -- COLUNA 2: gráfico de precision/recall
        thresholds = np.arange(0.01, 0.1, 0.005)
        window_sizes = [2, 6, 10]

        data_analysis = WindTurbineTest(falha_b_fixa, add_select_turbine, None, None)
        detector = SensorFaultDetector(falha_b_fixa)

        prec_res, rec_res = data_analysis.fixed_tradeoff(
            detector, falha_b_fixa, 'Pitch_angle_1_mean', window_sizes, thresholds, consecutive_points, 1)

        plot_fig = data_analysis.plot_precision_recall_graph(
            thresholds,
            prec_res[window_size],
            rec_res[window_size],
            window_size,
            threshold_setpoint=threshold)

        st.plotly_chart(plot_fig, use_container_width=True)

    with col3: # -- COLUNA 3: matriz de confusão atualizada
        y_pred, _, _ = detector.detect_constant_value(1, falha_b_fixa, 'Pitch_angle_1_mean',
                                                    window_size=window_size,
                                                    threshold=threshold,
                                                    consecutive_points=consecutive_points)
        y_true = falha_b_fixa['Label']
        cm = detector.binary_cm(y_true, y_pred, fault_label=1)
        fig_cm = data_analysis.plot_binary_cm(cm, fault_name='Beta-1 fixed')
        st.plotly_chart(fig_cm, use_container_width=True)

    ################################
    ####### Falha 2 - Ganho ########
    ################################
    st.write("## 2) Encoder with a gain fault")
    falha_ganho = df_labeled.iloc[2250:2450]
    col1, col2, col3 = st.columns([1, 2, 2])

    window_sizes = [2, 6, 10]
    thresholds = np.arange(0.0, 0.7, 0.05)
    threshold_setpoint = 0.15

    with col1:  # -- COLUNA 1: parâmetros
        st.markdown("#### Parameters")
        window_size = st.selectbox("Window size (s)", options=[2, 6, 10], index=1, key="enc_gain_window")
        consecutive_points = st.slider("Consecutive points", min_value=1, max_value=20, value=6, key="enc_gain_points")
        threshold = st.slider("Threshold", min_value=0.0, max_value=0.7, step=0.05, value=0.15, key="enc_gain_threshold")

    with col2:  # -- COLUNA 2: gráfico precision/recall
        data_analysis = WindTurbineTest(df_labeled, add_select_turbine, None, None)
        detector = SensorFaultDetector(df_labeled)
        
        precision_results, recall_results = data_analysis.gain_tradeoff(
            detector, falha_ganho,'Pitch_angle_1_mean', 'Pitch_angle_2_mean', None,
            window_sizes, thresholds, consecutive_points, 2)
        
        fig = data_analysis.plot_precision_recall_graph(
            thresholds,
            precision_results[window_size],
            recall_results[window_size],
            window_size,
            threshold_setpoint=threshold)
        st.plotly_chart(fig, use_container_width=True)

    with col3:  # -- COLUNA 3: matriz de confusão
        y_pred, _, _ = detector.detect_gain(
            2, falha_ganho, 'Pitch_angle_1_mean', 'Pitch_angle_2_mean',
            window_size=window_size,
            threshold=threshold,
            consecutive_points=consecutive_points)
        y_true = falha_ganho['Label']
        cm = detector.binary_cm(y_true, y_pred, fault_label=2)
        fig_cm = data_analysis.plot_binary_cm(cm, fault_name='Beta-2 gain')
        st.plotly_chart(fig_cm, use_container_width=True)

    ################################
    ############ Falha 3 ###########
    ################################
    st.write("## 3) Encoder with a trend")
    falha_trend = df_labeled.iloc[2550:2750]
    col1, col2, col3 = st.columns([1, 2, 2])

    window_sizes = [2, 6, 10]
    thresholds = np.arange(0.1, 2, 0.1)
    threshold_setpoint = 0.9

    with col1:  # -- COLUNA 1: parâmetros
        st.markdown("#### Parameters")
        window_size = st.selectbox("Window size (s)", options=[2, 6, 10], index=1, key="trend_window")
        threshold = st.slider("Slope threshold", min_value=0.1, max_value=2.0, step=0.1, value=0.9, key="trend_threshold")

    with col2:  # -- COLUNA 2: gráfico precision/recall
        precision_results, recall_results = data_analysis.trend_tradeoff(
            detector, falha_trend, window_sizes, thresholds, 3)
        
        fig = data_analysis.plot_precision_recall_graph(
            thresholds,
            precision_results[window_size],
            recall_results[window_size],
            window_size,
            threshold_setpoint=threshold)
        # Ajuste manual dos limites do eixo Y para tendência
        fig.update_yaxes(range=[0.9, 1.0])
        fig.update_yaxes(range=[0.9, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    with col3:  # -- COLUNA 3: matriz de confusão
        y_pred, _, _ = detector.detect_trend(3, falha_trend, 'Pitch_angle_3_mean',
                                             window_size=window_size,
                                             threshold=threshold)
        y_true = falha_trend['Label']
        cm = detector.binary_cm(y_true, y_pred, fault_label=3)
        fig_cm = data_analysis.plot_binary_cm(cm, fault_name='Beta-3 trend')
        st.plotly_chart(fig_cm, use_container_width=True)

    ################################
    ############ Falha 4 ###########
    ################################
    st.write("## 4) Tachometer with a fixed fault")
    falha_tach_fixed = df_labeled.iloc[1450:1650]
    col1, col2, col3 = st.columns([1, 2, 2])

    with col1: # -- COLUNA 1: sliders
        st.markdown("#### Parameters")
        window_size = st.selectbox("Window size (s)", options=[2,6,10], index=0, key="tach_fixed_window")
        consecutive_points = st.slider("Consecutive points", min_value=1, max_value=10, value=2, key="tach_fixed_points")
        threshold = st.slider("Threshold", min_value=0.03, max_value=0.08, step=0.002, value=0.04, key="tach_fixed_threshold")

    with col2: # -- COLUNA 2: gráfico de precision/recall
        thresholds = np.arange(0.03, 0.08, 0.002)
        window_sizes = [2, 6, 10]
        consecutive_points = 2

        data_analysis = WindTurbineTest(falha_tach_fixed, add_select_turbine, None, None)
        detector = SensorFaultDetector(falha_tach_fixed)

        prec_res, rec_res = data_analysis.fixed_tradeoff(
            detector, falha_tach_fixed,'Rotor_speed_sensor_mean', window_sizes, thresholds, consecutive_points, 4)

        plot_fig = data_analysis.plot_precision_recall_graph(
            thresholds,
            prec_res[window_size],
            rec_res[window_size],
            window_size,
            threshold_setpoint=threshold)

        st.plotly_chart(plot_fig, use_container_width=True)

    with col3: # -- COLUNA 3: matriz de confusão atualizada
        y_pred, _, _ = detector.detect_constant_value(4, falha_tach_fixed, 'Rotor_speed_sensor_mean',
                                                    window_size=window_size,
                                                    threshold=threshold,
                                                    consecutive_points=consecutive_points)
        y_true = falha_tach_fixed['Label']
        cm = detector.binary_cm(y_true, y_pred, fault_label=4)
        fig_cm = data_analysis.plot_binary_cm(cm, fault_name='Tach fixed')
        st.plotly_chart(fig_cm, use_container_width=True)

    ################################
    ############ Falha 5 ###########
    ################################
    st.write("## 5) Tachometer with a gain fault")
    falha_t_gain = df_labeled.iloc[350:550]
    col1, col2, col3 = st.columns([1, 2, 2])

    with col1: # -- COLUNA 1: sliders
        st.markdown("#### Parameters")
        window_size = st.selectbox("Window size (s)", options=[2, 6, 10], index=1, key="tach_gain_window")
        cps = st.slider("Consecutive points", min_value=1, max_value=20, value=8, key="tach_gain_points")
        th = st.slider("Threshold", min_value=0.01, max_value=0.1, step=0.005, value=0.035, key="tach_gain_threshold")

    with col2: # -- COLUNA 2: gráfico de precision/recall
        window_sizes = [2, 6, 10]
        thresholds = np.arange(0.01, 0.1, 0.005)
        threshold_setpoint = 0.035

        data_analysis = WindTurbineTest(falha_t_gain, add_select_turbine, None, None)
        detector = SensorFaultDetector(falha_t_gain)

        prec_res, rec_res = data_analysis.gain_tradeoff(
            detector, falha_t_gain, 'Rotor_speed_sensor_mean', 'Generator_speed_sensor_mean', 95,
            window_sizes, thresholds, cps, 5)

        fig = data_analysis.plot_precision_recall_graph(
            thresholds,
            precision_results[window_size],
            recall_results[window_size],
            window_size,
            threshold_setpoint=th)
        st.plotly_chart(fig, use_container_width=True)

    with col3:  # -- COLUNA 3: matriz de confusão
        y_pred, _, _ = detector.detect_gain(
            5, falha_t_gain, 'Rotor_speed_sensor_mean', 'Generator_speed_sensor_mean', 95,
            window_size=window_size,
            threshold=threshold,
            consecutive_points=consecutive_points)
        y_true = falha_t_gain['Label']
        cm = detector.binary_cm(y_true, y_pred, fault_label=5)
        fig_cm = data_analysis.plot_binary_cm(cm, fault_name='Tach gain')
        st.plotly_chart(fig_cm, use_container_width=True)
    
    st.write("""### User's insights and comments""")
    q02 = query(
                "Do the threshold settings and binary alerts reflect realistic maintenance scenarios in wind turbines? Would these parameter configurations support accurate and timely fault identification in a real operational environment? Are the monitored variables (e.g., pitch angle, rotor speed) suitable for the type of failures being diagnosed? Please elaborate on your assessment.", key='q02')
    user_response['Q2'] = q02 # Calling query function for user feedback

    st.write('''# Single-shot multiclass prediction''')
    st.markdown("""
    After defining statistical thresholds in the binary analysis stage, we now move toward a **multiclass classification**, which better reflects the practical challenges faced by wind turbine maintenance managers.  
    In real-world operations, it's not enough to detect that a fault may exist — it is also necessary to classify **what type of fault is occurring** to support timely and accurate maintenance actions.

    To accomplish this, we apply a single classification algorithm across the entire dataset, allowing the system to automatically identify **if a fault is present** and **which specific fault class it belongs to**.

    This multiclass analysis compares two complementary approaches:

    - **RBS (Rule-Based System):** The same logic-based system refined during the binary fault detection phase, now extended to handle multiple fault types.
    
    - **Stacking Ensemble:** An ensemble learning technique that combines the predictive power of multiple base models using a meta-model to make the final decision. This setup helps improve overall accuracy and generalization across various fault types.

        - The base predictors used include:
            - `Quadratic Discriminant Analysis`
            - `SVC Linear`
            - `Stochastic Gradient Descent Classifier`
            - `Decision Tree Classifier`
            - `Random Forest Classifier`    
            - `Extra Trees Classifier`
    
    - *Each predictor has been previously optimized offline using the best available hyperparameters.*

    To compare both approaches, simply click the **"Run stacking model"** button. *(Please note that the stacking model may take a few seconds to execute.)*

    The system will output a **confusion matrix** and key performance metrics such as **precision** and **recall**, allowing a direct performance comparison between the rule-based approach and the machine learning model.
    """)
    # MLAs matrix
    df6 = load_data('Train_data/set1.csv') # 14 m/s
    df5 = load_data('Train_data/set2.csv') # 13 m/s
    df4 = load_data('Train_data/set3.csv') # 12 m/s
    df3 = load_data('Train_data/set4.csv') # 7 m/s
    df2 = load_data('Train_data/set5.csv') # 6 m/s
    df1 = load_data('Train_data/set6.csv') # 5 m/s
    dfmla = pd.concat([df1,df2,df3,df4,df5,df6], axis=0)
    dfmla = dfmla.dropna()
    
    # EXPERIMENTA VER A CM DO RBS COM O dfmla - Recall baixo pro rbs
    # EXPERIMENTA BALANCEAR OS DADOS

    multi_data_analysis = WindTurbineTest(dfmla, add_select_turbine=None, add_select_fault=None, add_duration=None)
    detector = SensorFaultDetector(dfmla)
    y_pred = detector.detect_faults()
    y_true = dfmla['Label']
    cm_multi = detector.multiclass_cm(y_true, y_pred)
    fig_multi = multi_data_analysis.plot_multiclass_cm(cm_multi, 'Multiclass RBS confusion matrix')
    st.plotly_chart(fig_multi, use_container_width=True)
    # Two dictionaries were created to store the performances for posterior analysis
    recall_performance = {}
    precision_performance = {}
    rbs_rec, rbs_prec = detector.metrics_multi(y_true, y_pred)
    recall_performance['Rule-based'] = rbs_rec
    precision_performance['Rule-based'] = rbs_prec

    if st.button('Run stacking model'):
        with st.spinner('Running detection with stacked models... This may take a few seconds.'):
            y_test, y_pred_stack = detector.sa_mlas(dfmla) # Pode demorar um pouco
        st.success('Detection completed successfully!')
        cm_multi_mlas = detector.multiclass_cm(y_test, y_pred_stack)
        fig_multi_mlas = multi_data_analysis.plot_multiclass_cm(
            cm_multi_mlas, 'Multiclass Ensemble Learning Confusion Matrix')
        st.plotly_chart(fig_multi_mlas, use_container_width=True)
    # Compare RBS x STACK
        stc_rec, stc_prec = detector.metrics_multi(y_test, y_pred_stack) # Se ficar fora do If, dá erro !!
        recall_performance['Ensamble (Stacking)'] = stc_rec
        precision_performance['Ensamble (Stacking)'] = stc_prec
    
    st.write("""### Overall performance""")
    # Criando o DataFrame df_perf com os dados existentes
    ovl_perf = pd.DataFrame({
        'Model': recall_performance.keys(),
        'Recall': recall_performance.values(),
        'Precision': precision_performance.values(),
    })
    ovl_perf = ovl_perf.reset_index(drop=True)
    st.dataframe(ovl_perf)
    st.write("""### User's insights and comments""")
    q03 = query(
    "Does the system’s classification performance, as seen in the confusion matrices, provide an accurate and operationally useful representation of fault and healthy conditions? How do you interpret the trade-offs between detection rates and misclassifications across different fault types? In your experience, do the input variables and rule logic offer sufficient discriminatory power for real-time diagnostics? Would you recommend any changes in variable selection, rule thresholds, or system logic to improve fault differentiation and reduce diagnostic ambiguity?",
    key='q03'
)
    user_response['Q3'] = q03
else:
    st.text('To be released soon')

# # Calling query function for user feedback
# q1 = query("Write some remarkable comments regarding the statistical assessment made above,"
# "then rate the application from 1 to 5 below.")
# user_response['Q1'] = q1

# ==============================================
# =========== Running the application ==========
# ==============================================

# Action to trigger the button
if st.sidebar.button('Run analysis', key="run_analysis"): # A sidebar button to trigger KB analysis
    st.markdown('# Carrying knowledge-based analysis')

    # Executar o diagnóstico
    description, td, tn, fp, fn, tp = detector.run()

    # Exibir no Streamlit
    st.write("### Wind turbine diagnosis")
    st.write(description)
    cm_sum = tn + tp + fp + fn
    # Exibir a métricas
    st.write("### Metrics")
    col1, col2, col3, col4 = st.columns(4, vertical_alignment = "center")
    col1.metric("Detection time", f"{td:.2f}s")
    col2.metric("False Positive Rate", f"{fp/(fp+tn):.2%}") # Número de falsos positivos
    col3.metric("Miss detection rate", f"{fn/(fn+tp):.2%}")
    col4.metric("Accuracy", f"{(tn + tp)/cm_sum:.2%}")

    st.write("### Key performance indicators")
    a, b, = st.columns(2); c, d, = st.columns(2)

    a.metric("Mean Sensor Deviation", "77 deg", "5%", border=True) # Desvio médio do sinal do sensor real para o sensor virtual.
    b.metric("Sensor Ratio Efficiency", "77%", "5%", border=True)
    c.metric("Efficiency Loss Factor", "4%", "2%", border=True)
    d.metric("Energy Production Loss", "30kW", "-9kW", border=True)

    # Calling query function for user feedback
    q2 = query("Write some remarkable comments regarding the detection and diagnosis of the abrupt fault,"
    "then rate the application from 1 to 5 below.")
    user_response['Q2'] = q2

# ==============================================
# ================ Link to rate ================
# ==============================================
import io
features = [
    "Effectiveness of classification output",
    "Similarity of simulated failures to real-world conditions",
    "Clarity of diagnostic comments and visualizations",
    "Relevance of thresholds and parameters for diagnostics",
    "Ease of navigation and interface clarity",
    "Usefulness for maintenance planning and prioritization",
    "Transparency of inference process",
    "System responsiveness/speed",
    "Overall usefulness of the prototype",
    "Willingness to use the system in daily workflows"]
likert_score = likert(features)
import io
from unidecode import unidecode

# Coleta dos dados
likert_df = pd.DataFrame([
    {"Feature": feature, "Score": data["score"], "Comment": data["comment"]}
    for feature, data in likert_score.items() ])

likert_df = likert_df.reset_index(drop=True)
likert_df.set_index('Feature', inplace=True)

query_df = pd.DataFrame({
    'Question': user_response.keys(),
    'Comment': user_response.values() })
query_df = query_df.reset_index(drop=True)

# Exibir as duas tabelas no Streamlit
st.write("Review your Likert scores:")
st.dataframe(likert_df, use_container_width=True)

st.write("Review your open-ended responses:")
st.dataframe(query_df, use_container_width=True)

# Salvar tudo no mesmo CSV, com título separando
csv_buffer = io.BytesIO()

# Escrever texto separado manualmente em bytes com encoding utf-8-sig
csv_buffer.write("Likert scale scores\n".encode('utf-8-sig'))
likert_df.to_csv(csv_buffer, encoding='utf-8-sig')
csv_buffer.write("\nOpen-ended questions\n".encode('utf-8-sig'))
query_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

# Posição volta para o início para leitura posterior no download
csv_buffer.seek(0)

csv_bytes = csv_buffer.read()

# Botão de download
st.download_button(
    label="Download form",
    data=csv_bytes,
    file_name=(f"{unidecode(name).split()[0].lower()}_{role.lower()}_data.csv" if name.strip() else "unknown_user.csv"),
    mime="text/csv"
)

st.write("Fill in your details in the sidebar and on the main screen, then download the CSV file.")

print('\nOK!')
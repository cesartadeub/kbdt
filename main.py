print('\n-----------------------')
import numpy as np
import pandas as pd
import streamlit as st
import requests

import plotly.graph_objects as go
# User module
from pitch_controller import PitchControllerComponent, Sensor
from real_data import WindTurbineTest, WindTurbine2000, WindTurbine4800
from virtual_data import TurbineOptimizer
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
                                     # Wind turbine datasets
                                     ''')
# Add a select box:
add_select_turbine = st.sidebar.radio(
    'Choose one', (
        'Training data',
        '4.8MW (OpenFAST)',
        #'2.0MW (Kelmarsh)'
        ),
        disabled = False )

# ==============================================
# ================= Main window ================
# ==============================================
st.title("""
         Wind turbine knowledge-based digital twin fault diagnosis
         *_Cesar Tadeu NM Branco v0.25.06.2_*
         """)
st.text(" Welcome to the KBDT prototype for conditional monitoring of wind turbines. In the window on the left, download the user guide by clicking the button and see what can be done with the prototype in question.")

if add_select_turbine == 'Training data':
    st.write(""" # Presentation""")
    st.markdown("""
                In this first stage, you will explore a set of simulated datasets designed to support the development of a knowledge base for future machine learning applications in wind turbine fault diagnosis.
                
                The simulations were conducted for a 4.8MW turbine model under various wind speed regimes, aiming to replicate operational diversity. At this point, the focus is exclusively on the pitch control system, where multiple fault scenarios were deliberately introduced, including: encoder faults, tachometer faults and actuator faults.
                
                These datasets represent typical failure modes observed in real turbines and are intended to feed supervised and knowledge-based diagnostic models in later stages.
                
                As a domain expert, your role is to: examine the simulated data and fault behaviors; verify whether the selected input variables (e.g., pitch angle, rotor/generator speed, power output) are appropriate and sufficient for characterizing each type of fault; assess the consistency and usefulness of the statistical summaries, correlation results, and visual diagnostics provided; suggest improvements to increase the realism or diagnostic value of the simulated data.
                
                This stage is foundational for building reliable diagnostic models based on the digital twin paradigm. In the next stage, we will extend the analysis to include gearbox vibration data, enabling a more comprehensive drivetrain health assessment.
                """)
    
    st.write(""" # 1) Pitch controller""")
    st.write('''## Data analysis''')
    intro_text = (
        f'Here are results from a sensitivity analysis, considering constant wind speed with different intensities and in different regions of wind turbine operation.\n\n'
        'Therefore, choose a dataset below considering the wind speed and the turbine operating zone.'
        )
    with st.container():
        select_test_dataset = st.radio(intro_text,
            ["Wind speed: 5m/s (under rated speed)",
             "Wind speed: 6m/s (under rated speed)",
             "Wind speed: 8m/s (under rated speed)",
             "Wind speed: 10m/s (under rated speed)",
             "Wind speed: 12m/s (above rated speed)",
             "Wind speed: 15m/s (above rated speed)",
             "Wind speed: 17m/s (above rated speed)"], index=0, horizontal=True)

    if select_test_dataset is not None:
        speed_mapping = {
            "Wind speed: 5m/s (under rated speed)": "Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_7.csv",
            "Wind speed: 6m/s (under rated speed)": "Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_1.csv",
            "Wind speed: 8m/s (under rated speed)": "Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_2.csv",
            "Wind speed: 10m/s (under rated speed)": "Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_3.csv",
            "Wind speed: 12m/s (above rated speed)": "Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_4.csv",
            "Wind speed: 15m/s (above rated speed)": "Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_5.csv",
            "Wind speed: 17m/s (above rated speed)": "Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_6.csv"
            }
        
        data_link = speed_mapping[select_test_dataset]
        @st.cache_data
        def load_data(data_link):
            sensor = Sensor(data_link, 10)
            sensor.load_and_preprocess()
            sensor.extract_features()
            return sensor.label_data_train() 
        
        df_labeled = load_data(data_link)
    
    data_analysis = WindTurbineTest(df_labeled, add_select_turbine)
    data_analysis.display_plots()

    st.write("""### User's insights and comments""")
    q01 = query(
        "How do you evaluate the usefulness of the statistical analyses provided in supporting diagnostic decision-making? In your opinion, do the profiles in the synthetic datasets reasonably reflect the behavior of turbine variables observed in real wind farms? Given the lack of real datasets with labeled faults, to what extent do you believe these synthetic datasets are suitable for training fault classification systems?", key='q01')
    user_response['Q1'] = q01 # Calling query function for user feedback

    st.write('''## Fault analysis window''')
    st.markdown("""
    In this section, you will assess the **fault detection performance** of the pitch control system within the digital twin environment. The simulation introduces a set of predefined faults related to **encoders**, **tachometers**, and **pitch actuators** under controlled conditions.

    Rather than adjusting detection parameters, this stage focuses on analyzing how well the system responds to each fault scenario based on a set of predefined metrics. Each scenario includes a different fault type and timing, allowing for evaluation under diverse conditions. 

    For each fault type, the system calculates and presents the following **Key Performance Indicators (KPIs)**:

    - **Total Energy (MWh):** Energy output during the simulation, helping to assess whether the fault had an impact on turbine efficiency.
    - **Capacity Factor (%):** Ratio of actual energy produced versus the theoretical maximum, indicating how effectively the turbine was utilized.
    - **Full Load Hours (h):** Equivalent hours the turbine would have to operate at full capacity to generate the same energy.
    - **MTBF (s):** Mean Time Between Failures, representing the average operational time between two fault detections.

    These indicators provide operational insight into the severity and consequences of each fault.

    Your task in this section is to:

    - Review the detection results and performance metrics for each simulated fault.
    - Assess whether the diagnostic system behaves consistently with what would be expected in real wind turbine operations.
    - Identify whether the current logic and detection performance are sufficient for distinguishing between normal operation and fault conditions.

    This evaluation is critical for refining the digital twin's diagnostic capabilities before advancing to the model testing phase.
    """)

    # Crie uma instância da classe
    optimizer = TurbineOptimizer(df_labeled, None, 4.8)
    ################################
    ######## Falha 1 - Fixa ########
    ################################
    st.write("""### a) Encoder with a fixed fault""")
    st.text("Blade A with fixed value at 5° regardless of the wind intensity used.")
    b1_fixed_window = df_labeled.iloc[1950:2150]
    ref_window = df_labeled.iloc[1900:1999]
    # -- Plot 1: Wind vs Pitch 1, 2, 3
    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(x=b1_fixed_window['Wind_speed_mean'], y=b1_fixed_window['Pitch_angle_1_mean'],
                                mode='markers', name='Blade A', marker=dict(color='red', symbol='cross')))
    fig_pitch.add_trace(go.Scatter(x=b1_fixed_window['Wind_speed_mean'], y=b1_fixed_window['Pitch_angle_2_mean'],
                                mode='markers', name='Blade B', marker=dict(color='green', symbol='square')))
    fig_pitch.add_trace(go.Scatter(x=b1_fixed_window['Wind_speed_mean'], y=b1_fixed_window['Pitch_angle_3_mean'],
                                mode='markers', name='Blade C', marker=dict(color='blue')))

    fig_pitch.update_layout(title="Pitch angle as a function of wind speed",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Pitch angle (°)",
                            legend_title="Pitch Blades")

    st.plotly_chart(fig_pitch, use_container_width=True)

    # -- Plot 2: Wind vs Power
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=b1_fixed_window['Wind_speed_mean'], y=b1_fixed_window['Power_sensor_mean'],
                                mode='markers', name='Power', marker=dict(color='blue')))

    fig_power.update_layout(title="Power curve dispersion",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Power output (MW)",
                            legend_title="")

    st.plotly_chart(fig_power, use_container_width=True)

    # Metrics
    total_energy, flh, cf, mtbf = optimizer.kpi_metrics(window=b1_fixed_window)
    ref_total_energy, ref_flh, ref_cf, _ = optimizer.kpi_metrics(ref_window)

    # Cálculo das variações
    delta_energy =  ref_total_energy - total_energy
    delta_cf = ref_cf - cf
    delta_flh = ref_flh - flh

    col1, col2 = st.columns(2)
    col1.metric("Total Energy (kWh)", f"{(total_energy*1000):.2f}", f"{delta_energy*1000:.2f}", border=True)
    col2.metric("Capacity Factor", f"{cf:.2%}", f"{delta_cf:.3%}", border=True)
    col3, col4 = st.columns(2)
    col3.metric("Full Load Hours", f"{flh:.3f}", f"{delta_flh:.3f}", border=True)
    col4.metric("MTBF (s)", f"{mtbf:.1f}" if not np.isnan(mtbf) else "N/A", border=True)
    
    ################################
    ####### Falha 2 - Ganho ########
    ################################
    st.write('''### b) Encoder with a gain fault''')
    st.text("Blade B with a gain factor of 1.2 in the time period from 2300 to 2400 s.")
    b2_gain_window = df_labeled.iloc[2250:2450]
    ref_window = df_labeled.iloc[2200:2299]
    # -- Plot 1: Wind vs Pitch 1, 2, 3
    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(x=b2_gain_window['Wind_speed_mean'], y=b2_gain_window['Pitch_angle_1_mean'],
                                mode='markers', name='Blade A', marker=dict(color='green', symbol='square')))
    fig_pitch.add_trace(go.Scatter(x=b2_gain_window['Wind_speed_mean'], y=b2_gain_window['Pitch_angle_2_mean'],
                                mode='markers', name='Blade B', marker=dict(color='red', symbol='cross')))
    fig_pitch.add_trace(go.Scatter(x=b2_gain_window['Wind_speed_mean'], y=b2_gain_window['Pitch_angle_3_mean'],
                                mode='markers', name='Blade C', marker=dict(color='blue')))

    fig_pitch.update_layout(title="Pitch angle as a function of wind speed",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Pitch angle (°)",
                            legend_title="Pitch Blades")

    st.plotly_chart(fig_pitch, use_container_width=True, key="pitch2_chart_gain")
    # -- Plot 2: Wind vs Power
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=b2_gain_window['Wind_speed_mean'], y=b2_gain_window['Power_sensor_mean'],
                                mode='markers', name='Power', marker=dict(color='blue')))

    fig_power.update_layout(title="Power curve dispersion",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Power output (MW)",
                            legend_title="")

    st.plotly_chart(fig_power, use_container_width=True, key="pcurve2_chart_gain")

    # Metrics
    total_energy, flh, cf, mtbf = optimizer.kpi_metrics(window=b2_gain_window)
    ref_total_energy, ref_flh, ref_cf, _ = optimizer.kpi_metrics(ref_window)

    # Cálculo das variações
    delta_energy =  ref_total_energy - total_energy
    delta_cf = ref_cf - cf
    delta_flh = ref_flh - flh

    col1, col2 = st.columns(2)
    col1.metric("Total Energy (kWh)", f"{(total_energy*1000):.2f}", f"{delta_energy*1000:.2f}", border=True)
    col2.metric("Capacity Factor", f"{cf:.2%}", f"{delta_cf:.3%}", border=True)
    col3, col4 = st.columns(2)
    col3.metric("Full Load Hours", f"{flh:.3f}", f"{delta_flh:.3f}", border=True)
    col4.metric("MTBF (s)", f"{mtbf:.1f}" if not np.isnan(mtbf) else "N/A", border=True)

    ################################
    ############ Falha 3 ###########
    ################################
    st.write('''### c) Encoder with a trend''')
    st.text("A progressive change from 2° to 4° in blade C")
    b3_drift_window = df_labeled.iloc[2550:2750]
    ref_window = df_labeled.iloc[2500:2599]
    # -- Plot 1: Wind vs Pitch 1, 2, 3
    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(x=b3_drift_window['Wind_speed_mean'], y=b3_drift_window['Pitch_angle_1_mean'],
                                mode='markers', name='Blade A', marker=dict(color='green', symbol='square')))
    fig_pitch.add_trace(go.Scatter(x=b3_drift_window['Wind_speed_mean'], y=b3_drift_window['Pitch_angle_2_mean'],
                                mode='markers', name='Blade B', marker=dict(color='blue')))
    fig_pitch.add_trace(go.Scatter(x=b3_drift_window['Wind_speed_mean'], y=b3_drift_window['Pitch_angle_3_mean'],
                                mode='markers', name='Blade C', marker=dict(color='red', symbol='cross')))

    fig_pitch.update_layout(title="Pitch angle as a function of wind speed",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Pitch angle (°)",
                            legend_title="Pitch Blades")

    st.plotly_chart(fig_pitch, use_container_width=True, key="pitch_chart_drift")

    # -- Plot 2: Wind vs Power
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=b3_drift_window['Wind_speed_mean'], y=b3_drift_window['Power_sensor_mean'],
                                   marker=dict(color='blue'), mode='markers', name='Power'))

    fig_power.update_layout(title="Power curve dispersion",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Power output (MW)",
                            legend_title="")

    st.plotly_chart(fig_power, use_container_width=True, key="pcurve_chart_drift")

    # Metrics
    total_energy, flh, cf, mtbf = optimizer.kpi_metrics(window=b3_drift_window)
    ref_total_energy, ref_flh, ref_cf, _ = optimizer.kpi_metrics(ref_window)

    # Cálculo das variações
    delta_energy =  ref_total_energy - total_energy
    delta_cf = ref_cf - cf
    delta_flh = ref_flh - flh

    col1, col2 = st.columns(2)
    col1.metric("Total Energy (kWh)", f"{(total_energy*1000):.2f}", f"{delta_energy*1000:.2f}", border=True)
    col2.metric("Capacity Factor", f"{cf:.2%}", f"{delta_cf:.3%}", border=True)
    col3, col4 = st.columns(2)
    col3.metric("Full Load Hours", f"{flh:.3f}", f"{delta_flh:.3f}", border=True)
    col4.metric("MTBF (s)", f"{mtbf:.1f}" if not np.isnan(mtbf) else "N/A", border=True)

    ################################
    ############ Falha 4 ###########
    ################################
    st.write("### d) Rotor tachometer with a fixed fault")
    st.text("A fixed value on rotor tachometer equal to 1.4 rad/s in the time period from 1500 to 1600s.")
    tach_r_fixed_window = df_labeled.iloc[1450:1650]
    ref_window = df_labeled.iloc[1400:1499]

    fig_rotor = go.Figure()
    fig_rotor.add_trace(go.Scatter(x=tach_r_fixed_window['Wind_speed_mean'],y=tach_r_fixed_window['Rotor_speed_sensor_mean'],
                                mode='markers', name='Rotor speed', marker=dict(color='green', symbol='square')))
    fig_rotor.update_layout(title="Wind Speed vs Rotor Speed",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Rotor speed (rpm)",
                            legend_title="")

    fig_generator = go.Figure()
    fig_generator.add_trace(go.Scatter(x=tach_r_fixed_window['Wind_speed_mean'], y=tach_r_fixed_window['Generator_speed_sensor_mean'],
                                    mode='markers', name='Generator speed', marker=dict(color='blue')))
    fig_generator.update_layout(title="Wind Speed vs Generator Speed",
                                xaxis_title="Wind speed (m/s)",
                                yaxis_title="Generator speed (rpm)",
                                legend_title="")

    # -- Mostrar os dois plots lado a lado
    col_rotor, col_gen = st.columns(2)
    col_rotor.plotly_chart(fig_rotor, use_container_width=True, key="rotor_plot_tach")
    col_gen.plotly_chart(fig_generator, use_container_width=True, key="gen_plot_tach")

    # -- Plot 2: Wind vs Power
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=tach_r_fixed_window['Wind_speed_mean'], y=tach_r_fixed_window['Power_sensor_mean'],
                                mode='markers', name='Power', marker=dict(color='blue')))

    fig_power.update_layout(title="Power curve dispersion",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Power output (MW)",
                            legend_title="")

    st.plotly_chart(fig_power, use_container_width=True, key="pcurve2_chart_tach")

    # Metrics
    total_energy, flh, cf, mtbf = optimizer.kpi_metrics(window=tach_r_fixed_window)
    ref_total_energy, ref_flh, ref_cf, _ = optimizer.kpi_metrics(ref_window)

    # Cálculo das variações
    delta_energy =  ref_total_energy - total_energy
    delta_cf = ref_cf - cf
    delta_flh = ref_flh - flh

    col1, col2 = st.columns(2)
    col1.metric("Total Energy (kWh)", f"{(total_energy*1000):.2f}", f"{delta_energy*1000:.2f}", border=True)
    col2.metric("Capacity Factor", f"{cf:.2%}", f"{delta_cf:.3%}", border=True)
    col3, col4 = st.columns(2)
    col3.metric("Full Load Hours", f"{flh:.3f}", f"{delta_flh:.3f}", border=True)
    col4.metric("MTBF (s)", f"{mtbf:.1f}" if not np.isnan(mtbf) else "N/A", border=True)

    ################################
    ############ Falha 5 ###########
    ################################
    st.write("### e) Rotor and generator tachometer with gain fault")
    st.text("Rotor and generator gain fault, respectively, equal to 1.1 and 0.9 in the time period from 1000 to 1100 s")
    both_tach_fault = df_labeled.iloc[950:1150]
    ref_window = df_labeled.iloc[900:999]

    fig_rotor = go.Figure()
    fig_rotor.add_trace(go.Scatter(x=both_tach_fault['Wind_speed_mean'], y=both_tach_fault['Rotor_speed_sensor_mean'],
                                   mode='markers', name='Rotor speed',
                                   marker=dict(color='green', symbol='square')))
    fig_rotor.update_layout(title="Wind Speed vs Rotor Speed",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Rotor speed (rpm)")

    fig_generator = go.Figure()
    fig_generator.add_trace(go.Scatter(x=both_tach_fault['Wind_speed_mean'], y=both_tach_fault['Generator_speed_sensor_mean'],
                                    mode='markers', name='Generator speed', marker=dict(color='blue')))
    fig_generator.update_layout(title="Wind Speed vs Generator Speed",
                                xaxis_title="Wind speed (m/s)", yaxis_title="Generator speed (rpm)")
    # -- Mostrar os dois plots lado a lado
    col_rotor, col_gen = st.columns(2)
    col_rotor.plotly_chart(fig_rotor, use_container_width=True, key="both_plot_rot")
    col_gen.plotly_chart(fig_generator, use_container_width=True, key="both_plot_gen")
    # -- Plot 2: Wind vs Power
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=both_tach_fault['Wind_speed_mean'], y=both_tach_fault['Power_sensor_mean'],
                                mode='markers', name='Power', marker=dict(color='blue')))

    fig_power.update_layout(title="Power curve dispersion",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Power output (MW)",
                            legend_title="")

    st.plotly_chart(fig_power, use_container_width=True, key="both_chart_tach")
    # Metrics
    total_energy, flh, cf, mtbf = optimizer.kpi_metrics(window=both_tach_fault)
    ref_total_energy, ref_flh, ref_cf, _ = optimizer.kpi_metrics(ref_window)
    # Cálculo das variações
    delta_energy =  ref_total_energy - total_energy
    delta_cf = ref_cf - cf
    delta_flh = ref_flh - flh

    col1, col2 = st.columns(2)
    col1.metric("Total Energy (kWh)", f"{(total_energy*1000):.2f}", f"{delta_energy*1000:.2f}", border=True)
    col2.metric("Capacity Factor", f"{cf:.2%}", f"{delta_cf:.3%}", border=True)
    col3, col4 = st.columns(2)
    col3.metric("Full Load Hours", f"{flh:.3f}", f"{delta_flh:.3f}", border=True)
    col4.metric("MTBF (s)", f"{mtbf:.1f}" if not np.isnan(mtbf) else "N/A", border=True)

    ################################
    ############ Falha 6 ###########
    ################################
    st.write("### f) Abrupt actuator fault")
    st.text("There is a change in the dynamics due to hydraulic pressure drop of the pitch actuator 2; the fault is assumed to be abrupt and it is present in the time period from 2900 to 3000 s")
    act_abru_fault = df_labeled.iloc[2850:3050]
    ref_window = df_labeled.iloc[2800:2899]

    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(x=act_abru_fault['Wind_speed_mean'], y=act_abru_fault['Pitch_angle_1_mean'],
                                mode='markers', name='Blade A', marker=dict(color='green', symbol='square')))
    fig_pitch.add_trace(go.Scatter(x=act_abru_fault['Wind_speed_mean'], y=act_abru_fault['Pitch_angle_2_mean'],
                                mode='markers', name='Blade B', marker=dict(color='red', symbol='cross')))
    fig_pitch.add_trace(go.Scatter(x=act_abru_fault['Wind_speed_mean'], y=act_abru_fault['Pitch_angle_3_mean'],
                                mode='markers', name='Blade C', marker=dict(color='blue')))

    fig_pitch.update_layout(title="Pitch angle as a function of wind speed",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Pitch angle (°)",
                            legend_title="Pitch Blades")

    st.plotly_chart(fig_pitch, use_container_width=True, key="act_chart")

    # -- Plot 2: Wind vs Power
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=act_abru_fault['Wind_speed_mean'], y=act_abru_fault['Power_sensor_mean'],
                                mode='markers', name='Power', marker=dict(color='blue')))

    fig_power.update_layout(title="Power curve dispersion",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Power output (MW)",
                            legend_title="")

    st.plotly_chart(fig_power, use_container_width=True, key="act_chart_pcurve")

    # Metrics
    total_energy, flh, cf, mtbf = optimizer.kpi_metrics(window=act_abru_fault)
    ref_total_energy, ref_flh, ref_cf, _ = optimizer.kpi_metrics(ref_window)

    # Cálculo das variações
    delta_energy =  ref_total_energy - total_energy
    delta_cf = ref_cf - cf
    delta_flh = ref_flh - flh

    col1, col2 = st.columns(2)
    col1.metric("Total Energy (kWh)", f"{(total_energy*1000):.2f}", f"{delta_energy*1000:.2f}", border=True)
    col2.metric("Capacity Factor", f"{cf:.2%}", f"{delta_cf:.3%}", border=True)
    col3, col4 = st.columns(2)
    col3.metric("Full Load Hours", f"{flh:.3f}", f"{delta_flh:.3f}", border=True)
    col4.metric("MTBF (s)", f"{mtbf:.1f}" if not np.isnan(mtbf) else "N/A", border=True)

    ################################
    ############ Falha 7 ###########
    ################################
    st.write("### e) Slow actuator fault")
    st.text("Thre is a change in the dynamics due to increased air content in the oil on pitch actuator 3. The fault is slowly introduced during 30 s with a constant rate; afterward the fault is active during 40 s, and again decreases during 30 s. The fault begins at 3500 s and ends at 3600 s.")
    act_slow_fault = df_labeled.iloc[3450:3650]
    ref_window = df_labeled.iloc[3400:3499]

    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(x=act_slow_fault['Wind_speed_mean'], y=act_slow_fault['Pitch_angle_1_mean'],
                                mode='markers', name='Blade A', marker=dict(color='green', symbol='square')))
    fig_pitch.add_trace(go.Scatter(x=act_slow_fault['Wind_speed_mean'], y=act_slow_fault['Pitch_angle_2_mean'],
                                mode='markers', name='Blade B', marker=dict(color='blue')))
    fig_pitch.add_trace(go.Scatter(x=act_slow_fault['Wind_speed_mean'], y=act_slow_fault['Pitch_angle_3_mean'],
                                mode='markers', name='Blade C', marker=dict(color='red', symbol='cross')))
    fig_pitch.update_layout(title="Pitch angle as a function of wind speed",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Pitch angle (°)",
                            legend_title="Pitch Blades")

    st.plotly_chart(fig_pitch, use_container_width=True, key="act_slow_fault_pcurve_pitch")

    # -- Plot 2: Wind vs Power
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=act_slow_fault['Wind_speed_mean'], y=act_slow_fault['Power_sensor_mean'],
                                mode='markers', name='Power', marker=dict(color='blue')))

    fig_power.update_layout(title="Power curve dispersion",
                            xaxis_title="Wind speed (m/s)",
                            yaxis_title="Power output (MW)",
                            legend_title="")

    st.plotly_chart(fig_power, use_container_width=True, key="act_slow_fault_pcurve")

    # Metrics
    total_energy, flh, cf, mtbf = optimizer.kpi_metrics(window=act_slow_fault)
    ref_total_energy, ref_flh, ref_cf, _ = optimizer.kpi_metrics(ref_window)

    # Cálculo das variações
    delta_energy =  ref_total_energy - total_energy
    delta_cf = ref_cf - cf
    delta_flh = ref_flh - flh

    col1, col2 = st.columns(2)
    col1.metric("Total Energy (kWh)", f"{(total_energy*1000):.2f}", f"{delta_energy*1000:.2f}", border=True)
    col2.metric("Capacity Factor", f"{cf:.2%}", f"{delta_cf:.3%}", border=True)
    col3, col4 = st.columns(2)
    col3.metric("Full Load Hours", f"{flh:.3f}", f"{delta_flh:.3f}", border=True)
    col4.metric("MTBF (s)", f"{mtbf:.1f}" if not np.isnan(mtbf) else "N/A", border=True)
  
    st.write("""### User's insights and comments""")
    q02 = query(
                "To what extent do the fault scenarios modeled in this prototype represent realistic failure behaviors in wind turbines? Are the selected variables and performance indicators adequate to support future generalization and practical application of the digital twin diagnosis system? Please share your first impressions on this initial stage of the prototype and potencial to support the development of a reliable digital twin knowledge base.", key='q02')
    user_response['Q2'] = q02 # Calling query function for user feedback

    ################################# DRIVETRAIN #################################
    st.write("""# 2) Drivetrain""")
    # EDA For vibration analysis - 10 time-domain features
    st.markdown('To be released soon')

elif add_select_turbine == '4.8MW (OpenFAST)':
    st.write(f""" # {add_select_turbine[:5]} wind turbine diagnosis screen""")
    st.markdown(f"""
                This section presents a comprehensive analysis of the {add_select_turbine[:5]} wind turbine behavior using a knowledge-based digital twin (KBDT). The system integrates data-driven modeling, statistical evaluation, fault detection, classification, and fault-tolerant control.

                Initially, the operational data are explored through descriptive statistics and visual diagnostics, including wind speed variability, power curve modeling using clustering, and pitch angle consistency across the blades. These analyses support early anomaly detection and turbine performance evaluation.

                Subsequently, a multiclass classification model is introduced to identify different types of sensor faults. The model is trained on limited and imbalanced datasets, replicating real-world constraints, and evaluated using classification metrics such as precision, recall, and F1-score.

                Finally, a fault-tolerant control strategy is applied to correct faulty sensor signals. The system automatically detects the time of failure, reconstructs the signal using PID-based correction, and evaluates performance through key metrics like detection time, mean absolute error (MAE), and settling time.

                Together, these components showcase the digital twin's ability to diagnose faults, support decision-making, and maintain reliable turbine operation under faulty conditions.
               """)
    # Criando uma instância da classe
    sensor = Sensor(file_path='Testing_data/dataset_4800.csv', sampling_frequency=10)
    sensor.load_and_preprocess()
    sensor.extract_features()
    dt_test = sensor.label_data_test() # Já está com o Label
    
    # Creating the wind turbine visualization
    data_analysis = WindTurbine4800(dt_test, add_select_turbine)
    data_analysis.display_plots()

    s7 = Sensor("Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_7.csv", 10)
    s7.load_and_preprocess()
    s7.extract_features()
    df7 = s7.label_data_test() # Já está com o Label

    s1 = Sensor("Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_1.csv", 10)
    s1.load_and_preprocess()
    s1.extract_features()
    df1 = s1.label_data_test() # Já está com o Label

    s2 = Sensor("Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_2.csv", 10)
    s2.load_and_preprocess()
    s2.extract_features()
    df2 = s2.label_data_test() # Já está com o Label

    s3 = Sensor("Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_3.csv", 10)
    s3.load_and_preprocess()
    s3.extract_features()
    df3 = s3.label_data_test() # Já está com o Label

    s4 = Sensor("Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_4.csv", 10)
    s4.load_and_preprocess()
    s4.extract_features()
    df4 = s4.label_data_test() # Já está com o Label

    s5 = Sensor("Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_5.csv", 10)
    s5.load_and_preprocess()
    s5.extract_features()
    df5 = s5.label_data_test() # Já está com o Label

    s6 = Sensor("Training_data/WECS_4800MW_ORIGINAL_v3_sensitivity_6.csv", 10)
    s6.load_and_preprocess()
    s6.extract_features()
    df6 = s6.label_data_test() # Já está com o Label
    dt_train = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=0)
    dt_train = dt_train.dropna()
    st.markdown("---")
    st.write(""" # Multiclass classification""")
    st.markdown('''
    This section offers an opportunity to evaluate a machine learning model designed for fault diagnosis in wind turbines. We will utilize the training data from the previous section and apply it to a separate test dataset.

    The core idea here is to replicate a modeling scenario that accounts for the scarcity of training data, particularly concerning both healthy and faulty conditions in wind turbine fleets. You can influence this simulation directly. As you interact with the slider, the program will adjust the amount of data available for training. This specifically addresses the common real-world challenge where acquiring numerous samples of actual turbine faults can be difficult. It's also important to note that the datasets are intentionally imbalanced; this choice helps avoid the need for synthetic sample generation, allowing for a more direct assessment of the model's performance on realistic, skewed data distributions.

    Here's how to proceed:
    * Use the slider to select the desired percentage of training data to include in the model's learning process.
    * Once you've set the training data fraction, click the "Train model" button to initiate the diagnostic process.

    As a system expert, your role is to carefully analyze the model's performance. Once the training is complete, examine the Classification Report and Confusion Matrix. These tools provide detailed insights into the model's accuracy, precision, recall, and F1-score for each distinct fault type. Pay particular attention to how the model performs with varying amounts of training data, especially for the minority fault classes, as this reflects its robustness in scenarios with limited real-world fault observations.
    ''')
    # Use st.session_state para armazenar os resultados e exibi-los após a execução
    if 'results' not in st.session_state:
        st.session_state.results = None

    col1, col2 = st.columns(2, border=True, vertical_alignment="center")

    with col1:
        fraction_percentage = st.slider("Select the effective training data volume to create a model:",
                                        min_value=10.0, max_value=100.0, value=100.0, step=10.0, format="%.0f%%")
        st.info(f"{fraction_percentage:.0f}% of the data will be available for generating a model.")

    with col2:
        col_button_left, col_button_center, col_button_right = st.columns([1, 2, 1])
        with col_button_center:
            clf = SensorFaultDetector(dt_test)
            if st.button('Train model', use_container_width=True):
                with st.spinner('Running diagnosis... This may take a few seconds.'):
                    train_acc, val_acc, cr_df, test_acc, cm, dt_test_updated = clf.multiclass_classification(
                        fraction_percentage / 100, dt_train, dt_test)
                st.success('Detection completed successfully!')
                st.session_state.results = {
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'cr_df': cr_df,
                    'test_acc': test_acc,
                    'cm': cm,
                    'dt_test': dt_test_updated}

    if st.session_state.results is not None:
        st.write('## Classification report')
        st.markdown(''' This Classification Report presents essential metrics for evaluating model performance across different fault classes. Precision indicates the accuracy of positive predictions for each class (how many identified faults were truly faults), while Recall measures the model's ability to find all positive samples (how many actual faults were found). The F1-score provides a balanced measure of both precision and recall. Accuracy reflects the overall proportion of correct predictions. Macro avg calculates the unweighted average of precision, recall, and F1-score across all classes, treating each class equally. Conversely, weighted avg considers the number of instances per class, offering a more representative average for imbalanced datasets. ''')
        class_report_df = pd.DataFrame(st.session_state.results['cr_df'])
        class_report_df = class_report_df.drop(class_report_df.index[3])
        st.dataframe(class_report_df)
        
        st.write('### Training and testing accuracies')
        st.markdown('''The key here is to assess the model's learning capability and generalization performance.
                            To do this, a mathematical model is trained using the training dataset and then evaluated with a small validation set. Ideally, the validation accuracy should be equal to or higher than the training accuracy to ensure good generalization.
                            Depending on the selected data volume, this value fluctuates and also impacts the test accuracy on the real dataset.''')
        st.write(f"Training accuracy: {st.session_state.results['train_acc']:.4f}")
        st.write(f"Validation accuracy: {st.session_state.results['val_acc']:.4f}")
        st.write(f"Test accuracy: {st.session_state.results['test_acc']:.4f}")

        st.write('### Confusion Matrix')
        st.markdown('''
        The Confusion Matrix is another common metric used for fault classification. Since the matrix is normalized by row, each cell represents a proportion. The values in each row sum up to 1.0, showing how samples of a true class were classified.

        The main diagonal cells indicate the proportion of samples that were correctly classified for each true class. For instance, the value in the cell where 'True: Fault-free' intersects 'Predicted: Fault-free' shows the proportion of actual fault-free cases that the model correctly identified as such. This metric is also known as Recall (or True Positive Rate) for each specific fault type. High values along the diagonal are desirable, signifying effective fault identification.

        The off-diagonal cell represents a misclassification. This is grouped into false positive and false negative, which is precisely what one aims to mitigate. Such error can mask real fault or lead to incorrect diagnosis, significantly hindering turbine performance and potentially causing operational issue.
        ''')
        detector = SensorFaultDetector(dt_test)
        multi_data_analysis = WindTurbine4800(dt_test, add_select_turbine=None)
        fig_multi_mlas = multi_data_analysis.plot_multiclass_cm(st.session_state.results['cm'],
                                                                 'Multiclass confusion matrix normalized by row')
        st.plotly_chart(fig_multi_mlas, use_container_width=True)
        
        st.write('## Fault tolerant control')
        st.markdown("""
        This section demonstrates the ability of the digital twin to detect, isolate, and mitigate sensor faults in wind turbine systems through a fault-tolerant control strategy. After the classification and diagnosis stages, the system identifies sensor anomalies and applies a corrective control approach to restore the affected signals using PID-based reconstruction.

        For each fault scenario, the following metrics are presented to evaluate the correction performance:
        * Time detection: Interval between the fault injection and the moment the anomaly was detected.
        * Mean Absolute Error (MAE) Quantifies the deviation between the reconstructed signal and the reference signal.
        * Settling time Time required for the corrected signal to stabilize within a defined tolerance band around the reference.

        Below, each sensor fault is illustrated with its corresponding fault window and detection point, followed by the application of the correction mechanism. These visualizations reflect the ability of the control logic to respond promptly and accurately to various fault conditions.
        """)
        # Exibir gráfico FTC
        services = TurbineOptimizer(st.session_state.results['dt_test'],
                                    n_clusters=None, Pnom=None)
        ######################
        # Control of fault 1 #
        ######################
        fault = st.session_state.results['dt_test'].iloc[1950:2050]
        fault_injected_idx = 2000
        fault_start_idx = 2003
        initial_fault_value = fault.loc[fault_start_idx, 'Pitch_angle_1_mean']
        ref_signal = fault.loc[fault_start_idx:, 'Pitch_angle_2_mean'].values
        corrected_signal = services.pid_correction(initial_fault_value, ref_signal)
        fig_ftc = services.ftc_plots(fault_injected_idx, fault_start_idx,
            fault_signal = fault['Pitch_angle_1_mean'], corrected_signal = corrected_signal,
            figure_title = "Encoder A with a fixed fault")
        st.plotly_chart(fig_ftc, use_container_width=True)

        col1, col2, col3 = st.columns(3, border = True)
        col1.metric("Time detection", f"{fault_start_idx - fault_injected_idx}s")
        mae = np.mean(np.abs(np.array(corrected_signal) - ref_signal[:len(corrected_signal)]))
        col2.metric("Mean absolute error", f"{mae:.4f}")
        diff = np.array(corrected_signal) - ref_signal[:len(corrected_signal)]
        settling_time_tol = 1 # Difference for settling time
        settling_time = next((i for i in range(len(diff)) if all(abs(d) < settling_time_tol for d in diff[i:])), None)
        col3.metric("Settling time", f"{settling_time}s" if settling_time is not None else "Not settled")

        ######################
        # Control of fault 2 #
        ######################
        fault = st.session_state.results['dt_test'].iloc[2250:2350]
        fault_injected_idx = 2300
        fault_start_idx = 2310
        initial_fault_value = fault.loc[fault_start_idx, 'Pitch_angle_2_mean']
        ref_signal = fault.loc[fault_start_idx:, 'Pitch_angle_3_mean'].values
        corrected_signal = services.pid_correction(initial_fault_value, ref_signal)
        fig_ftc = services.ftc_plots(fault_injected_idx, fault_start_idx,
            fault_signal = fault['Pitch_angle_2_mean'], corrected_signal = corrected_signal,
            figure_title = "Encoder B with a gain fault")
        st.plotly_chart(fig_ftc, use_container_width=True)

        col1, col2, col3 = st.columns(3, border = True)
        col1.metric("Time detection", f"{fault_start_idx - fault_injected_idx}s")
        mae = np.mean(np.abs(np.array(corrected_signal) - ref_signal[:len(corrected_signal)]))
        col2.metric("Mean absolute error", f"{mae:.4f}")
        diff = np.array(corrected_signal) - ref_signal[:len(corrected_signal)]
        settling_time = next((i for i in range(len(diff)) if all(abs(d) < settling_time_tol for d in diff[i:])), None)
        col3.metric("Settling time", f"{settling_time}s" if settling_time is not None else "Not settled")
        
        ######################
        # Control of fault 3 #
        ######################
        fault = st.session_state.results['dt_test'].iloc[2550:2650]
        fault_injected_idx = 2600
        fault_start_idx = 2608
        initial_fault_value = fault.loc[fault_start_idx, 'Pitch_angle_3_mean']
        ref_signal = fault.loc[fault_start_idx:, 'Pitch_angle_1_mean'].values
        corrected_signal = services.pid_correction(initial_fault_value, ref_signal)
        fig_ftc = services.ftc_plots(fault_injected_idx, fault_start_idx,
            fault_signal = fault['Pitch_angle_3_mean'], corrected_signal = corrected_signal,
            figure_title = "Encoder C with a trend")
        st.plotly_chart(fig_ftc, use_container_width=True)

        col1, col2, col3 = st.columns(3, border = True)
        col1.metric("Time detection", f"{fault_start_idx - fault_injected_idx}s")
        mae = np.mean(np.abs(np.array(corrected_signal) - ref_signal[:len(corrected_signal)]))
        col2.metric("Mean absolute error", f"{mae:.4f}")
        diff = np.array(corrected_signal) - ref_signal[:len(corrected_signal)]
        settling_time = next((i for i in range(len(diff)) if all(abs(d) < settling_time_tol for d in diff[i:])), None)
        col3.metric("Settling time", f"{settling_time}s" if settling_time is not None else "Not settled")

        ######################
        # Control of fault 4 #
        ######################
        fault = st.session_state.results['dt_test'].iloc[1450:1550] # Fault window
        fault_injected_idx = 1500
        fault_start_idx = 1506
        initial_fault_value = fault.loc[fault_start_idx, 'Rotor_speed_sensor_mean'] # Value of the current fault
        ref_signal = fault.loc[fault_start_idx:, 'Generator_speed_sensor_mean'].values / 95 # Values of the reference signal
        corrected_signal = services.pid_correction(initial_fault_value, ref_signal)
        fig_ftc = services.ftc_plots(fault_injected_idx, fault_start_idx,
            fault_signal = fault['Rotor_speed_sensor_mean'], corrected_signal = corrected_signal,
            figure_title = "Rotor and generator tachometers with a fixed fault - REAL")
        st.plotly_chart(fig_ftc, use_container_width=True)

        col1, col2, col3 = st.columns(3, border = True)
        col1.metric("Time detection", f"{fault_start_idx - fault_injected_idx}s")
        mae = np.mean(np.abs(np.array(corrected_signal) - ref_signal[:len(corrected_signal)]))
        col2.metric("Mean absolute error", f"{mae:.4f}")
        diff = np.array(corrected_signal) - ref_signal[:len(corrected_signal)]
        settling_time_tol = 0.5 # Difference for settling time
        settling_time = next((i for i in range(len(diff)) if all(abs(d) < settling_time_tol for d in diff[i:])), None)
        col3.metric("Settling time", f"{settling_time}s" if settling_time is not None else "Not settled")

        ######################
        # Control of fault 5 #
        ######################
        fault = st.session_state.results['dt_test'].iloc[950:1050]
        fault_injected_idx = 1000
        fault_start_idx = 1005
        initial_fault_value = fault.loc[fault_start_idx, 'Rotor_speed_sensor_mean']
        ref_signal = fault.loc[fault_start_idx:, 'Generator_speed_sensor_mean'].values / 95
        corrected_signal = services.pid_correction(initial_fault_value, ref_signal)
        fig_ftc = services.ftc_plots(fault_injected_idx, fault_start_idx,
            fault_signal = fault['Rotor_speed_sensor_mean'], corrected_signal = corrected_signal,
            figure_title = "Rotor and generator tachometer with gain fault")
        st.plotly_chart(fig_ftc, use_container_width=True)

        col1, col2, col3 = st.columns(3, border = True)
        col1.metric("Time detection", f"{fault_start_idx - fault_injected_idx}s")
        mae = np.mean(np.abs(np.array(corrected_signal) - ref_signal[:len(corrected_signal)]))
        col2.metric("Mean absolute error", f"{mae:.4f}")
        diff = np.array(corrected_signal) - ref_signal[:len(corrected_signal)]
        settling_time = next((i for i in range(len(diff)) if all(abs(d) < settling_time_tol for d in diff[i:])), None)
        col3.metric("Settling time", f"{settling_time}s" if settling_time is not None else "Not settled")
    
    st.write("""### User's insights and comments""")
    q03 = query(
    """Does the system’s classification performance, as seen in the confusion matrices, KPIs, power curve, etc.., provide an accurate and operationally useful representation of fault and healthy conditions? In your experience, do the input variables and inference offer sufficient discriminatory power for real diagnostics? Would you recommend any improvements in the analysis above?""",
    key='q03' )
    user_response['Q3'] = q03
    
elif add_select_turbine == '2.0MW (Kelmarsh)':

    data_link = 'Testing_data/dataset_2100_Kelmarsh.csv'
    pc = PitchControllerComponent(data_link, 10)
    pc.load_and_preprocess() # Carrega e limpa os dados
    pc.extract_features() # Extrai as estatísticas dos dados

    df = pc.get_data() # Obtendo os dados processados

    # Creating the wind turbine visualization
    data_analysis = WindTurbine2000(df, add_select_turbine)
    data_analysis.display_plots()

    # Storing the outliers to analysis
    dv = data_analysis.generate_dataframe()
    st.write('''###### The digital twin was uploaded successfully!''')

    # Proceed to fault analysis
    detector = SensorInferenceSystem(df, 0.01, 3, 119.6, dv)

else:
    st.text('To be released soon')

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
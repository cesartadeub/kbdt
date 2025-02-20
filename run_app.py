print('\n-----------------------')
import numpy as np
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# ==============================================
# === Functions to add faults to the dataset ===
# ==============================================
def pitch_fixed(df, duration):
    fault_start = 1000 # np.random.randint(0, len(df) - duration) # Random start
    fixed_value = 15
    df.loc[fault_start:fault_start+duration, "Pitch_angle_1_mean"] = fixed_value
    return df

def pitch_offset(df, duration):
    fault_start = np.random.randint(0, len(df) - duration)
    df.loc[fault_start:fault_start+duration, "Pitch_angle_1_mean"] += 5 # Adiciona um offset fixo
    return df

def pitch_gain(df, duration):
    fault_start = np.random.randint(0, len(df) - duration)
    df.loc[fault_start:fault_start+duration, "Pitch_angle_1_mean"] *= 1.2 # Aumenta o ganho em 20%
    return df

def omega_fixed(df, duration):
    fault_start = np.random.randint(0, len(df) - duration)
    df.loc[fault_start:fault_start+duration, "Rotor_speed_sensor_mean"] = 50 # Define um valor fixo no RPM
    return df

def omega_offset(df, duration):
    fault_start = np.random.randint(0, len(df) - duration)
    df.loc[fault_start:fault_start+duration, "Rotor_speed_sensor_mean"] += 10 # Offset no RPM
    return df

# Fault selector to insert it in the dataframe of the plotting
def fault_selector(add_select_fault, df, duration):
    if add_select_fault == 'Encoder with a fixed value':
        df = pitch_fixed(df, duration)
    elif add_select_fault == 'Encoder with an offset':
        df = pitch_offset(df, duration)
    elif add_select_fault == 'Encoder with gain':
        df = pitch_gain(df, duration)
    elif add_select_fault == 'Tachometer with a fixed value':
        df = omega_fixed(df, duration)
    elif add_select_fault == 'Tachometer with an offset':
        df = omega_offset(df, duration)
    else:
        pass
    ## Add as many faults as you want
    return df

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
    'Select a fault',
    ('Encoder with a fixed value', 'Encoder with an offset', 'Encoder with gain',
     'Tachometer with a fixed value', 'Tachometer with an offset'), placeholder='Chose an option'
)

add_duration = st.sidebar.slider("Fault duration (s):",
                                     min_value=0, max_value=100, value=50, step=1)

add_run = st.sidebar.button('Run analysis')

# ==============================================
# ================= Main window ================
# ==============================================
st.title("""
         Wind turbine knowledge-based digital twin sensor faults diagnosis
         *_Cesar Tadeu NM Branco v0.25.02.3_*
         """)

if add_select_turbine == '4.8MW':
    df = load_a_file(file_path = 'https://raw.githubusercontent.com/cesartadeub/kbdt/refs/heads/main/turbine_dataset/dataset_4800.csv')
    # Fault selector acts below
    df = fault_selector(add_select_fault, df, add_duration)
    
    # Plotting power curve
    WsxP = px.scatter(df, x="Wind_speed_mean", y="Power_sensor_mean",
    labels = {"Wind speed (m/s)": "Power generated (W)"})
    WsxP.update_traces(hovertemplate='<i>Ws: </i>%{x:.1f} m/s'+
                        '<br><b>P</b>: %{y:.1f}<br>W')
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
    pit.update_traces(hovertemplate="Angle: %{y:.2f} deg")
    pit.update_layout(showlegend=False)
    st.plotly_chart(pit, on_select="rerun", use_container_width=True, color = [77, 183, 211])

    # Plotting turbine speed
    omega = make_subplots(rows=1, cols=2,
                          subplot_titles=("Rotor speed (rpm)", "Generator speed (rpm)"))
    om1 = px.scatter(df, y="Rotor_speed_sensor_mean")
    om2 = px.scatter(df, y="Generator_speed_sensor_mean")
    omega.add_trace(om1.data[0], row=1, col=1)
    omega.add_trace(om2.data[0], row=1, col=2)
    omega.update_traces(hovertemplate="Speed: %{y:.2f} rpm")
    omega.update_layout(showlegend=False)
    st.plotly_chart(omega, on_select="rerun", use_container_width=True, color = [77, 183, 211])
else:
    st.text('To be released')

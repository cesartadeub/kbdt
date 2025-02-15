print('\n-----------------------')
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests

from SensorData import SensorData as sd

file_path = 'https://raw.githubusercontent.com/cesartadeub/kbdt/refs/heads/main/set1.csv'
sampling_frequency = 10  # Example frequency

sd = sd(file_path, sampling_frequency)  # Instantiate the class
df = sd.feature_extraction(sampling_frequency)  # Call the method without arguments

time = np.linspace(0, len(df)-1, len(df))  
df["Time"] = time  # Adicionar ao DataFrame

# Interface do Streamlit
st.title("""
Knowledge-based digital twin wind turbine sensor faults diagnosis
_*Cesar Tadeu NM Branco* v0.25.02.3_
""")
# Version index organization:
# 1) 0 only for me, 1 for first approach with specialist 
# 2) year; 3) month; 4) month week 

# Add a selectbox to the sidebar:
# Sidebar button action
if st.sidebar.button("User's guide"):
    url = "https://raw.githubusercontent.com/cesartadeub/kbdt/refs/heads/main/README.md"
    response = requests.get(url)
    if response.status_code == 200:
        st.markdown(response.text, unsafe_allow_html=True)
    else:
        st.error("Failed to load README file.")

add_sidebar_title = st.sidebar.text("PARAMETERS")
add_selectbox = st.sidebar.selectbox(
    'Select a turbine',
    ('2MW', '4.8MW', '10MW')
)
# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

left_column, right_column = st.columns(2)

with left_column: # Original turbine
    st.write('Original turbine')
    st.line_chart(df, x = 'Time', y = 'Wind_speed_mean', color = [77, 183, 211])
    st.line_chart(df, x = 'Time', y = 'Power_sensor_mean', color = [77, 183, 211])
    st.line_chart(df, x = 'Time', y = 'Pitch_angle_1_mean', color = [77, 183, 211])
    st.line_chart(df, x = 'Time', y = 'Pitch_angle_2_mean', color = [77, 183, 211])
    st.line_chart(df, x = 'Time', y = 'Pitch_angle_3_mean', color = [77, 183, 211])
    st.line_chart(df, x = 'Time', y = 'Rotor_speed_sensor_mean', color = [77, 183, 211])
    st.line_chart(df, x = 'Time', y = 'Generator_speed_sensor_mean', color = [77, 183, 211])

with right_column: # Digital twin
    st.write('Virtual turbine')
    st.line_chart(df, x = 'Time', y = 'Wind_speed_mean', color = [255, 117, 0])
    st.line_chart(df, x = 'Time', y = 'Power_sensor_mean', color = [255, 117, 0])
    st.line_chart(df, x = 'Time', y = 'Pitch_angle_1_mean', color = [255, 117, 0])
    st.line_chart(df, x = 'Time', y = 'Pitch_angle_2_mean', color = [255, 117, 0])
    st.line_chart(df, x = 'Time', y = 'Pitch_angle_3_mean', color = [255, 117, 0])
    st.line_chart(df, x = 'Time', y = 'Rotor_speed_sensor_mean', color = [255, 117, 0])
    st.line_chart(df, x = 'Time', y = 'Generator_speed_sensor_mean', color = [255, 117, 0])

st.text('Turbine status')
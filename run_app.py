print('\n-----------------------')
import numpy as np
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from SensorData import SensorData as sd


file_path = 'https://raw.githubusercontent.com/cesartadeub/kbdt/refs/heads/main/set1.csv'
sampling_frequency = 10  # Example frequency

sd = sd(file_path, sampling_frequency) # Instantiate the class
df = sd.feature_extraction(sampling_frequency) # Call the method without arguments

time = np.linspace(0, len(df)-1, len(df))  
df["Time"] = time # Adicionar ao DataFrame

# Interface do Streamlit
st.title("""
         Wind turbine knowledge-based digital twin sensor faults diagnosis
         *_Cesar Tadeu NM Branco v0.25.02.3_*
         """)
# Version index organization:
# 1) 0 only for me, 1 for first approach with specialist 
# 2) year; 3) month; 4) month week 

# =============== Sidebar button action ===============
# User's document download:
url = "https://raw.githubusercontent.com/cesartadeub/kbdt/main/users_guide.pdf"
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
                                     # ''')
# Add a select box:
add_select_turbine = st.sidebar.selectbox(
    'Select a turbine',
    ('2MW', '4.8MW', '10MW'), disabled = False
)
st.sidebar.write('A',add_select_turbine,' wind turbine dataset selected')

add_select_fault = st.sidebar.selectbox(
    'Select a fault',
    ('Encoder fixed', 'Encoder offset', 'Encoder gain',
     'Tachometer fixed', 'Tachometer offset')
)

add_select_range = st.sidebar.slider("Fault duration (s):", 0, 10, 5)
st.sidebar.write(add_select_range,'s of fault duration')

add_run = st.sidebar.button('Run analysis')

# =============== Main window ===============
# Plotting power curve
WsxP = px.scatter(df, x="Wind_speed_mean", y="Power_sensor_mean",
labels = {"Wind speed (m/s)": "Power generated (W)"})
WsxP.update_traces(hovertemplate='<i>Ws: </i>%{x:.1f} m/s'+
                    '<br><b>P</b>: %{y:.1f}<br>W')
st.plotly_chart(WsxP, on_select="rerun", color = [77, 183, 211])

# Plotting blade angles
pit = make_subplots(rows=1, cols=3,
                    subplot_titles=("Blade A (deg)", "Blade B (deg)", "Blade C (deg)"))
pit1 = px.scatter(df, x="Time", y="Pitch_angle_1_mean")
pit2 = px.scatter(df, x="Time", y="Pitch_angle_2_mean")
pit3 = px.scatter(df, x="Time", y="Pitch_angle_3_mean")
pit.add_trace(pit1.data[0], row=1, col=1)
pit.add_trace(pit2.data[0], row=1, col=2)
pit.add_trace(pit3.data[0], row=1, col=3)
pit.update_traces(hovertemplate="Angle: %{y:.2f} deg")
pit.update_layout(showlegend=False)
st.plotly_chart(pit, on_select="rerun", use_container_width=True, color = [77, 183, 211])

# Plotting turbine speed
omega = make_subplots(rows=1, cols=2,
                    subplot_titles=("Rotor speed (rpm)", "Generator speed (rpm)"))
om1 = px.scatter(df, x="Time", y="Rotor_speed_sensor_mean")
om2 = px.scatter(df, x="Time", y="Generator_speed_sensor_mean")
omega.add_trace(om1.data[0], row=1, col=1)
omega.add_trace(om2.data[0], row=1, col=2)
omega.update_traces(hovertemplate="Speed: %{y:.2f} rpm")
omega.update_layout(showlegend=False)
st.plotly_chart(omega, on_select="rerun", use_container_width=True, color = [77, 183, 211])

# st.line_chart(df, x = 'Time', y = 'Wind_speed_mean', color = [77, 183, 211])
# st.line_chart(df, x = 'Time', y = 'Power_sensor_mean', color = [77, 183, 211])
# st.line_chart(df, x = 'Time', y = 'Pitch_angle_1_mean', color = [77, 183, 211])
# st.line_chart(df, x = 'Time', y = 'Pitch_angle_2_mean', color = [77, 183, 211])
# st.line_chart(df, x = 'Time', y = 'Pitch_angle_3_mean', color = [77, 183, 211])
# st.line_chart(df, x = 'Time', y = 'Rotor_speed_sensor_mean', color = [77, 183, 211])
# st.line_chart(df, x = 'Time', y = 'Generator_speed_sensor_mean', color = [77, 183, 211])

# with right_column: # Digital twin
st.write('Virtual turbine')
st.line_chart(df, x = 'Time', y = 'Wind_speed_mean', x_label = 'Time', y_label = 'Wind speed', color = [255, 117, 0])
# st.line_chart(df, x = 'Time', y = 'Power_sensor_mean', color = [255, 117, 0])
# st.line_chart(df, x = 'Time', y = 'Pitch_angle_1_mean', color = [255, 117, 0])
# st.line_chart(df, x = 'Time', y = 'Pitch_angle_2_mean', color = [255, 117, 0])
# st.line_chart(df, x = 'Time', y = 'Pitch_angle_3_mean', color = [255, 117, 0])
# st.line_chart(df, x = 'Time', y = 'Rotor_speed_sensor_mean', color = [255, 117, 0])
# st.line_chart(df, x = 'Time', y = 'Generator_speed_sensor_mean', color = [255, 117, 0])]

st.text('Turbine status')
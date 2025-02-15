import pandas as pd
import numpy as np

class SensorData:
    '''Code for sensor fault detection and tolerant control.'''
    def __init__(self, file_path, sampling_frequency):
        self.file_path = file_path
        self.sampling_frequency = sampling_frequency

    def load_and_process(self):
        df = pd.read_csv(self.file_path, skiprows=5, delimiter=';')
        df.columns = ['Time', 'Wind_speed', 'Power_sensor',
                      'Pitch_angle_1', 'Pitch_angle_2', 'Pitch_angle_3',
                      'Rotor_speed_sensor', 'Generator_speed_sensor', 'Torque_sensor', 'Unnamed']
        df.drop(columns=['Unnamed'], inplace=True)
        df = df.set_index('Time')
        return df

    def feature_extraction(self, sampling_frequency):
        df = self.load_and_process() # Carregar e processar o CSV
        group = df.groupby(np.arange(len(df)) // sampling_frequency) # Agrupar e calcular mean e std
        df = group.agg(['mean', 'max','min', 'std'])
        df.columns = ['{}_{}'.format(col, stat) for col, stat in df.columns]
        return df
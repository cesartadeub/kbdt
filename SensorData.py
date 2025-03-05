import pandas as pd
import numpy as np

class SensorDataProcessor:
    '''Sensor class for feature extraction and data labelling.'''
    def __init__(self, file_path, sampling_frequency):
        self.file_path = file_path
        self.sampling_frequency = sampling_frequency
        self.df = None

    def load_and_process(self):
        '''Carrega e processa o arquivo CSV.'''
        self.df = pd.read_csv(self.file_path, skiprows=5, delimiter=';')
        self.df.columns = ['Time', 'Wind_speed', 'Power_sensor',
                           'Pitch_angle_1', 'Pitch_angle_2', 'Pitch_angle_3',
                           'Rotor_speed_sensor', 'Generator_speed_sensor', 'Torque_sensor', 'Unnamed']
        self.df.drop(columns=['Unnamed'], inplace=True)
        self.df = self.df.set_index('Time')

    def feature_extraction(self):
        '''Extrai as características de média, máximo, mínimo e desvio padrão.'''
        if self.df is None:
            raise ValueError("Data not loaded. Please call 'load_and_process' first.") 
        group = self.df.groupby(np.arange(len(self.df)) // self.sampling_frequency)  # Agrupar e calcular mean e std
        self.df = group.agg(['mean', 'max', 'min', 'std'])
        self.df.columns = ['{}_{}'.format(col, stat) for col, stat in self.df.columns]
    
    def labelling(self):
        '''Rotula os dados com base em intervalos de índice.'''
        if self.df is None:
            raise ValueError("Data not loaded or processed. Please call 'load_and_process' and 'feature_extraction' first.")
        
        self.df['Label'] = 0  # Criando uma nova coluna e colocando um rótulo
        # Set labels for specific ranges of indices
        self.df.loc[2000:2100, 'Label'] = 1  # Fixo
        self.df.loc[2300:2400, 'Label'] = 2  # Ganho
        self.df.loc[2600:2700, 'Label'] = 3  # Drift
        self.df.loc[1500:1600, 'Label'] = 4  # Fixo
        self.df.loc[400:500, 'Label'] = 5  # Ganho
    
    def get_data(self):
        '''Retorna o DataFrame processado.'''
        if self.df is None:
            raise ValueError("Data not processed. Please call 'load_and_process', 'feature_extraction', and 'labelling' first.")
        return self.df
    
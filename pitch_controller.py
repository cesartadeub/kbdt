import pandas as pd
import numpy as np

class PitchControllerComponent:
    '''Class that represents the attributes and methods of a pitch controller.'''
    def __init__(self, file_path, sampling_frequency):
        self.file_path = file_path
        self.sampling_frequency = sampling_frequency
        self.df = None

    def load_and_preprocess(self):
        '''Carrega e processa o arquivo CSV.'''
        try: # Tentativa para o dataset Kelmarsh
            self.df = pd.read_csv(self.file_path, encoding='utf-8')
            self.df['Time'] = pd.to_datetime(self.df['Time'])
            self.df = self.df.set_index('Time')
            self.df = self.df.loc[(0 < self.df['Power_sensor']) & (self.df['Power_sensor'] <= 2050)]
            self.df = self.df.loc[(2 < self.df['Wind_speed']) & (self.df['Wind_speed'] <= 20)]
            self.df = self.df.loc[(0 <= self.df['Pitch_angle_1']) & 
                                  (0 <= self.df['Pitch_angle_2']) &
                                  (0 <= self.df['Pitch_angle_3'])]
            self.df['Power_sensor'] = self.df['Power_sensor'] / 1000
            # Passo e velocidade angular ?
        except:    
            self.df = pd.read_csv(self.file_path, skiprows=5, delimiter=';')
            self.df.columns = ['Time', 'Wind_speed', 'Power_sensor',
                            'Pitch_angle_1', 'Pitch_angle_2', 'Pitch_angle_3',
                            'Rotor_speed_sensor', 'Generator_speed_sensor', 'Torque_sensor', 'Unnamed']
            self.df.drop(columns=['Unnamed'], inplace=True)
            self.df = self.df.set_index('Time')

    def extract_features(self):
        '''Extrai as características de média, máximo, mínimo e desvio padrão.'''
        if self.df is None:
            raise ValueError("Data not loaded. Please call 'load_and_preprocess' first.") 
        group = self.df.groupby(np.arange(len(self.df)) // self.sampling_frequency)  # Agrupar e calcular mean e std
        self.df = group.agg(['mean', 'max', 'min', 'std'])
        self.df.columns = ['{}_{}'.format(col, stat) for col, stat in self.df.columns]
    
    def get_data(self):
        '''Retorna o DataFrame processado.'''
        if self.df is None:
            raise ValueError("Data not processed. Please call 'to_load_and_preprocess', 'to_extract_features', and 'to_labell' first.")
        return self.df.dropna(axis=0)

class Sensor(PitchControllerComponent):
    def __init__(self, file_path, sampling_frequency):
        '''Inicializa a classe Sensor chamando a classe pai.'''
        super().__init__(file_path, sampling_frequency) # Chama o construtor da classe pai

    def label_data(self):
        '''Rotula os dados com base em intervalos de índice.'''
        if self.df is None:
            raise ValueError("Data not loaded or processed. Please call 'to_load_and_preprocess' and 'to_extract_features' first.")
        
        self.df['Label'] = 0  # Criando uma nova coluna e colocando um rótulo
        self.df.loc[2000:2100, 'Label'] = 1  # Fixo
        self.df.loc[2300:2400, 'Label'] = 2  # Ganho
        self.df.loc[2600:2700, 'Label'] = 3  # Drift
        self.df.loc[1500:1600, 'Label'] = 4  # Fixo
        self.df.loc[400:500, 'Label'] = 5  # Ganho
       
class Actuator(PitchControllerComponent):
    def __init__(self, file_path, sampling_frequency):
        '''Inicializa a classe Actuator chamando a classe pai.'''
        super().__init__(file_path, sampling_frequency) # Chama o construtor da classe pai
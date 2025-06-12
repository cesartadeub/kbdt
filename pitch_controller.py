import pandas as pd
import numpy as np

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
        try:
            self.df = pd.read_csv(self.file_path, encoding='utf-8')
            self.df['Time'] = pd.to_datetime(self.df['Time'])
            self.df = self.df.set_index('Time')
            self.df = self.df.loc[(0 < self.df['Power_sensor']) & (self.df['Power_sensor'] <= 2050)]
            self.df = self.df.loc[(2 < self.df['Wind_speed']) & (self.df['Wind_speed'] <= 20)]
            self.df = self.df.loc[(0 <= self.df['Pitch_angle_1']) & 
                                  (0 <= self.df['Pitch_angle_2']) & 
                                  (0 <= self.df['Pitch_angle_3'])]
            self.df['Power_sensor'] = self.df['Power_sensor'] / 1000
        except:
            self.df = pd.read_csv(self.file_path, skiprows=5, delimiter=';')
            self.df.columns = ['Time', 'Wind_speed', 'Power_sensor',
                            'Pitch_angle_1', 'Pitch_angle_2', 'Pitch_angle_3',
                            'Rotor_speed_sensor', 'Generator_speed_sensor', 'Torque_sensor', 'Unnamed']
            self.df.drop(columns=['Unnamed'], inplace=True)
            self.df = self.df.set_index('Time')

    def extract_features(self):
        '''Extrai as características de média, máximo, mínimo e desvio padrão.'''
        group = self.df.groupby(np.arange(len(self.df)) // self.sampling_frequency)
        self.df = group.agg(['mean', 'max', 'min', 'std'])
        self.df.columns = ['{}_{}'.format(col, stat) for col, stat in self.df.columns]
        self.df = self.df.dropna(axis=0)
        return self.df

class Sensor(PitchControllerComponent):
    def __init__(self, file_path, sampling_frequency):
        super().__init__(file_path, sampling_frequency)

    def label_data_train(self):
        self.df['Label'] = 0
        self.df.loc[2000:2100, 'Label'] = 1
        self.df.loc[2300:2400, 'Label'] = 2
        self.df.loc[2600:2700, 'Label'] = 3
        self.df.loc[1500:1600, 'Label'] = 4
        self.df.loc[1000:1100, 'Label'] = 5
        if self.df["Wind_speed_mean"].mean() > 11:
            self.df.loc[2900:3000, 'Label'] = 6
            self.df.loc[3500:3600, 'Label'] = 7
        return self.df

    def label_data_test(self):
        self.df['Label'] = 0
        self.df.loc[2000:2100, 'Label'] = 1
        self.df.loc[2300:2400, 'Label'] = 2
        self.df.loc[2600:2700, 'Label'] = 3
        self.df.loc[1500:1600, 'Label'] = 4
        self.df.loc[1000:1100, 'Label'] = 5
        self.df.loc[2900:3000, 'Label'] = 6
        self.df.loc[3500:3600, 'Label'] = 7
        return self.df

class Actuator(PitchControllerComponent):
    def __init__(self, file_path, sampling_frequency):
        super().__init__(file_path, sampling_frequency)
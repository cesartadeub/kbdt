from sklearn.metrics import confusion_matrix, precision_score, recall_score
import pandas as pd
import random

class SensorInferenceSystem:
    def __init__(self, df, tolerance, consecutive_points, Ng, dv):
        #Precisa entrar o dataset do virtual aqui!!
        self.df = df.copy()
        self.tolerance = tolerance
        self.consecutive_points = consecutive_points
        self.Ng = Ng
        self.affected_sensor = None
        self.failure_type = None
        self.failure_index = None
        self.failure_duration = 0
        self.df["y_pred"] = 0
        self.dv = dv.copy() # Dataset do gêmeo digital - Optimized curve power
        
        self.sensor_map = {
            'AE': ('Pitch_angle_1_mean', 'Pitch_angle_1_std'), # Blade A Encoder
            'BE': ('Pitch_angle_2_mean', 'Pitch_angle_2_std'), # Blade B Encoder
            'CE': ('Pitch_angle_3_mean', 'Pitch_angle_3_std'), # Blade C Encoder
            'RT': ('Rotor_speed_sensor_mean', 'Rotor_speed_sensor_std'), # Rotor Tachometer
            'GT': ('Generator_speed_sensor_mean', 'Generator_speed_sensor_std') # Generator Tachometer
        }
        self.decode = { # Decoding dictionary. Used in the describe_anomaly. - Tentar incorporar no sensor_map mais tarde
            'AE': 'Blade A Encoder',
            'BE': 'Blade B Encoder',
            'CE': 'Blade C Encoder',
            'RT': 'Rotor Tachometer',
            'GT': 'Generator Tachometer'
        }
    
    def detect_anomalies(self):
        '''Method to detect anomalies on encoders and tachometers.
        Input: dataframe
        Output: Bool, affected_sensor, failure_index, failure_duration'''
        # Anomaly detection on encoders
        # a) Encoders
        diff_1_2 = self.df['Pitch_angle_1_mean'] - self.df['Pitch_angle_2_mean']
        diff_1_3 = self.df['Pitch_angle_1_mean'] - self.df['Pitch_angle_3_mean']
        diff_2_3 = self.df['Pitch_angle_2_mean'] - self.df['Pitch_angle_3_mean']

        ba_mean = self.df['Pitch_angle_1_mean'].mean()
        bb_mean = self.df['Pitch_angle_2_mean'].mean()
        bc_mean = self.df['Pitch_angle_3_mean'].mean()
        # b) Power
        diff_power = self.df['Power_sensor_mean'] - self.dv['Virtual power']

        is_encoder_anomaly = ( \
            (diff_1_2.abs() > 0.001) | (diff_1_3.abs() > 0.001) | (diff_2_3.abs() > 0.001) \
        ) & diff_power > 2 # Tem que ver  gráfico do sujo com o limpo

        if is_encoder_anomaly.any() == True:
            failure_indices = is_encoder_anomaly[is_encoder_anomaly].index # Armazena os índices com falhas
            self.failure_index = failure_indices[0]
            self.failure_duration = len(failure_indices)
            # Determinar qual é o sensor com falha
            if ba_mean / ((bb_mean + bc_mean) / 2) > 1:
                self.affected_sensor = 'AE'
            elif bb_mean / ((ba_mean + bc_mean) / 2) > 1:
                self.affected_sensor = 'BE'
            elif bc_mean / ((ba_mean + bb_mean) / 2) > 1:
                self.affected_sensor = 'CE'
            else: # Acho que deve ter algo parecido aqui
                self.affected_sensor = None
            return True        
        
        # Detecção de anomalia nos tacômetros
        # Cálculo do ratio entre os tacômetros
        ratio = self.df[self.sensor_map['RT'][0]] * self.Ng / self.df[self.sensor_map['GT'][0]]

        # Ajuste de outlier na primeira amostra
        ratio.iloc[0] = (ratio.iloc[1] + ratio.iloc[2]) / 2  

        # Definição do limite de anomalia
        upper_threshold = 1.1
        lower_threshold = 0.8

        # Há anomalia nos tacômetros?
        is_tach_anomaly = (ratio > upper_threshold) | (ratio < lower_threshold)

        if is_tach_anomaly.any() == True:
            failure_indices = is_tach_anomaly[is_tach_anomaly].index # Armazena os índices com falhas
            self.failure_index = failure_indices[0] # Retorna o primeiro índice ###################################### PEGAR A LISTA
            self.failure_duration = len(failure_indices)
            # Determina qual sensor está falhando com base no valor do ratio
            if ratio.iloc[self.failure_index] > upper_threshold:
                self.affected_sensor = 'RT'
            elif ratio.iloc[self.failure_index] < lower_threshold:
                self.affected_sensor = 'GT'
            else:
                self.affected_sensor = None            
            self.df.loc[is_tach_anomaly, "y_pred"] = 1
            return True
        return False
 
    def found_consecutive(self, values, consecutive_points):
        counter = 1  # Iniciamos com 1 porque já consideramos o primeiro valor como parte da sequência
        repeated_value = None
        
        for i in range(1, len(values)):  # Começamos no segundo elemento
            if values[i] == values[i - 1]:
                counter += 1
                if counter == consecutive_points:
                    repeated_value = values[i]
                    return True, repeated_value
            else:
                counter = 1  # Reinicia a contagem se os valores forem diferentes
        return False, None
    
    def diagnose_failure(self):
        if self.affected_sensor is None:
            self.failure_type = "It didn't even go through the failure analysis"
            return self.failure_type
        
        faulty_sensor, faulty_std_sensor = self.sensor_map[self.affected_sensor]
        failure_period = self.df.loc[self.failure_index:self.failure_index + self.failure_duration]

        # 1️⃣ Fixed value
        is_fixed_by_mean, fixed_repeated_value = self.found_consecutive(failure_period[faulty_sensor].values, self.consecutive_points)
        is_fixed_by_std, _ = self.found_consecutive(failure_period[faulty_std_sensor].values, self.consecutive_points)

        if is_fixed_by_mean and is_fixed_by_std:
            self.failure_type = f"The sensor ({self.affected_sensor}) is stucked with the value of {fixed_repeated_value}"
            return self.failure_type
        
        # 2️⃣ Gain fault
        # Selecting the reference sensor
        if self.affected_sensor in ["AE", "BE", "CE"]:
            copy_blades = ["AE", "BE", "CE"] # Copying the list of encoders
            copy_blades.remove(self.affected_sensor) # Removing the faulty encoder from the list
            reference_sensor_key = copy_blades[random.randint(0, len(copy_blades) - 1)] # Selecting randomly the remaining sensor
            reference_sensor = failure_period[self.sensor_map[reference_sensor_key][0]] # Just select the value of the chosen blade sensor
        else: # Se for no tacômetro
            if self.affected_sensor == "RT":
                reference_sensor = failure_period[self.sensor_map['GT'][0]] / self.Ng
            else:
                reference_sensor = failure_period[self.sensor_map['RT'][0]] * self.Ng
        
        gain_ratio = abs(failure_period[faulty_sensor] / reference_sensor)
        is_gain_by_ratio, gain_repeated_value = self.found_consecutive(gain_ratio.values, self.consecutive_points)
        
        if is_gain_by_ratio == True:
            self.failure_type = f"The sensor presents a gain fault at {self.affected_sensor}.  \n"
            self.failure_type += f"The gain ratio of the signal is {gain_repeated_value}.  \n"
            return self.failure_type
        
        # 3️⃣ Falha por Drift (placeholder - lógica futura)
        self.failure_type = ("No fault patterns were found in the knowledge base.")
        return self.failure_type
    
    # def detect_trend1(df, column, window_size=5, threshold=1.0):
    #     '''Trend detection by moving average'''
    #     # Calculate the moving average
    #     moving_average = df[column].rolling(window=window_size).mean()

    #     # Create a list to store the results
    #     trend_detected = [0] * len(df)
    #     fault_start_time = None
    #     inclination = None

    #     # Check if the moving average exceeds the threshold at each point
    #     for i in range(window_size, len(moving_average)):
    #         if moving_average.iloc[i] > threshold:
    #             trend_detected[i] = 3

    #             # Calculate the inclination (slope) for the window
    #             window_data = df[column].iloc[i-window_size+1:i+1]
    #             x_values = np.arange(window_size)  # x values (0, 1, 2, ..., window_size-1)
    #             slope, _ = np.polyfit(x_values, window_data, 1)  # Fit a line

    #             if fault_start_time is None:  # Store the first occurrence of the fault
    #                 fault_start_time = df.index[i]
    #                 inclination = slope

    #     return trend_detected, fault_start_time, inclination

    def describe_anomaly(self):
        """Fornece uma descrição da anomalia detectada."""
        if self.affected_sensor is None:
            return "✅ No anomaly detected!"
        else:
            return (
                f"⚠️ Fault detected at the sensor: **{self.decode.get(self.affected_sensor)}**  \n"
                f"📝 Fault type: **{self.failure_type}**  \n"
                f"🟢 Fault index: **{self.failure_index}**  \n"
                f"🕒 Duration: **{self.failure_duration}s**"
                )

    def metrics(self):
        """Gera e retorna a matriz de confusão entre y_pred e y_true."""
        if "y_true" not in self.df.columns:
            raise ValueError("O DataFrame precisa conter a coluna 'y_true' para calcular a matriz de confusão.")
        
        cm = confusion_matrix(self.df["y_true"], self.df["y_pred"])
        tn, fp, fn, tp = cm.ravel()
        td = 1 # Detection time - Tem que ver de onde vem esse atraso para detectar a falha.
        return td, tn, fp, fn, tp

    def run(self):
        """Executa todas as etapas da detecção e diagnóstico."""
        if "y_true" in self.df.columns and self.df["y_true"].sum() == 0: # Acho que posso deletar isso aqui
            return "✅ Sistema em condição saudável. Nenhuma falha foi encontrada.", None, None, None, None # Acho que posso deletar isso aqui
        anomaly_found = self.detect_anomalies() # Dynamic binding
        if anomaly_found == True:
            self.diagnose_failure()
        
        description = self.describe_anomaly()
        td, tn, fp, fn, tp = self.metrics()
        return description, td, tn, fp, fn, tp
    
    def run_ml(self):
        pass
##############################################################
###################### Dataset de teste ######################
##############################################################
from scipy.stats import linregress
import numpy as np
class SensorFaultDetector:
    def __init__(self, df):
        self.df = df.copy()
        self.labels = [0] * len(df)
    ####### Methods to classify faults #######
    def detect_constant_value(self, label_val, df, column, 
                              window_size=5,
                              threshold=0.011,
                              consecutive_points=6):
        moving_average = df[column].rolling(window=window_size).mean()
        label = [0] * len(df)
        consecutive_count = 0

        for i in range(window_size, len(moving_average)):
            if i > window_size:
                variation = abs(moving_average.iloc[i] - moving_average.iloc[i - 1])
                if variation <= threshold:
                    consecutive_count += 1
                else:
                    consecutive_count = 0

                if consecutive_count >= consecutive_points:
                    label[i] = label_val

        return label, None, None

    def detect_gain(self, label_val, df, column1, column2, Ng = None, 
                    window_size=6,
                    threshold=0.15,
                    consecutive_points=20):
        if Ng is not None: # For tachometers
            ratio = Ng * df[column1] / df[column2]
        else: # For encoders
            ratio = df[column1] / df[column2]

        moving_average = ratio.rolling(window=window_size).mean()
        label = [0] * len(df)
        consecutive_count = 0

        for i in range(window_size, len(moving_average)):
            if pd.notna(moving_average.iloc[i]):
                if moving_average.iloc[i] < (1 - threshold) or moving_average.iloc[i] > (1 + threshold):
                    consecutive_count += 1
                else:
                    consecutive_count = 0

                if consecutive_count >= consecutive_points:
                    label[i] = label_val

        return label, None, None

    def detect_trend(self, label_val, df, column,
                     window_size=6,
                     threshold=0.9):
        moving_average = df[column].rolling(window=window_size).mean()
        label = [0] * len(df)

        for i in range(window_size, len(moving_average)):
            if moving_average.iloc[i] > threshold:
                label[i] = label_val

        return label, None, None
    ####### Binary classification #######
    def binary_cm(self, y_true, y_pred, fault_label):
        y_true_binary = [1 if y == fault_label else 0 for y in y_true]
        y_pred_binary = [1 if y == fault_label else 0 for y in y_pred]
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        return cm

    def metrics(self, cm):
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return precision, recall, f1_score, accuracy

    ####### Multiclass classification #######
    def detect_faults(self):
        # Falha 1: Valor fixo no encoder
        fixed1, _, _ = self.detect_constant_value(1, self.df, 'Pitch_angle_1_mean',
                                                  window_size=4,
                                                  threshold=0.011,
                                                  consecutive_points=6)
        # Falha 2: Ganho no encoder
        gain2, _, _ = self.detect_gain(2, self.df, 'Pitch_angle_1_mean', 'Pitch_angle_2_mean',
                                       window_size=6,
                                       threshold=0.15,
                                       consecutive_points=20)
        # Falha 3: Drift no encoder
        drift3, _, _ = self.detect_trend(3, self.df, 'Pitch_angle_3_mean',
                                         window_size=10,
                                         threshold=0.9)
        # Falha 4: Valor fixo no tacômetro
        fixed4, _, _ = self.detect_constant_value(4, self.df, 'Rotor_speed_sensor_mean',
                                                  window_size=5,
                                                  threshold=0.012,
                                                  consecutive_points=2)
        # Falha 5: Ganho no tacômetro
        gain5, _, _ = self.detect_gain(5, self.df, 'Rotor_speed_sensor_mean', 'Generator_speed_sensor_mean', Ng=95,
                                       window_size=6,
                                       threshold=0.035,
                                       consecutive_points=20)

        for i in range(len(self.df)):
            if fixed1[i]: 
                self.labels[i] = 1
            elif fixed4[i]: 
                self.labels[i] = 4
            elif gain2[i]: 
                self.labels[i] = 2
            elif gain5[i]: 
                self.labels[i] = 5
            elif drift3[i]: 
                self.labels[i] = 3
            else:
                self.labels[i] = 0 # Fault-free

        return self.labels
    
    def multiclass_cm(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=range(6))
        return cm
    
    def sa_mlas(self, df):
        "Method to develop machine learning algorithms on sensitivity analysis datasets"
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=['Label'])
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Calcular a frequência das classes em y_test
        y_test_series = pd.Series(y_test)
        class_counts = y_test_series.value_counts()
        # Calcular pesos de classe inversamente proporcionais
        total_samples = len(y_test)
        class_weights = {label: total_samples / (len(class_counts) * count) for label, count in class_counts.items()}

        from sklearn.ensemble import StackingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.linear_model import SGDClassifier
        from sklearn.svm import LinearSVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

        stacking_clf = StackingClassifier(
            estimators=[
                ('qda', QuadraticDiscriminantAnalysis(reg_param = 1)),
                ('sgd', SGDClassifier(loss="hinge",max_iter=500,tol=1e-3,
                                    alpha=1e-3,
                                    penalty='l1',
                                    random_state=42)),
                ('svm', LinearSVC(C=1000, random_state=42, dual='auto', max_iter=20000,
                                class_weight=class_weights)),
                ('dt', DecisionTreeClassifier(max_depth=11,
                                            min_samples_split=2,
                                            class_weight=class_weights,
                                            random_state=42, criterion='entropy')),
                ('rf', RandomForestClassifier(max_depth=12,
                                            min_samples_split=8,
                                            class_weight=class_weights,
                                            random_state=42)),
                ('et', ExtraTreesClassifier(max_depth=20,
                                            min_samples_split=5,
                                            class_weight=class_weights,
                                            random_state=42, bootstrap = False)) ],
            final_estimator=RandomForestClassifier(),
            cv=5)

        pipe_stacking = make_pipeline(StandardScaler(), stacking_clf)

        # Ajustar o pipeline com os dados de treino
        pipe_stacking.fit(X_train, y_train)

        #Generate predictions with the model using our X values
        y_pred_stack = pipe_stacking.predict(X_test) # At test dataset!!
        return y_test, y_pred_stack
    
    def metrics_multi(df, y_test, y_pred):
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        return recall, precision
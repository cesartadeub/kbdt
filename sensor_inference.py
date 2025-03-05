from sklearn.metrics import confusion_matrix
import random

class SensorInferenceSystem:
    def __init__(self, df, tolerance=0.01, consecutive_points=3, Ng=95):
        self.df = df.copy()
        self.tolerance = tolerance
        self.consecutive_points = consecutive_points
        self.Ng = Ng
        self.affected_sensor = None
        self.failure_type = None
        self.failure_start_index = None
        self.failure_duration = 0
        self.df["y_pred"] = 0
        
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
        blade_encoders = {key: value[0] for key, value in self.sensor_map.items() if key in ['AE', 'BE', 'CE']}
        blade_encoders_std = [value[1] for key, value in self.sensor_map.items() if key in ['AE', 'BE', 'CE']]
        tachometers = {key: value[0] for key, value in self.sensor_map.items() if key in ['RT', 'GT']}
        tachometers_std = [value[1] for key, value in self.sensor_map.items() if key in ['RT', 'GT']]

        # Anomaly detection on encoders
        self.df['median_encoder'] = self.df[list(blade_encoders.values())].median(axis=1)
        discrepancies = {key: abs(self.df[value] - self.df['median_encoder']) for key, value in blade_encoders.items()}
        dynamic_threshold = self.df[blade_encoders_std].mean(axis=1) * self.tolerance

        for encoder, diff in discrepancies.items():
            anomaly_mask = diff > dynamic_threshold
            if anomaly_mask.any():
                self.affected_sensor = encoder
                self.failure_start_index = anomaly_mask.idxmax()
                self.failure_duration = anomaly_mask.sum()
                self.df.loc[anomaly_mask, "y_pred"] = 1
                return True
        
        # Detecção de anomalia nos tacômetros
        # Cálculo correto do ratio entre os tacômetros
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
            self.failure_start_index = failure_indices[0] # Retorna o primeiro índice
            self.failure_duration = len(failure_indices)
            # Determina qual sensor está falhando com base no valor do ratio
            if ratio.iloc[self.failure_start_index] > upper_threshold:
                self.affected_sensor = 'RT'
            elif ratio.iloc[self.failure_start_index] < lower_threshold:
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
        failure_period = self.df.loc[self.failure_start_index:self.failure_start_index + self.failure_duration]

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

    def describe_anomaly(self):
        """Fornece uma descrição da anomalia detectada."""
        if self.affected_sensor is None:
            return "✅ No anomaly detected!"
        else:
            return (
                f"⚠️ Fault detected at the sensor: **{self.decode.get(self.affected_sensor)}**  \n"
                f"📝 Fault type: **{self.failure_type}**  \n"
                f"🟢 Fault begin: **{self.failure_start_index}**  \n"
                f"🕒 Duration: **{self.failure_duration}s**"
                )

    def metrics(self):
        """Gera e retorna a matriz de confusão entre y_pred e y_true."""
        if "y_true" not in self.df.columns:
            raise ValueError("O DataFrame precisa conter a coluna 'y_true' para calcular a matriz de confusão.")
        
        cm = confusion_matrix(self.df["y_true"], self.df["y_pred"])
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        pod = tp / (tp + fn)
        pofa = fp / (fp + tn)
        return cm, accuracy, pod, pofa

    def run(self):
        """Executa todas as etapas da detecção e diagnóstico."""
        if "y_true" in self.df.columns and self.df["y_true"].sum() == 0:
            return "✅ Sistema em condição saudável. Nenhuma falha foi encontrada.", None, None, None, None
        anomaly_found = self.detect_anomalies()
        if anomaly_found:
            self.diagnose_failure()
        
        description = self.describe_anomaly()
        cm, accuracy, pod, pofa = self.metrics()
        return description, cm, accuracy, pod, pofa
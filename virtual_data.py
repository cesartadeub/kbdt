import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

class TurbineOptimizer:
    def __init__(self, df, n_clusters, Pnom):
        self.df = df
        self.n_clusters = n_clusters
        self.Pnom = Pnom
#############################################################
    def objective_function(self, wind_speed, gain, shift, bias, function='sigmoid'):
        """Objective function to adjust the power curve. 
        Pnom: Constant value
        wind_speed = List of wind speed values
        Gain: float, controls the steepness (slope) of the curve
        Shift: float, horizontal shift (x-axis)
        Bias: float, vertical shift (y-axis)
        """
        if function == 'sigmoid':
            return self.Pnom / (1 + np.exp(-gain * (wind_speed - shift))) + bias
        elif function == 'tanh':
            ex = np.exp(gain * (wind_speed - shift))
            enx = np.exp(-gain * (wind_speed - shift)) 
            return bias + (self.Pnom / 2) * (ex - enx) / (ex + enx)
        else:
            raise ValueError(f"Function '{function}' is not supported. Choose between 'sigmoid' or 'tanh'")

    def fit_objective_function(self, xc, yc, params, function):
        """Ajusta a função sigmoidal aos centros do KMeans."""
        gain_min, shift_min, bias_min = params[0:3]
        gain_max, shift_max, bias_max = params[3:6]

        popt, _ = curve_fit(
            lambda xc, gain, shift, bias: self.objective_function(xc, gain, shift, bias, function),
            xc, yc, bounds = ([gain_min, shift_min, bias_min],
                            [gain_max, shift_max, bias_max])
        )
        return popt
    
    def compute_control_limits(self, df, popt, clusters, n_sigma=3):
        """Calculates control limits with n_sigma-sigma and marks outliers in the DataFrame."""
        df['cluster'] = clusters

        # 1. Calculation of wind speed standard deviation per cluster
        stds = df.groupby('cluster')['Wind_speed_mean'].std()
        # 2. Average of standard deviations
        sigma = stds.mean()
        # 3. Generates the central curve and horizontally displaced limits
        x_vals = np.linspace(df['Wind_speed_mean'].min(), df['Wind_speed_mean'].max(), 
                            len(df['Wind_speed_mean']))
        y_middle = self.objective_function(x_vals, *popt, function='sigmoid')
        y_upper  = self.objective_function(x_vals + n_sigma * sigma, *popt, function='sigmoid')
        y_lower  = self.objective_function(x_vals - n_sigma * sigma, *popt, function='sigmoid')
        # 4. Evaluate the curve at the actual point of each sample (not interpolated)
        y_upper_each = self.objective_function(df['Wind_speed_mean'] + n_sigma * sigma, *popt, function='sigmoid')
        y_lower_each = self.objective_function(df['Wind_speed_mean'] - n_sigma * sigma, *popt, function='sigmoid')
        # 5. Mark the outliers
        df['outlier'] = ~df['Power_sensor_mean'].between(y_lower_each, y_upper_each)

        return y_middle, y_upper, y_lower
    
    ############# CLUSTERING #############
    def kmeans_centers(self, df, n_clusters=10):
        """Applies KMeans to obtain power curve centers."""
        Xc = df[['Wind_speed_mean', 'Power_sensor_mean']]
        kmeans = KMeans(n_clusters=n_clusters, algorithm = "lloyd", random_state=42, n_init=10)
        clusters = kmeans.fit_predict(Xc)
        x_centers = kmeans.cluster_centers_[:, 0]
        y_centers = kmeans.cluster_centers_[:, 1]
        return clusters, x_centers, y_centers

    def anomaly_analysis(self, df_raw, Pnom, n_clusters=10):
        """Performs clustering, fits the power curve, and identifies anomalies.
        Returns:
        df - DataFrame with columns y_middle, y_upper, y_lower, and Is_anomaly
        clusters - Labels of the clusters (np.array)
        x_centers - X-coordinates of the centroids
        y_centers - Y-coordinates of the centroids
        y_middle - Curve fitted over the clusters
        y_upper - Upper control limit
        y_lower - Lower control limit
        """
        sig = 2
        df = df_raw.copy()
        # 1. Clustering
        clusters, x_centers, y_centers = self.kmeans_centers(df, n_clusters)
        # 2. Sigmoidal curve fitting
        intervals = [0.1, 6, 0, # Minimum interval: gain, shift, bias
                    1, 10, 1]   # Maximum interval: gain, shift, bias
        popt = self.fit_objective_function(x_centers, y_centers, params=intervals, function='sigmoid')
        # 3. Calculation of control limits and identification of outliers
        y_middle, y_upper, y_lower = self.compute_control_limits(df, popt, clusters, n_sigma=sig)
        # 4. Evaluation and tagging on DataFrame
        df['y_middle'] = self.objective_function(df['Wind_speed_mean'], *popt, function='sigmoid')
        df['y_upper']  = self.objective_function(df['Wind_speed_mean'] + sig*df.groupby('cluster')['Wind_speed_mean'].transform('std').mean(),
                                            *popt, function='sigmoid')
        df['y_lower']  = self.objective_function(df['Wind_speed_mean'] - sig*df.groupby('cluster')['Wind_speed_mean'].transform('std').mean(),
                                            *popt, function='sigmoid')
        df['Is_anomaly'] = df['outlier'].astype(int)

        return df, clusters, x_centers, y_centers, y_middle, y_upper, y_lower
    
    def virtual_metrics(self, df):
        """Returns:
        - R² between the power dispersion data and the fitting curve.
        - MAE between the dispersion data and the curve.
        - MSE between the centroids and the fitting curve.
        """
        # Raw data
        Xc = df[['Wind_speed_mean', 'Power_sensor_mean']]
        # Curva ajustada: já gerada no df
        y_pred = df['y_middle']
        # R²
        y_mean = Xc['Power_sensor_mean'].mean()
        ss_res = np.sum((Xc['Power_sensor_mean'] - y_pred) ** 2)
        ss_tot = np.sum((Xc['Power_sensor_mean'] - y_mean) ** 2)
        r2 = 1 - ss_res / ss_tot
        # MAE
        mae = np.mean(np.abs(Xc['Power_sensor_mean'] - y_pred))
        # Faltou o MSE da curva de ajuste com os centróides
        return r2, mae
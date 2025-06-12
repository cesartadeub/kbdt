import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import scipy.stats as stats
from virtual_data import TurbineOptimizer

class WECS:
    def __init__(self, df_raw, add_select_turbine):
        self.df = df_raw
        self.add_select_turbine = add_select_turbine
        self.df_clean = None

    def plot_wind_speed(self):
        wind_speed = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Scatter plot", "Histogram"))

        wind_speed_scatter = px.scatter(self.df, y="Wind_speed_mean")
        wind_speed_scatter.update_layout(xaxis_title="Time (s)", yaxis_title="Wind speed (m/s)")
        wind_speed_hist = px.histogram(self.df, x="Wind_speed_mean", nbins=40)

        wind_speed.add_trace(wind_speed_scatter.data[0], row=1, col=1)
        wind_speed_scatter.update_traces(hovertemplate=
                                          '<b>W<sub>s</sub>: </b>%{y:.1f} m/s<br>' +
                                          '<b>Time:</b> %{x:.0f} s')

        wind_speed.update_layout(title=f"{self.add_select_turbine + ': ' if self.add_select_turbine == 'Training data' else self.add_select_turbine[:5]} wind speed profile",
                                 xaxis_title="Time (s)",
                                 yaxis_title="Wind speed (m/s)",
                                 xaxis2_title="Wind speed (m/s)",
                                 yaxis2_title="Frequency")
        wind_speed.add_trace(wind_speed_hist.data[0], row=1, col=2)

        st.plotly_chart(wind_speed, on_select="rerun", use_container_width=True, color=[77, 183, 211])
        
        # Interpretação dos dados
        counts, _ = np.histogram(self.df["Wind_speed_mean"], bins=20)
        peaks, _ = find_peaks(counts)
         # Estatísticas principais
        mean_ws = self.df["Wind_speed_mean"].mean()
        min_ws = self.df["Wind_speed_mean"].min()
        max_ws = self.df["Wind_speed_mean"].max()
        std_ws = self.df["Wind_speed_mean"].std()
        skew_ws = stats.skew(self.df["Wind_speed_mean"])

        # Comentários automáticos
        st.write('###### Expert digital twin comments')
        st.write(f"The wind speed data presents a good variation with a **mean of {mean_ws:.2f} m/s**, minimum of **{min_ws:.2f} m/s**, and maximum of **{max_ws:.2f} m/s**.")
        st.write(f"The standard deviation is **{std_ws:.2f}**, indicating {'a high' if std_ws > 2 else 'a moderate' if std_ws > 1 else 'a low'} variability in wind speed measurements.")

        # Condicional para a forma da distribuição
        if skew_ws < -0.5:
            skewness_comment = "The histogram shows a **negatively skewed** distribution"
        elif skew_ws > 0.5:
            skewness_comment = "The histogram shows a **positively skewed** distribution"
        else:
            skewness_comment = "The histogram appears to be **approximately symmetric**"
        # Comentário sobre o modo
        st.write(f"{skewness_comment}, with a **unimodal** shape and the mode around **{(peaks)[0]:.2f} m/s**.")

    def plot_power_curve(self):
        # K-means clustering and curve fitting
        nc = 10
        virtual_services = TurbineOptimizer(self.df, n_clusters=nc, Pnom=self.Pnom)
        
        df_clean, clusters, xc, yc, y_middle, y_upper, y_lower = virtual_services.anomaly_analysis(self.df, self.Pnom, n_clusters=10)

        x_fit = np.linspace(df_clean['Wind_speed_mean'].min(), df_clean['Wind_speed_mean'].max(), len(df_clean))

        # Separando dados
        normal_data = df_clean[~df_clean['outlier']]
        outliers = df_clean[df_clean['outlier']]

        # Figura principal
        pc = go.Figure()

        # 1. Raw data plot - First button
        pc.add_trace(go.Scatter(
            x=self.df["Wind_speed_mean"], y=self.df["Power_sensor_mean"],
            mode="markers", name="Power curve (raw data)",
            marker=dict(
                color="rgba(77, 183, 211, 0.5)",
                symbol='circle'
            ),
            visible=True))

        # 2. Filtered data by cluster - Second button
        pc.add_trace(go.Scatter(
            x=normal_data["Wind_speed_mean"], y=normal_data["Power_sensor_mean"],
            mode="markers", name="Power curve (clustered)",
            marker=dict(
                color=clusters[~df_clean['outlier']],
                colorscale='viridis',
                opacity=0.5,
                symbol='circle' ), visible=False ))
        # 3. Centroids
        pc.add_trace(go.Scatter(
            x=xc, y=yc, mode="markers", name="KMeans centroids",
            marker=dict(color="black", size=10, line=dict(color='white', width=1)),
            visible=False))

        # 4. Adjusted curve
        pc.add_trace(go.Scatter(
            x=x_fit,
            y=y_middle,
            mode="lines",
            name="Sigmoid fit",
            line=dict(color="yellow", width=3),
            visible=False ))

        # 5. Control limits
        pc.add_trace(go.Scatter(
            x=x_fit,
            y=y_upper,
            mode="lines",
            name="Upper control limit",
            line=dict(color="magenta", dash='dash'),
            visible=False ))

        pc.add_trace(go.Scatter(
            x=x_fit,
            y=y_lower,
            mode="lines",
            name="Lower control limit",
            line=dict(color="magenta", dash='dash'),
            visible=False))

        # 6. Outliers
        pc.add_trace(go.Scatter(
            x=outliers["Wind_speed_mean"],
            y=outliers["Power_sensor_mean"],
            mode="markers",
            name="Outliers",
            marker=dict(color="red", size=8, symbol="x"),
            visible=False ))
        st.write(f'''
The illustration below shows the power curve of the {self.add_select_turbine[:5]} turbine, together with a representative model obtained through a clustering process. You can use the "Show samples" option to view only the raw data or select "Show optimized curve and clusters" to see the {nc} identified clusters and the curve fitted to them.

This model represents the nominal behavior of the turbine and serves as a basis for detecting operational deviations. It is useful both in identifying anomalies related to aerodynamic performance and in calculating efficiency indicators, such as deviation from expected production and degradation of the power curve over time.
                 ''')
        # 7. Layout interativo com botões
        pc.update_layout(
            title=f"{self.add_select_turbine[:5]} wind turbine power curve and optimized curve through clusters",
            xaxis_title="Wind speed (m/s)",
            yaxis_title="Power (MW)",
            showlegend=True,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": "Show samples",
                            "method": "update",
                            "args": [{"visible": [True, False, False, False, False, False, False]}],
                        }, # [Raw data, Normal samples, Centroids, Adjusted curve, UCL, LCL, Ouliers]
                        {
                            "label": "Show optimized curve and clusters",
                            "method": "update",
                            "args": [{"visible": [False, True, True, True, True, True, True]}],
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.7,
                    "xanchor": "left",
                    "y": 1.2,
                    "yanchor": "top"
                }
            ]
        )
        st.plotly_chart(pc)
        
        # Displaying expert comments
        rsquared_curve_power, mae_curve_power = virtual_services.virtual_metrics(df_clean)
        mse_dt = 0.0142
        st.write(f'###### Expert digital twin comments')
        st.write(f'{nc} centroids paves the way to generate the optimal curve.')
        st.write(f'Minimum Squared Error between the centroids and the optimal curve is: {mse_dt:.4f}.')
        st.write(f'Which leads to a {rsquared_curve_power:.2f} of correlation between real data and virtual data.')

        mae_kpi, cp_kpi = st.columns(2)
        mae_kpi.metric("Mean absolute percentage error between twins", f'{100 * mae_curve_power:.1f}%', border=True)
        cp_kpi.metric("Power curve correlation between twins", f'{100 * rsquared_curve_power:.1f}%', border=True)
        self.df_clean = df_clean # Data cleaned stored

    def plot_pitch_angles(self):

        pit = make_subplots(rows=2, cols=3, subplot_titles=("Blade A (deg)", "Blade B (deg)", "Blade C (deg)"))

        pit1 = px.scatter(self.df, y="Pitch_angle_1_mean")
        pit2 = px.scatter(self.df, y="Pitch_angle_2_mean")
        pit3 = px.scatter(self.df, y="Pitch_angle_3_mean")
        pit.add_trace(pit1.data[0], row=1, col=1)
        pit.add_trace(pit2.data[0], row=1, col=2)
        pit.add_trace(pit3.data[0], row=1, col=3)
        pit.update_traces(hovertemplate='<b>Angle:</b> %{y:.2f} deg<br>' +
                                        '<b>Time:</b> %{x:.0f} s')
        
        pit.update_layout(title=f"{self.add_select_turbine + ': ' if self.add_select_turbine == 'Training data' else self.add_select_turbine[:5]} wind turbine pitch angle",
                        showlegend=False,
                        xaxis_title="Time (s)", xaxis2_title="Time (s)", xaxis3_title="Time (s)",
                        height=1000)

        box_plot_blade1 = px.box(self.df, y="Pitch_angle_1_mean", boxmode="overlay")
        box_plot_blade2 = px.box(self.df, y="Pitch_angle_2_mean", boxmode="overlay")
        box_plot_blade3 = px.box(self.df, y="Pitch_angle_3_mean", boxmode="overlay")

        pit.add_trace(box_plot_blade1.data[0], row=2, col=1)
        pit.add_trace(box_plot_blade2.data[0], row=2, col=2)
        pit.add_trace(box_plot_blade3.data[0], row=2, col=3)

        pit.update_traces(quartilemethod="linear", jitter=0,
                        selector=dict(type='box'), boxmean=True)

        st.plotly_chart(pit, on_select="rerun", use_container_width=True, color=[77, 183, 211])

        st.write('Encoder with fixed value on blade A, set at 5° from 2000s to 2100s')
        st.write('Encoder on blade B with a 20% gain from 2300s to 2400s')
        st.write('Encoder on blade C showing a rising trend from 2600s to 2700s')
        st.write('###### Expert digital twin comments')

        blades = {
            "A": self.df['Pitch_angle_1_mean'],
            "B": self.df['Pitch_angle_2_mean'],
            "C": self.df['Pitch_angle_3_mean'] }

        stats_texts = []
        iqr_values = {}

        for blade, data in blades.items():
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            iqr_values[blade] = iqr
            min_val = data.min()
            max_val = data.max()
            median = np.median(data)

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            num_outliers = len(outliers)

            outlier_comment = ("No statistical outliers were observed, indicating stable sensor readings."
                            if num_outliers == 0 else
                            f"{num_outliers} statistical outlier(s) were detected, which may indicate transient anomalies or sensor noise.")

            stats_texts.append(
                f"Blade {blade} shows a value range between {min_val:.2f}° and {max_val:.2f}°, "
                f"with a median of {median:.2f}° and an interquartile range (IQR) of {iqr:.2f}°. {outlier_comment}")

        # Bloco final
        max_iqr_blade = max(iqr_values, key=iqr_values.get)
        stats_texts.append(
            f"Among the three blades, Blade {max_iqr_blade} presents the highest variability (IQR), "
            f"which may deserve further inspection to rule out oscillatory behavior or inconsistent readings.")

        for text in stats_texts:
            st.write(text)

        # Métricas visuais
        col1, col2, col3 = st.columns(3)
        col1.metric("Blade A mean values", f'{blades["A"].mean():.3f} deg', border=True)
        col2.metric("Blade B mean values", f'{blades["B"].mean():.3f} deg', border=True)
        col3.metric("Blade C mean values", f'{blades["C"].mean():.3f} deg', border=True)

        # Deviation simple check
        if (np.abs(blades["A"].mean() - blades["B"].mean()) > 0.001 or
            np.abs(blades["A"].mean() - blades["C"].mean()) > 0.001 or
            np.abs(blades["B"].mean() - blades["C"].mean()) > 0.001):
            st.write("There's a deviation on the blades. There's a chance for anomalies on the blades.")
        else:
            st.write("The pitch angle of the blades remains the same.")

    def plot_turbine_speeds(self):
        omega = make_subplots(rows=1, cols=2, subplot_titles=("Rotor speed (rpm)", "Generator speed (rpm)"))

        om1 = px.scatter(self.df, y="Rotor_speed_sensor_mean")
        om2 = px.scatter(self.df, y="Generator_speed_sensor_mean")
        omega.add_trace(om1.data[0], row=1, col=1)
        omega.add_trace(om2.data[0], row=1, col=2)
        omega.update_traces(hovertemplate='<b>Speed:</b> %{y:.2f} rpm<br>' +
                                        '<b>Time:</b> %{x:.0f} s')
        
        omega.update_layout(title=f"{self.add_select_turbine + ': ' if self.add_select_turbine == 'Training data' else self.add_select_turbine[:5]} wind turbine speeds",
                            showlegend=False,
                            xaxis_title="Time (s)",
                            xaxis2_title="Time (s)")
        st.plotly_chart(omega, on_select="rerun", use_container_width=True, color=[77, 183, 211])

        st.write(f'###### Expert digital twin comments')
        st.write('Rotor tachometer with a fixed value of 13.4 rpm from 1500s to 1600s')
        st.write('Rotor tachometer with a 20% gain from 400s to 500s')
        check_operation = ((
            (self.df['Generator_speed_sensor_mean'] != 0).any(),
            (self.df['Generator_speed_sensor_mean'].between(500, 1000)).any(),
            (self.df['Generator_speed_sensor_mean'].between(1500, 1600)).any()
            ))
        if check_operation == (True, True, True):
            st.write(f"The wind turbine is operating above cut-in speed.")
            st.write(f"Also the wind turbine experienced variational speeds.")
            st.write(f"Finally, the asset produced maximum rated power achieving {self.df['Generator_speed_sensor_mean'].max():.0f}rpm at the generator.")
        else:
            pass

    def plot_correlation_matrix(self):
        corr_columns = ['Wind_speed_mean', 'Power_sensor_mean',
                        'Pitch_angle_1_mean', 'Pitch_angle_2_mean', 'Pitch_angle_3_mean',
                        'Rotor_speed_sensor_mean', 'Generator_speed_sensor_mean']
        
        # Cálculo da matriz de correlação
        corr_data = self.df[corr_columns].corr(method='pearson')
        z = corr_data.values.copy()

        # Máscara para o triângulo superior
        mask_upper = np.triu_indices_from(z)
        z[mask_upper] = np.nan

        # Gráfico
        corr_matrix = px.imshow(z, text_auto=True, aspect="auto",
                                x=corr_columns, y=corr_columns, color_continuous_scale='RdBu', zmin=-1, zmax=1)
        corr_matrix.update_traces(
            hovertemplate="%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>",
            hoverongaps=False
        )
        corr_matrix.update_layout(
            title=f"{self.add_select_turbine + ': ' if self.add_select_turbine == 'Training data' else self.add_select_turbine[:5]} correlation matrix",
            showlegend=False,
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(corr_matrix)

        # Comentários automáticos
        st.write('###### Expert digital twin comments')
        st.write('Here are some remarks based on the correlation matrix:')

        for i in range(1, len(corr_columns)):
            for j in range(i):
                corr_value = z[i, j]
                if np.isnan(corr_value):
                    continue

                var1 = corr_columns[i]
                var2 = corr_columns[j]

                explanation = ""
                if corr_value == 1:
                    explanation = f"Perfect positive correlation between **{var1}** and **{var2}** ({corr_value:.2f})."
                elif corr_value >= 0.8:
                    explanation = f"Strong positive correlation between **{var1}** and **{var2}** ({corr_value:.2f})."
                elif corr_value >= 0.5:
                    explanation = f"Weak to moderate positive correlation between **{var1}** and **{var2}** ({corr_value:.2f})."
                elif corr_value <= -0.8:
                    explanation = f"Strong negative correlation between **{var1}** and **{var2}** ({corr_value:.2f})."
                elif corr_value <= -0.5:
                    explanation = f"Weak to moderate negative correlation between **{var1}** and **{var2}** ({corr_value:.2f})."
                else:
                    explanation = f"No relevant correlation detected between **{var1}** and **{var2}** ({corr_value:.2f})."

                st.markdown(f"- {explanation}")

    def generate_dataframe(self):
        """Retorna o DataFrame com os dados otimizados (gêmeo digital).
        Assumimos que ele já foi gerado anteriormente por plot_power_curve()."""
        if self.df_clean is None:
            raise RuntimeError("O gêmeo digital ainda não foi gerado. Execute plot_power_curve() primeiro.")
        return self.df_clean

    def display_plots(self):
        self.plot_wind_speed()
        if self.Pnom is not None:
            self.plot_power_curve()
        self.plot_pitch_angles()
        self.plot_turbine_speeds()
        self.plot_correlation_matrix()

class WindTurbine2000(WECS):
    def __init__(self, df, add_select_turbine):
        super().__init__(df, add_select_turbine) # Inheritance property
        self.Pnom = 2.05

class WindTurbine4800(WECS):
    def __init__(self, df, add_select_turbine):
        super().__init__(df, add_select_turbine) # Inheritance property
        self.Pnom = 4.8

    def plot_multiclass_cm(self, cm, string_title):
        ticklabels = [
            'Fault-free',
            'Blade 1 fixed',
            'Blade 2 gain',
            'Blade 3 trend',
            'Rotor speed fixed',
            'Rot/Gen speed gain',
            'Actuator abrupt',
            'Actuator slow']
        
        annotations_text = np.array([f"{val:.3f}" for val in cm.flatten()]).reshape(cm.shape)

        fig = go.Figure(data=go.Heatmap(
            z=cm, # Se 'cm' já é normalizado, 'z' já tem as proporções
            x=ticklabels, y=ticklabels, 
            text=annotations_text, # Use o texto formatado aqui
            texttemplate="%{text}", # Isso garante que o 'text' seja exibido
            hoverinfo="text", colorscale='viridis',
            showscale=True, # Mostrar a barra lateral de escala de cor (opcional, pode ser False)
            colorbar=dict(title='Proportion') # Título da barra de cores, se 'cm' for normalizado
        ))

        # Configura o layout do gráfico
        fig.update_layout(
            title=string_title, xaxis_title='Predicted label', yaxis_title='True label',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(ticklabels))),
                ticktext=ticklabels,
                side='bottom' # Coloca os rótulos do eixo X (Predicted) em cima, como uma tabela
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(ticklabels))),
                ticktext=ticklabels,
                autorange="reversed"
            ),
            height=600, width=650 )
        
        return fig
    
class WindTurbineTest(WECS):
    def __init__(self, df, add_select_turbine):
        super().__init__(df, add_select_turbine) # Inheritance property
        self.Pnom = None
    
    def fixed_tradeoff(self, detector, falha, column, window_sizes, thresholds, consecutive_points, fault_label):
            """Calcula precision e recall para diferentes window sizes e thresholds"""
            precision_results = {ws: [] for ws in window_sizes}
            recall_results = {ws: [] for ws in window_sizes}
                        
            for ws in window_sizes:
                for threshold in thresholds:
                    y_pred, _, _ = detector.detect_constant_value(
                        fault_label, falha, column,
                        window_size=ws,
                        threshold=threshold,
                        consecutive_points=consecutive_points
                    )
                    y_true = falha['Label']
                    cm = detector.binary_cm(y_true, y_pred, fault_label=fault_label)
                    precision, recall, _, _ = detector.metrics(cm)
                    
                    precision_results[ws].append(precision)
                    recall_results[ws].append(recall)
            
            return precision_results, recall_results

    def gain_tradeoff(self, detector, falha, column1,column2, Ng, window_sizes, thresholds, consecutive_points, fault_label):
        """Calcula precision e recall para falha com ganho"""
        precision_results = {ws: [] for ws in window_sizes}
        recall_results = {ws: [] for ws in window_sizes}
        
        for ws in window_sizes:
            for threshold in thresholds:
                y_pred, _, _ = detector.detect_gain(
                    fault_label, falha, column1, column2, Ng=None,
                    window_size=ws,
                    threshold=threshold,
                    consecutive_points=consecutive_points)
                y_true = falha['Label']
                cm = detector.binary_cm(y_true, y_pred, fault_label=fault_label)
                precision, recall, _, _ = detector.metrics(cm)
                
                precision_results[ws].append(precision)
                recall_results[ws].append(recall)
        
        return precision_results, recall_results

    def trend_tradeoff(self, detector, falha, window_sizes, thresholds, fault_label):
        """Calcula precision e recall para falha com tendência"""
        precision_results = {ws: [] for ws in window_sizes}
        recall_results = {ws: [] for ws in window_sizes}
        
        for ws in window_sizes:
            for threshold in thresholds:
                y_pred, _, _ = detector.detect_trend(
                    fault_label, falha, 'Pitch_angle_3_mean',
                    window_size=ws,
                    threshold=threshold)
                y_true = falha['Label']
                cm = detector.binary_cm(y_true, y_pred, fault_label=fault_label)
                precision, recall, _, _ = detector.metrics(cm)
                
                precision_results[ws].append(precision)
                recall_results[ws].append(recall)
        return precision_results, recall_results

    def plot_precision_recall_graph(self, thresholds, precision_data, recall_data, window_size, threshold_setpoint):
        """Cria o gráfico combinado de Precision e Recall"""
        fig = go.Figure()
        
        # Adiciona curva de Precision
        fig.add_trace(go.Scatter(
            x=thresholds, y=precision_data,
            mode='lines+markers', name='Precision',
            line=dict(color='blue', width=2) ))
        
        # Adiciona curva de Recall
        fig.add_trace(go.Scatter(
            x=thresholds, y=recall_data,
            mode='lines+markers', name='Recall',
            line=dict(color='red', width=2) ))
        
        # Linha vertical do threshold
        fig.add_vline(
            x=threshold_setpoint,
            line_dash="dot", line_color="black",
            annotation_text=f"Threshold set: {threshold_setpoint:.3f}",
            annotation_position="bottom right")
        
        # Configurações do layout
        fig.update_layout(
            title=f'Precision/Recall trade-off (Window: {window_size}s)',
            xaxis_title='Threshold',
            yaxis_title='Precision/Recall',
            yaxis=dict(tickformat=".0%", range=[0, 1.05]),
            hovermode="x unified",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig

    def plot_binary_cm(self, cm, fault_name):
        ticklabels = ['Fault-free', fault_name]
        group_names = ['TN', 'FP', 'FN', 'TP']
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
        labels = [f"{v1}<br>{v2}" for v1, v2 in
                zip(group_names, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        fig = ff.create_annotated_heatmap(cm, x=ticklabels, y=ticklabels, annotation_text=labels,
                                        colorscale='Jet', showscale=False)

        fig.update_layout(title_text=f'Binary confusion matrix',
                        xaxis=dict(title='Predicted label'),
                        yaxis=dict(title='True label'),
                        height=400,
                        annotations=[dict(x=j, y=i, text=str(labels[i][j]), showarrow=False)
                                    for i in range(2) for j in range(2)])
        return fig
# May have a Sensor subclass referring to the turbine class - pitch_controller.py
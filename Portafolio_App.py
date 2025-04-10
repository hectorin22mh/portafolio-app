import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import google.generativeai as genai
from math import ceil

tokenAI = "AIzaSyDjFAIJkM_2TIlJOTG_rmj7mS6f8IVWG-s"

def translate_with_gemini(text):
    try:
        genai.configure(api_key=tokenAI)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            f"Traduce y resume en español el siguiente texto de forma concisa y clara. Usa lenguaje natural y profesional. No agregues encabezados ni introducciones. Solo devuelve el texto sintetizado en español: {text}"
        )
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'candidates'):
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return text
    except Exception as e:
        return f"Error al traducir: {str(e)}"

st.set_page_config(page_title="Optimizador de Portafolios", layout="wide")
st.title("📈 Optimizador de Portafolios con Simulación Monte Carlo")

# Input de usuario
tickers_input = st.text_input("Introduce los tickers separados por comas (ej. AAPL, MSFT, TSLA):")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if tickers:
    if st.button("Generar Portafolio"):
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=5*365)

        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
                # st.write(f"Columnas descargadas para {ticker}: {df.columns.tolist()}")
                # st.write(f"Primeras filas de {ticker}:")
                # st.dataframe(df.head())
                df_close = df["Close"]
                if isinstance(df_close, pd.DataFrame):
                    df_close = df_close.iloc[:, 0]
                if isinstance(df_close, pd.Series) and not df_close.empty:
                    data[ticker] = df_close
                else:
                    st.warning(f"Datos inválidos o vacíos para {ticker}, se omitirá.")
            except Exception as e:
                st.error(f"No se pudieron obtener datos para {ticker}: {e}")

        if not data:
            st.error("No se obtuvieron datos válidos para ninguno de los tickers. Por favor intenta con otros.")
            st.stop()

        # Mostrar datos fundamentales de todas las empresas válidas
        st.markdown("<h2 style='text-align: center;'>📂 Acciones en portafolio</h2>", unsafe_allow_html=True)

        # Determinar número óptimo de columnas
        num_tickers = len(data)
        if num_tickers <= 3:
            cols = st.columns(num_tickers)
        else:
            cols = st.columns(2)

        # Mostrar la información en las columnas
        for i, (tck, serie) in enumerate(data.items()):
            with cols[i % len(cols)]:
                try:
                    info = yf.Ticker(tck).info
                    st.markdown(f"<h3 style='text-align: center; margin-top: 0.5em;'>{tck}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: justify;'><b>Nombre:</b> {info.get('longName', 'No disponible')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: justify;'><b>Sector:</b> {info.get('sector', 'No disponible')}</div>", unsafe_allow_html=True)
                    descripcion = info.get('longBusinessSummary', 'No disponible')
                    descripcion_traducida = translate_with_gemini(descripcion)
                    st.markdown(f"<div style='text-align: justify;'><b>Descripción:</b> {descripcion_traducida}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"No se pudo obtener la información fundamental de {tck}: {e}")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Mostrar CAGR y Volatilidad lado a lado
        col1, col2 = st.columns(2)

        cagr_data = []
        vol_data = []

        for ticker, series in data.items():
            try:
                dias = (series.index[-1] - series.index[0]).days
                años_totales = dias / 365
                fila = {"Ticker": ticker}

                for años in [1, 3, 5]:
                    target_date = series.index[-1] - pd.DateOffset(years=años)
                    nearest_index = series.index.get_indexer([target_date], method='nearest')[0]
                    if series.index[nearest_index] >= series.index[0]:
                        precio_inicio = series.iloc[nearest_index]
                        precio_final = series.iloc[-1]
                        cagr = (precio_final / precio_inicio) ** (1 / años) - 1
                        fila[f"{años} año(s)"] = f"{cagr:.2%}"
                    else:
                        fila[f"{años} año(s)"] = "No disponible"
                cagr_data.append(fila)
            except Exception as e:
                st.warning(f"Error al calcular CAGR para {ticker}: {e}")

        with col1:
            st.markdown("### Rendimientos Anualizados (CAGR)")
            st.markdown("Este cálculo se basa en el crecimiento compuesto del precio desde el inicio hasta el final de cada periodo. Se muestra el rendimiento anualizado para 1, 3 y 5 años (si hay datos suficientes).")
            if cagr_data:
                st.dataframe(pd.DataFrame(cagr_data))

        for ticker, series in data.items():
            try:
                daily_returns = series.pct_change().dropna()
                std_diaria = np.std(daily_returns)
                vol_anual = std_diaria * np.sqrt(252)
                vol_data.append({"Ticker": ticker, "Volatilidad Anual": f"{vol_anual:.2%}"})
            except Exception as e:
                st.warning(f"Error al calcular la volatilidad para {ticker}: {e}")

        with col2:
            st.markdown("###  Volatilidad Anualizada")
            st.markdown("La volatilidad anualizada representa el riesgo del activo, calculado como la desviación estándar de los rendimientos diarios multiplicada por la raíz cuadrada de 252 (días hábiles por año).")

            if vol_data:
                st.dataframe(pd.DataFrame(vol_data))

        # Mostrar gráfica histórica del benchmark
        if data:
            benchmark_ticker = "SPY"
            benchmark_df = yf.download(benchmark_ticker, start=start_date, end=end_date, auto_adjust=False)
            benchmark_returns = pd.Series(
                np.log(benchmark_df["Close"].values.flatten()[1:] / benchmark_df["Close"].values.flatten()[:-1]),
                index=benchmark_df["Close"].index[1:]
            )
            benchmark_cum_values = (1 + benchmark_returns).cumprod().values.flatten()
            benchmark_cum_return = pd.Series(benchmark_cum_values, index=benchmark_returns.index)

        benchmark_returns = np.log(benchmark_df["Close"] / benchmark_df["Close"].shift(1)).dropna()

        df_prices = pd.DataFrame(data)
        log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

        num_assets = len(tickers)
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weights_list = []

        risk_free_rate = 0.044  # Tasa libre de riesgo estimada de EE.UU.
        progress_bar = st.progress(0)

        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_list.append(weights)
            annual_return = np.sum(weights * log_returns.mean()) * 250
            annual_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 250, weights)))

            results[0, i] = annual_return
            results[1, i] = annual_volatility
            results[2, i] = (results[0, i] - risk_free_rate) / results[1, i]  # Sharpe Ratio ajustado
            
            progress_bar.progress((i + 1) / num_portfolios)

        progress_bar.empty()

        results_df = pd.DataFrame({
            'Return': results[0],
            'Volatility': results[1],
            'Sharpe Ratio': results[2]
        })

        # Comparar benchmark vs portafolio con mayor Sharpe Ratio
        max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
        optimal_return = results_df.loc[max_sharpe_idx, 'Return']
        optimal_volatility = results_df.loc[max_sharpe_idx, 'Volatility']
        optimal_weights = weights_list[max_sharpe_idx]
        
        # Obtener índices de los tres portafolios clave
        idx_max_sharpe = results_df['Sharpe Ratio'].idxmax()
        idx_min_vol = results_df['Volatility'].idxmin()
        idx_max_return = results_df['Return'].idxmax()
        
        # Extraer resultados
        port_keys = {
            "🔵 Máximo Sharpe Ratio": idx_max_sharpe,
            "🟢 Mínima Volatilidad": idx_min_vol,
            "🔴 Máximo Retorno": idx_max_return
        }
        
        # Mostrar pestañas con cada portafolio
        st.markdown("<h2 style='text-align: center;'>🎯 Comparativa de Portafolios Óptimos</h2>", unsafe_allow_html=True)
        tabs = st.tabs(list(port_keys.keys()))
        
        for i, (nombre, idx) in enumerate(port_keys.items()):
            with tabs[i]:
                port_return = results_df.loc[idx, 'Return']
                port_vol = results_df.loc[idx, 'Volatility']
                port_sharpe = results_df.loc[idx, 'Sharpe Ratio']
                port_weights = weights_list[idx]
                port_pesos_dict = {tickers[j]: f"{w:.2%}" for j, w in enumerate(port_weights)}
                port_pesos_df = pd.DataFrame.from_dict(port_pesos_dict, orient='index', columns=['Peso'])
        
                col1, col2 = st.columns([1.1, 1.5])
        
                with col1:
                    st.markdown(f"<h4>{nombre}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 16px;'>• <b>Rendimiento Esperado:</b> {port_return:.2%}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 16px;'>• <b>Volatilidad Esperada:</b> {port_vol:.2%}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 16px;'>• <b>Sharpe Ratio:</b> {port_sharpe:.2f}</p>", unsafe_allow_html=True)
        
                with col2:
                    st.markdown("#### Composición del Portafolio")
                    st.dataframe(port_pesos_df)
                    # Análisis y visualización del portafolio seleccionado
                    st.markdown(f"<h3 style='text-align: center;'>📊 Análisis del Portafolio - {nombre}</h3>", unsafe_allow_html=True)
                    
                    # Composición Visual
                    fig_pie = go.Figure(data=[go.Pie(labels=list(port_pesos_dict.keys()), values=[float(w.strip('%')) for w in port_pesos_dict.values()], hole=0.3)])
                    fig_pie.update_layout(title="Distribución Visual del Portafolio", height=400)
                    st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{i}")
                    
                    # Métricas adicionales
                    port_returns_series = pd.Series(log_returns @ port_weights, index=log_returns.index)
                    benchmark_aligned = benchmark_returns.loc[log_returns.index.intersection(benchmark_returns.index)]
                    
                    common_index = port_returns_series.index.intersection(benchmark_aligned.index)
                    port_aligned = port_returns_series.loc[common_index]
                    bench_aligned = benchmark_aligned.loc[common_index]
                    
                    if len(port_aligned) > 1 and len(bench_aligned) > 1:
                        common_index = port_aligned.index.intersection(bench_aligned.index)
                        port_vals = port_aligned.loc[common_index].values.flatten()
                        bench_vals = bench_aligned.loc[common_index].values.flatten()
                        if len(port_vals) > 1 and len(bench_vals) > 1 and len(port_vals) == len(bench_vals):
                            cov_matrix = np.cov(port_vals, bench_vals)
                            beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
                        else:
                            beta = np.nan
                    else:
                        beta = np.nan
                    
                    benchmark_mean = float(benchmark_aligned.mean()) * 252
                    expected_return = risk_free_rate + beta * (benchmark_mean - risk_free_rate)
                    alpha = float(port_return - expected_return)
                    
                    aligned_benchmark = benchmark_aligned.reindex(port_returns_series.index).dropna()
                    aligned_portfolio = port_returns_series.loc[aligned_benchmark.index]
                    
                    if len(aligned_portfolio) > 1:
                        tracking_diff = aligned_portfolio.values - aligned_benchmark.values
                        tracking_error = float(np.std(tracking_diff) * np.sqrt(252))
                    else:
                        tracking_error = np.nan
                    
                    correl_matrix = df_prices.pct_change().dropna().corr()
                    upper_triangle = correl_matrix.where(np.triu(np.ones(correl_matrix.shape), k=1).astype(bool))
                    correl_promedio = float(upper_triangle.stack().mean())
                    
                    metricas = pd.DataFrame({
                        "Métrica": ["Beta", "Alpha", "Tracking Error", "Correlación Promedio"],
                        "Valor": [
                            f"{beta:.2f}" if not np.isnan(beta) else "N/A",
                            f"{alpha:.2%}" if not np.isnan(alpha) else "N/A",
                            f"{tracking_error:.2%}" if not np.isnan(tracking_error) else "N/A",
                            f"{correl_promedio:.2f}"
                        ],
                        "Descripción": [
                            "Sensibilidad del portafolio frente al mercado (S&P 500).",
                            "Rendimiento extra sobre lo esperado según su riesgo (modelo CAPM).",
                            "Diferencia estándar frente al benchmark.",
                            "Relación promedio entre los activos del portafolio."
                        ]
                    })
                    
                    st.dataframe(metricas)
                    
                    # Conclusión del Análisis con IA
                    st.subheader("🧠 Conclusión del Análisis del Portafolio")
                    try:
                        prompt_analisis = f'''
  Eres un asesor financiero profesional. A partir de las siguientes métricas de un portafolio de inversión:
  
  - Beta: {'N/A' if np.isnan(beta) else f"{beta:.2f}"}
  - Alpha: {'N/A' if np.isnan(alpha) else f"{alpha:.2%}"}
  - Tracking Error: {'N/A' if np.isnan(tracking_error) else f"{tracking_error:.2%}"}
  - Correlación Promedio entre activos: {correl_promedio:.2f}
  
  Redacta un párrafo de conclusión en español explicando qué significan estas métricas en conjunto para el inversionista. Evalúa el nivel de riesgo, la diversificación y si se percibe un rendimiento superior al esperado. Usa un lenguaje claro, técnico pero accesible. No uses encabezados ni introducciones.
  '''
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        genai.configure(api_key=tokenAI)
                        response = model.generate_content(prompt_analisis)
                        analisis_texto = response.text.strip() if hasattr(response, 'text') and response.text else "No se pudo generar el análisis."
                        st.markdown(f"<div style='text-align: justify; font-size: 18px;'>{analisis_texto}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"No se pudo generar la conclusión del análisis con IA: {str(e)}")
                    
                    # Gráfico de rendimiento acumulado
                    st.subheader("📈 Rendimiento Acumulado Comparado")
                    port_daily_returns = pd.Series(log_returns @ port_weights, index=log_returns.index)
                    port_cum_returns = (1 + port_daily_returns).cumprod()
                    
                    if benchmark_cum_return is not None:
                        comparison_df = pd.DataFrame({
                            'Portafolio': port_cum_returns,
                            'S&P 500': benchmark_cum_return
                        }).dropna()
                        st.line_chart(comparison_df)
                    else:
                        st.line_chart(port_cum_returns.rename("Portafolio"))

        # st.markdown("<h2 style='text-align: center;'>🎯 Portafolio Óptimo Recomendado</h2>", unsafe_allow_html=True)
        # col1, col2, col3 = st.columns([1.2, 1, 1.5])
        #
        # with col1:
        #     st.markdown("### 📈 Comparativa con Benchmark")
        #     st.markdown(f"<h6 style='margin-top: 1em;'>Portafolio Óptimo (Mayor Sharpe Ratio):</h6>", unsafe_allow_html=True)
        #     st.markdown(f"<p style='font-size: 18px;'>• <b>Rendimiento Esperado:</b> <span style='background-color:#1c1c1c; padding: 4px 10px; border-radius: 8px; color: #6f6;'> {optimal_return:.2%}</span></p>", unsafe_allow_html=True)
        #     st.markdown(f"<p style='font-size: 18px;'>• <b>Volatilidad Esperada:</b> <span style='background-color:#1c1c1c; padding: 4px 10px; border-radius: 8px; color: #6f6;'> {optimal_volatility:.2%}</span></p>", unsafe_allow_html=True)
        #     optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility
        #     st.markdown(f"<p style='font-size: 18px;'>• <b>Sharpe Ratio:</b> <span style='background-color:#1c1c1c; padding: 4px 10px; border-radius: 8px; color: #6f6;'> {optimal_sharpe:.2f}</span></p>", unsafe_allow_html=True)
        #
        # with col2:
        #     st.markdown("### 📌 Composición del Portafolio Óptimo")
        #     pesos_dict = {tickers[i]: f"{w:.2%}" for i, w in enumerate(optimal_weights)}
        #     pesos_df = pd.DataFrame.from_dict(pesos_dict, orient='index', columns=['Peso'])
        #     st.markdown(
        #         pesos_df.style.set_properties(**{
        #             'text-align': 'center',
        #             'font-size': '18px'
        #         }).to_html(), unsafe_allow_html=True
        #     )
        #
        # with col3:
        #     st.markdown("### 🍩 Distribución Visual del Portafolio Óptimo")
        #     fig_pie = go.Figure(data=[go.Pie(labels=list(pesos_dict.keys()), values=[float(w.strip('%')) for w in pesos_dict.values()], hole=0.3)])
        #     fig_pie.update_layout(title="Composición del Portafolio Óptimo", height=400)
        #     st.plotly_chart(fig_pie, use_container_width=True)

        # Nueva sección: Análisis del Portafolio
        st.markdown("<h2 style='text-align: center;'>📊 Análisis del Portafolio</h2>", unsafe_allow_html=True)

        # Calcular métricas adicionales
        benchmark_aligned = benchmark_returns.loc[log_returns.index.intersection(benchmark_returns.index)]
        port_returns_series = pd.Series(log_returns @ optimal_weights, index=log_returns.index)

        # Beta del portafolio
        common_index = port_returns_series.index.intersection(benchmark_aligned.index)
        port_aligned = port_returns_series.loc[common_index]
        bench_aligned = benchmark_aligned.loc[common_index]

        # Verificar si ambos conjuntos de datos tienen valores suficientes
        if len(port_aligned) > 1 and len(bench_aligned) > 1:
            # Asegurar que los datos estén alineados y del mismo tamaño
            common_index = port_aligned.index.intersection(bench_aligned.index)
            port_vals = port_aligned.loc[common_index].values.flatten()
            bench_vals = bench_aligned.loc[common_index].values.flatten()

            if len(port_vals) > 1 and len(bench_vals) > 1 and len(port_vals) == len(bench_vals):
                cov_matrix = np.cov(port_vals, bench_vals)
                beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
            else:
                beta = np.nan
        else:
            beta = np.nan

        # Alpha del portafolio (modelo CAPM)
        benchmark_mean = float(benchmark_aligned.mean()) * 252
        expected_return = risk_free_rate + beta * (benchmark_mean - risk_free_rate)
        alpha = float(optimal_return - expected_return)

        # Tracking Error
        aligned_benchmark = benchmark_aligned.reindex(port_returns_series.index).dropna()
        aligned_portfolio = port_returns_series.loc[aligned_benchmark.index]
        
        if len(aligned_portfolio) > 1:
            tracking_diff = aligned_portfolio.values - aligned_benchmark.values
            tracking_error = float(np.std(tracking_diff) * np.sqrt(252))
        else:
            tracking_error = np.nan

        # Correlación promedio
        correl_matrix = df_prices.pct_change().dropna().corr()
        upper_triangle = correl_matrix.where(np.triu(np.ones(correl_matrix.shape), k=1).astype(bool))
        correl_promedio = float(upper_triangle.stack().mean())

        # Mostrar métricas
        metricas = pd.DataFrame({
            "Métrica": ["Beta", "Alpha", "Tracking Error", "Correlación Promedio"],
            "Valor": [
                f"{beta:.2f}" if not np.isnan(beta) else "N/A",
                f"{alpha:.2%}" if not np.isnan(alpha) else "N/A",
                f"{tracking_error:.2%}" if not np.isnan(tracking_error) else "N/A",
                f"{correl_promedio:.2f}"
            ],
            "Descripción": [
                "Sensibilidad del portafolio frente al mercado (S&P 500).",
                "Rendimiento extra sobre lo esperado según su riesgo (modelo CAPM).",
                "Diferencia estándar frente al benchmark.",
                "Relación promedio entre los activos del portafolio."
            ]
        })

        st.dataframe(metricas)
 
        # Conclusión con IA sobre métricas del portafolio
        st.subheader("🧠 Conclusión del Análisis del Portafolio")
 
        try:
            prompt_analisis = f"""
Eres un asesor financiero profesional. A partir de las siguientes métricas de un portafolio de inversión:

- Beta: {'N/A' if np.isnan(beta) else f"{beta:.2f}"}
- Alpha: {'N/A' if np.isnan(alpha) else f"{alpha:.2%}"}
- Tracking Error: {'N/A' if np.isnan(tracking_error) else f"{tracking_error:.2%}"}
- Correlación Promedio entre activos: {correl_promedio:.2f}

Redacta un párrafo de conclusión en español explicando qué significan estas métricas en conjunto para el inversionista. Evalúa el nivel de riesgo, la diversificación y si se percibe un rendimiento superior al esperado. Usa un lenguaje claro, técnico pero accesible. No uses encabezados ni introducciones.
"""
 
            model = genai.GenerativeModel('gemini-1.5-flash')
            genai.configure(api_key=tokenAI)
            response = model.generate_content(prompt_analisis)
            analisis_texto = response.text.strip() if hasattr(response, 'text') and response.text else "No se pudo generar el análisis."
            st.markdown(f"<div style='text-align: justify; font-size: 18px;'>{analisis_texto}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"No se pudo generar la conclusión del análisis con IA: {str(e)}")
 
        # Simular precios del portafolio usando retornos reales
        port_daily_returns = pd.Series(log_returns @ port_weights, index=log_returns.index)
        port_cum_returns = (1 + port_daily_returns).cumprod()

        st.subheader("📈 Rendimiento Acumulado Comparado")

        if benchmark_cum_return is not None:
            comparison_df = pd.DataFrame({
                'Portafolio': port_cum_returns,
                'S&P 500': benchmark_cum_return
            }).dropna()
            st.line_chart(comparison_df)
        else:
            st.line_chart(port_cum_returns.rename("Portafolio"))

        st.subheader("📊 Frontera Eficiente (Simulada)")
        fig = go.Figure(data=go.Scatter(
            x=results_df['Volatility'],
            y=results_df['Return'],
            mode='markers',
            marker=dict(
                color=results_df['Sharpe Ratio'],
                colorscale='Viridis',
                showscale=True,
                size=5,
                colorbar=dict(title="Sharpe Ratio")
            )
        ))
        fig.update_layout(
            xaxis_title='Volatilidad Esperada',
            yaxis_title='Rendimiento Esperado',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

       # Conclusión basada en IA


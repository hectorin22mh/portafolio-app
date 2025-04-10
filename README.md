# 📈 Optimizador de Portafolios con Streamlit

Esta aplicación permite construir, analizar y comparar portafolios de inversión de manera interactiva, utilizando datos reales desde Yahoo Finance.

## 🚀 Características principales

- Cálculo del portafolio óptimo usando simulación Monte Carlo
- Comparativa entre portafolio de:
  - Máximo Sharpe Ratio
  - Mínima Volatilidad
  - Máximo Retorno
- Análisis de métricas: Beta, Alpha, Tracking Error, Correlación Promedio
- Conclusiones automáticas con IA (Gemini AI)
- Visualización de la frontera eficiente
- Comparación con benchmark (S&P 500)

## 📦 Requisitos

La aplicación requiere los siguientes paquetes:

```bash
streamlit
yfinance
numpy
pandas
plotly
google-generativeai
```

Ya incluidos en el archivo `requirements.txt`.

## ▶️ ¿Cómo correr la app localmente?

1. Clona el repositorio:
```bash
git clone https://github.com/hectorin22mh/portafolio-app.git
cd portafolio-app
```

2. Instala los requisitos:
```bash
pip install -r requirements.txt
```

3. Corre la app:
```bash
streamlit run Portafolio_App.py
```

## 🌐 App en línea

Puedes usar la versión desplegada en Streamlit Cloud aquí:  
🔗 [https://hectorin22mh-portafolio-app.streamlit.app](https://hectorin22mh-portafolio-app.streamlit.app)

---

Desarrollado por [hectorin22mh](https://github.com/hectorin22mh)

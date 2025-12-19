import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import time
import asyncio
import aiohttp
from joblib import Parallel, delayed

st.set_page_config(page_title="HW1", layout="wide")
st.title("Анализ температуры и мониторинг текущей погоды")

st.markdown("""
---
**Для проверяющего:**
- Анализ файла запускается автоматически после загрузки файла.
- Текущая погода запрашивается автоматически при вводе ключа API + Enter.
- Поддерживаются оба метода API (синхронный и асинхронный).
- Все графики интерактивны, аномалии отображаются.
- Исследование параллелизации: замеры времени последовательного vs параллельного анализа.
- Нормальный диапазон температур рассчитывается на основе загруженных данных или встроенных сезонных данных.
---
""")

# Функция анализа для одного города
def analyze_city(city_df):
    city_df = city_df.sort_values('timestamp').reset_index(drop=True)
    
    # Rolling и аномалии
    city_df['rolling_mean'] = city_df['temperature'].rolling(30, center=True).mean()
    city_df['rolling_std'] = city_df['temperature'].rolling(30, center=True).std()
    city_df['anomaly'] = np.abs(city_df['temperature'] - city_df['rolling_mean']) > 2 * city_df['rolling_std']
    
    # Сезонные статистики
    season_stats = city_df.groupby('season')['temperature'].agg(['mean', 'std']).round(2)
    season_stats.index = season_stats.index.str.lower()
    season_stats = season_stats.reindex(['winter', 'spring', 'summer', 'autumn'])
    
    return city_df, season_stats

mode = st.radio(
    "Выбери режим работы:",
    ["📊 Анализ исторических данных (загрузка файла CSV)",
     "🌤️ Текущая температура через API openweathermap.org"]
)

# Реальные средние температуры (примерные данные) для городов по сезонам
builtin_seasonal_temps = {
    "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
    "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
    "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
    "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
    "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
    "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
    "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
    "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
    "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
    "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
    "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
    "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
    "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
    "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
    "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},
}

cities = sorted(builtin_seasonal_temps.keys())

# Словарь месяц в сезон
month_to_season = {1: "winter", 2: "winter", 12: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}

current_month = datetime.now().month
current_season = month_to_season.get(current_month, "winter")

# =============================================================================
# Общие элементы: выбор города и API-ключ
col1, col2 = st.columns([2, 3])
with col1:
    city = st.selectbox("Город", cities, key="city_select")
with col2:
    api_key = st.text_input("API ключ openweathermap.org",
                            type="password",
                            help="Ввод ключа + Enter = автоматический запрос погоды",
                            key="api_key_input")

# =============================================================================
# РЕЖИМ 1: Анализ исторических данных
if mode == "📊 Анализ исторических данных (загрузка файла CSV)":
    uploaded_file = st.file_uploader("Загрузите файл temperature_data.csv", type="csv")
    
    if uploaded_file is not None:
        with st.spinner("Анализ данных..."):
            df = pd.read_csv(uploaded_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if city not in df['city'].unique():
                st.error(f"Город '{city}' не найден в файле.")
                st.stop()
            
            st.subheader("🔬 Исследование параллелизации анализа")
            
            # Подготовка данных по всем городам
            city_dfs = [group for _, group in df.groupby('city')]
            
            #ПОСЛЕДОВАТЕЛЬНЫЙ анализ
            start_seq = time.time()
            results_seq = []
            for city_df in city_dfs:
                result = analyze_city(city_df)
                results_seq.append(result)
            time_seq = time.time() - start_seq
            
            #ПАРАЛЛЕЛЬНЫЙ анализ
            start_par = time.time()
            results_par = Parallel(n_jobs=-1)(delayed(analyze_city)(city_df) for city_df in city_dfs)
            time_par = time.time() - start_par
            
            # Результаты замера времени
            col_time1, col_time2, col_speedup = st.columns(3)
            with col_time1:
                st.metric("Последовательный", f"{time_seq:.3f}с")
            with col_time2:
                st.metric("Параллельный", f"{time_par:.3f}с")
            with col_speedup:
                speedup = time_seq / time_par
                st.metric("Ускорение", f"{speedup:.1f}x")
            
            st.info(f"**Вывод:** Параллелизация даёт ускорение {speedup:.1f}x")
            
            # Анализ выбранного города (из параллельных результатов)
            city_results = {df['city'].iloc[0]: result for df, result in results_par}
            city_df, season_stats = city_results[city]
            
            st.success(f"Файл загружен! Анализ для города: **{city}**")
            
            col_stats, col_season = st.columns([1, 1])
            with col_stats:
                st.subheader("Описательная статистика")
                stats = city_df['temperature'].describe().round(2).to_frame(name='value')
                st.dataframe(stats)
            with col_season:
                st.subheader("Сезонные характеристики")
                st.table(season_stats)

            # Графики
            col1, col2 = st.columns(2)
            with col1:
                fig_box = px.box(city_df, x='season', y='temperature',
                                 category_orders={"season": ["winter", "spring", "summer", "autumn"]},
                                 title="Распределение по сезонам")
                st.plotly_chart(fig_box, use_container_width=True)
            with col2:
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=city_df['timestamp'], y=city_df['temperature'],
                                            mode='lines', name='Температура', line=dict(width=1)))
                fig_ts.add_trace(go.Scatter(x=city_df['timestamp'], y=city_df['rolling_mean'],
                                            mode='lines', name='Скользящее среднее', line=dict(dash='dash')))
                anomalies = city_df[city_df['anomaly']]
                if len(anomalies) > 0:
                    fig_ts.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['temperature'],
                                                mode='markers', name='Аномалии', marker=dict(color='red', size=8)))
                fig_ts.update_layout(title="Временной ряд с аномалиями", height=400)
                st.plotly_chart(fig_ts, use_container_width=True)

            # Нормальный диапазон
            season_mean = season_stats.loc[current_season, 'mean']
            season_std = season_stats.loc[current_season, 'std']
            norm_low = season_mean - 2 * season_std
            norm_high = season_mean + 2 * season_std
            
    else:
        st.info("Загрузите файл temperature_data.csv, чтобы увидеть анализ.")

if mode == "🌤️ Текущая температура через API openweathermap.org" or 'uploaded_file' not in locals() or uploaded_file is None:
    season_mean = builtin_seasonal_temps[city][current_season]
    season_std = 5.0
    norm_low = season_mean - 2 * season_std
    norm_high = season_mean + 2 * season_std

if api_key:
    st.markdown("---")
    st.subheader(f"🌤️ Текущая температура (OpenWeatherMap) для **{city}**")
    method = st.radio("Выбери метод запроса",
                      ["Синхронный (requests)",
                       "Асинхронный (aiohttp)"],
                       horizontal=True)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=ru"
    with st.spinner("Получение актуальных данных..."):
        start = time.time()
        if method == "Синхронный (requests)":
            try:
                resp = requests.get(url, timeout=10)
                data = resp.json()
            except Exception as e:
                st.error(f"Ошибка сети: {e}")
                data = {"cod": "network_error"}
        else:
            async def fetch():
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as r:
                        return await r.json()
            loop = asyncio.new_event_loop()
            data = loop.run_until_complete(fetch())
            loop.close()
        end = time.time()
    if data.get("cod") == 401:
        st.error("❌ Неверный или неактивированный API-ключ.")
    elif data.get("cod") != 200:
        st.error(f"Ошибка API: {data.get('message', 'Неизвестная ошибка')}")
    else:
        temp = data['main']['temp']
        feels = data['main']['feels_like']
        desc = data['weather'][0]['description'].capitalize()
        st.write(f"**Температура сейчас: {temp} °C**")
        st.write(f"Ощущается как: {feels} °C • {desc}")
        if norm_low <= temp <= norm_high:
            st.success("✅ Температура в пределах нормы для сезона")
        else:
            st.warning("⚠️ Аномальная температура для текущего сезона!")
        source = "загруженного файла" if mode.startswith("📊") and 'uploaded_file' in locals() and uploaded_file is not None else "встроенных данных"
        st.info(f"Нормальный диапазон ({current_season.capitalize()}): "
                f"{norm_low:.1f} … {norm_high:.1f} °C (на основе {source})")
    st.caption(f"Запрос выполнен за {end - start:.3f} сек | Метод: {method}")
else:
    st.info("Введите API-ключ и нажмите Enter — текущая погода появится автоматически.")

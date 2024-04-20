import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
from pmdarima import auto_arima
import sys

# Определяем путь к файлу, в который хотим записать вывод
output_file_path = 'output.csv'

# Определяем путь к файлу, в который хотим записать вывод
original_stdout = sys.stdout

try:
    # Открываем файл для записи вывода
    with open(output_file_path, 'w') as f:
        # Перенаправляем вывод в файл
        sys.stdout = f

        # Загрузим данные из файла Excel
        file_path = "rub_dol_22-24.xlsx"
        df = pd.read_excel(file_path)

        # Преобразуем столбец 'data' в формат datetime
        df['data'] = pd.to_datetime(df['data'])

        # Удаление выбросов с использованием Isolation Forest
        isolation_forest = IsolationForest(contamination=0.05)
        outlier_preds = isolation_forest.fit_predict(df[['curs']])

        # Добавим столбец с предсказанными значениями выбросов
        df['outlier'] = outlier_preds

        # Фильтруем данные за 2022-2024 годы
        df_filtered = df[(df['data'].dt.year >= 2022) & (df['data'].dt.year <= 2024)]

        # Создаем новый столбец с месяцем
        df_filtered['month'] = df_filtered['data'].dt.month

        # Группируем данные по месяцам и рассчитываем средний курс для каждого месяца
        monthly_avg_curs = df_filtered.groupby('month')['curs'].mean()

        # Устанавливаем столбец 'data' в качестве индекса
        df_filtered.set_index('data', inplace=True)

        # Создаем и обучаем модель SARIMA
        model = SARIMAX(df_filtered['curs'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        result = model.fit(maxiter=100)

        # Построим график курса доллара по времени
        plt.figure(figsize=(10, 6))
        plt.plot(df_filtered.index, df_filtered['curs'], color='blue', linestyle='-')
        plt.title('Изменение курса доллара США по времени (2022-2024)')
        plt.xlabel('Дата')
        plt.ylabel('Курс доллара')
        plt.grid(True)
        plt.show()

        # Строим график изменения среднего курса по месяцам
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=monthly_avg_curs.index, y=monthly_avg_curs.values)
        plt.title('Средний курс доллара США по месяцам (2022-2024)')
        plt.xlabel('Месяц')
        plt.ylabel('Средний курс')
        plt.xticks(range(1, 13))
        plt.grid(True)
        plt.show()

        # Визуализируем исходные данные и выбросы
        plt.figure(figsize=(10, 6))
        plt.scatter(df['data'], df['curs'], c=df['outlier'], cmap='viridis', label='Данные')
        plt.xlabel('Дата')
        plt.ylabel('Курс доллара')
        plt.title('Обнаружение выбросов с использованием Isolation Forest')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Прогнозируем будущие значения
        forecast = result.forecast(steps=12)  # Прогноз на 12 месяцев вперед

        # Выводим прогнозные значения
        print("Прогнозные значения курса доллара на следующие 12 месяцев:")
        print(forecast)

        # Сохраняем прогнозные значения в CSV файл
        forecast_df = pd.DataFrame({'Date': pd.date_range(start='2025-01-01', periods=12, freq='M'), 'Forecast': forecast})
        forecast_df.to_csv('dollar_forecast_2025.csv', index=False)

        # Фактическое значение курса доллара на 17.04.2024
        actual_value = [94.0742]  # Преобразуем в массив

        # Рассчитываем среднеквадратичную ошибку (MSE)
        mse = mean_squared_error([actual_value] * len(forecast), forecast)

        # Рассчитываем среднюю абсолютную ошибку (MAE)
        mae = mean_absolute_error([actual_value] * len(forecast), forecast)

        # Выводим метрики оценки точности
        print("Среднеквадратичная ошибка (MSE):", mse)
        print("Средняя абсолютная ошибка (MAE):", mae)

        # Настройка модели SARIMA с помощью автоматического подбора параметров
        model = auto_arima(df['curs'], seasonal=True, m=12, trace=True)
        print(model.summary())

finally:
    # Восстанавливаем оригинальный вывод
    sys.stdout = original_stdout

    import joblib

    # Сериализуем модель и сохраняем ее в файл
    joblib.dump(model, 'sarima_model.pkl')


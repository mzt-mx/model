import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Загрузим данные из файла Excel
file_path = "dollar_rub2008_2024.xlsx"
df = pd.read_excel(file_path)

# Преобразуем столбец 'data' в формат datetime
df['data'] = pd.to_datetime(df['data'])

# Создаем новый столбец с месяцем
df['month'] = df['data'].dt.month

# Группируем данные по месяцам и рассчитываем средний курс для каждого месяца
monthly_avg_curs = df.groupby('month')['curs'].mean()

# Устанавливаем столбец 'data' в качестве индекса
df.set_index('data', inplace=True)

# Построим график курса доллара по времени
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['curs'], color='blue', linestyle='-')
plt.title('Изменение курса доллара США по времени')
plt.xlabel('Дата')
plt.ylabel('Курс доллара')
plt.grid(True)
plt.show()

# Строим график изменения среднего курса по месяцам
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_avg_curs.index, y=monthly_avg_curs.values)
plt.title('Средний курс доллара США по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Средний курс')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

# Удаляем столбец 'cdx'
df_numeric = df.drop(columns=['cdx'])

# Рассчитываем корреляцию между числовыми столбцами
correlation_matrix = df_numeric.corr()

# Выводим матрицу корреляции
print(correlation_matrix)

# Рассчитываем стандартное отклонение курса доллара
std_deviation = df['curs'].std()

# Выводим стандартное отклонение
print("Стандартное отклонение курса доллара:", std_deviation)

# Создаем и обучаем модель SARIMA
model = SARIMAX(df['curs'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result = model.fit()

# Прогнозируем будущие значения
forecast = result.forecast(steps=12)  # Прогноз на 12 месяцев вперед, например

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








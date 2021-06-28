from datetime import datetime
from dateutil.relativedelta import relativedelta
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('example_retail_sales.csv')

last_data_str = df.iloc[len(df) - 1]['ds']
last_date = datetime.strptime(last_data_str, '%Y-%m-%d')

pivot = 290

while True:
    m = Prophet(seasonality_mode='multiplicative')
    forecast = m.fit(df)
    future = m.make_future_dataframe(periods=20, freq='MS')

    future = m.predict(future)

    plt.plot(future['ds'][pivot:], future['trend'][pivot:])
    plt.plot(future['ds'][pivot:], future['yhat'][pivot:])
    plt.plot(future['ds'][pivot:], future['yhat_lower'][pivot:])
    plt.plot(future['ds'][pivot:], future['yhat_upper'][pivot:])

    y_values = []
    x_values = []

    for i, row in df[pivot:].iterrows():
        y_values.append(future['ds'][i])
        x_values.append(row['y'])

    plt.plot(y_values, x_values)

    plt.show()

    next_value = int(input('Input next value : '))
    next_date = last_date + relativedelta(months=1)
    next_date_str = next_date.strftime('%Y-%m-%d')
    last_date = next_date
    new_row = pd.DataFrame({
        'ds': next_date_str,
        'y': next_value
    }, index=[0])
    print(new_row)
    df = df.append(new_row, ignore_index=True)
    print(next_date_str)
    pivot = pivot + 1
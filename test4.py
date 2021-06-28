import csv
import numpy as np
import matplotlib.pyplot as plt

optimal_xaxis = []
optimal_yaxis = []

with open("sample_data.csv", encoding='utf-8-sig') as data:
    reader = csv.DictReader(data)
    prices = []
    quantities = []
    for row in reader:
        price = int(row['price'])
        sale_qty = int(row['sale_qty'])
        prices.append(price)
        quantities.append(sale_qty)
        plt.scatter(price, sale_qty)

    x = np.array(prices)
    y = np.array(quantities)

    fit = np.polyfit(x, y, 2)
    print(fit)

    for price in range(10000, 100000, 1000):
        optimal_xaxis.append(price)
        optimal_yaxis.append(fit[0] * (price ** 2) + fit[1] * price + fit[2])

    plt.plot(optimal_xaxis, optimal_yaxis)
    plt.show()
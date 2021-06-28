import csv
import matplotlib.pyplot as plt

sale_data = []
with open("sample_data.csv", encoding='utf-8-sig') as data:
    reader = csv.DictReader(data)
    for row in reader:
        price = int(row['price'])
        sale_qty = int(row['sale_qty'])
        sale_data.append({
            'price': price,
            'qty': sale_qty,
        })
        plt.scatter(price, sale_qty)

    minimum_diff = -1
    optimal_weight = 0
    for divider in range(-100, 101):
        if divider == 0:
            continue
        for dividend in range(1, 101):
            weight = dividend / (divider * 1000000)

            sum_diff = 0
            for data in sale_data:
                estimate = data.get('price') * weight
                diff = estimate - data.get('qty')
                sum_diff += diff
            if minimum_diff == -1 or sum_diff < minimum_diff:
                optimal_weight = weight
                minimum_diff = sum_diff

    x_axis = []
    y_axis = []
    for price in range(10000, 100000, 1000):
        estimate_qty = optimal_weight * price
        x_axis.append(price)
        y_axis.append(estimate_qty)
    plt.plot(x_axis,y_axis)
    plt.show()
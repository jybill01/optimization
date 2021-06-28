from sklearn.datasets import load_boston
import pandas as pd
import statsmodels.formula.api as sm

boston_data = load_boston()

boston = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
boston['target'] = boston_data.target

train = boston.sample(frac=0.8, random_state=200)
test = boston.drop(train.index)

result = sm.glm(formula='target ~ CRIM + INDUS + ZN + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=train).fit()

#scatter_matrix(boston.drop(columns=['B']))

print(result.summary())

sum_difference = 0
for i, row in test.iterrows():
    params = result.params
    r_estimate = row['PTRATIO'] * params['PTRATIO'] + row['INDUS'] * params['INDUS'] + row['NOX'] * params['NOX'] + \
        row['B'] * params['B'] +row['CHAS'] * params['CHAS'] + row['RAD'] * params['RAD'] + \
        row['TAX'] * params['TAX'] + row['ZN'] * params['ZN'] + row['DIS'] * params['DIS'] + \
        row['CRIM'] * params['CRIM'] + row['RM'] * params['RM'] + row['LSTAT'] * params['LSTAT'] + \
        params['Intercept']
    sum_difference += abs(r_estimate - row['target'])

print(sum_difference)
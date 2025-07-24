from sklearn.metrics import mean_gamma_deviance, mean_absolute_percentage_error

target = [0.00002, 0.00003, 0.0000482, 0.00008264, 0.000014571]
pred = [0.00002847, 0.00005, 0.00002, 0.00006, 0.000082]
err1 = mean_gamma_deviance(target, pred)
err2 = mean_absolute_percentage_error(target, pred)
print(err1, err2)

target = [2,7,4,8,6]
pred = [2,6,8,5,2]
err1 = mean_gamma_deviance(target, pred)
err2 = mean_absolute_percentage_error(target, pred)
print(err1, err2)

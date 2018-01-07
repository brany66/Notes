#!/usr/bin/python
# -*- coding: utf-8 -*-
# create by YWJ, 2017.9.23

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
realNum = lambda x : x + x ** 0.1

x_train = np.linspace(0, 15, 100)
y_train = list(map(realNum, x_train))
y_noise = 2 * np.random.normal(size=x_train.size)
y_train = y_train + y_noise

x_valid = np.linspace(0, 15, 50)
y_valid = list(map(realNum, x_valid))
y_noise = 2 * np.random.normal(size=x_valid.size)
y_valid = y_valid + y_noise

prop = np.polyfit(x_train, y_train, 1)
prop_ = np.poly1d(prop)
overf = np.polyfit(x_train, y_train, 15)
overf_ = np.poly1d(overf)

# print(type(prop))
# print(type(prop_))
# print(type(overf))l
# print(type(overf_))

_ = plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
prop_e = np.mean((y_train - np.polyval(prop, x_train)) ** 2)
overf_e = np.mean((y_train - np.polyval(overf, x_train)) ** 2)
xp = np.linspace(-2, 17, 200)
plt.plot(x_train, y_train, '.')
plt.plot(xp, prop_(xp), '-', label='proper, err: %.3f' % (prop_e))
plt.plot(xp, overf_(xp), '--', label='overfit, err: %.3f' % (overf_e))
plt.ylim(-5, 20)
plt.legend()
plt.title('train set')

plt.subplot(1, 2, 2)
prop_e = np.mean((y_valid - np.polyval(prop,  x_valid)) ** 2)
overf_e = np.mean((y_valid - np.polyval(overf, x_valid)) ** 2)
xp = np.linspace(-2, 17, 200)
plt.plot(x_valid, y_valid, '.')
plt.plot(xp, prop_(xp), '-', label='proper, err: %.3f' % (prop_e))
plt.plot(xp, overf_(xp), '--', label='overfit, err: %.3f' % (overf_e))
plt.ylim(-5, 20)
plt.legend()
plt.title('validation set')


print("YWJ")
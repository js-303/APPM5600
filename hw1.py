import numpy as np
from matplotlib import pyplot as plt

p_0 = lambda x: (x-2)**9
p_1 = lambda x: ((x**9)-(18*x**8)+(144*x**7)-(672*x**6)+(2016*x**5)-(4032*x**4)+(5376*x**3)-(4608*x**2)+(2304*x)-512)
p_2 = lambda x: (((((((((x-18)*x+144)*x-672)*x+2016)*x-4032)*x+5376)*x-4608)*x+2304)*x-512)

x = np.arange(1.920, 2.080, 0.001)

y_0 = p_0(x)
y_1 = p_1(x)
y_2 = p_2(x)

plt.plot(x, y_1)
plt.plot(x, y_0)
plt.plot(x, y_2)
plt.show()
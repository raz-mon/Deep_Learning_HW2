import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)                           # Constant seed -> Constant results.
rand_arrs = np.random.rand(10000, 50)       # Generate random arrays.
for arr in rand_arrs:
    i = np.random.randint(20, 30)
    for ind in range(i-5, i+6):
        arr[ind] = arr[ind] * 0.1

plt.figure()
xs = np.arange(0, 50, 1)
for i in range(3):
    ind = np.random.randint(0, 10000)
    plt.plot(xs, rand_arrs[ind], label=f'{ind}')

plt.xlabel('t')
plt.ylabel('value')
plt.title('3 samples from synthetic data-set')
plt.legend()
# plt.savefig('samples_from_synthetic_data.png')
plt.show()













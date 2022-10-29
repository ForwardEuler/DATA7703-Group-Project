import timeit

import numpy as np
from f_ndarray import *
from lib_bc import optimal_cutoff, optimal_cutoff_np

SETUP_CODE = '''
import numpy as np
from rule_threshold import get_optimal_cutoff, optimal_cutoff, optimal_cutoff_np
'''

a = np.random.rand(400000)
b = np.random.rand(400000).round()
d = optimal_cutoff(b, a)
e = optimal_cutoff_np(b, a)

test1 = '''a = np.random.rand(400000)
b = np.random.rand(400000).round()
c = get_optimal_cutoff(b, a)'''
test2 = '''a = np.random.rand(400000)
b = np.random.rand(400000).round()
d = optimal_cutoff(b, a)'''
test3 = '''a = np.random.rand(400000)
b = np.random.rand(400000).round()
d = optimal_cutoff_np(b, a)'''
times = timeit.repeat(setup=SETUP_CODE, stmt=test1, repeat=1, number=4)
print("time = " + str(times))
times = timeit.repeat(setup=SETUP_CODE, stmt=test2, repeat=1, number=4)
print("time = " + str(times))
times = timeit.repeat(setup=SETUP_CODE, stmt=test3, repeat=1, number=4)
print("time = " + str(times))



# data = mload('train_fe.npz')
# my_x = data.drop(['isDefault'], axis=1).to_numpy()
# my_y = data['isDefault'].to_numpy(dtype=np.int8)
# d_test = mload('test_fe.npz')
# test_x = d_test.drop(['isDefault'], axis=1).to_numpy()
# test_y = d_test['isDefault'].to_numpy(dtype=np.int8)
#
# m = xgboost()
# m.train(my_x, my_y)
# m.evaluate(test_y, m.predict(test_x))
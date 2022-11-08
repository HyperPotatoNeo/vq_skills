import itertools
import numpy as np

idx = [[x for x in range(3)]]*2

iterlist = itertools.product(*idx)
idx_list = []
for i in iterlist:
	idx_list.append(i)
print(np.array(idx_list))
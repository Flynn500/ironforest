import array, ironforest as irn
arr = irn.ndutils.asarray(array.array('d', [1.0, 2.0, 3.0]))  # buffer protocol in

mv = memoryview(arr)
assert mv.format == 'd'

import numpy as np
back = np.asarray(arr)

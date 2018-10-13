import numpy as np


res = np.load("result.npy")

test = res[:4,:]

res = res[res[:,2].argsort()]

np.savetxt("foo.csv", res, delimiter=",")
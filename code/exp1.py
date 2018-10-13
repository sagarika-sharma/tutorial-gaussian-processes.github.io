from __future__ import division
import numpy as np
import matplotlib.pyplot as pl





def kernel(a, b, k=0.1):
	a_2 = np.sum(a**2,1)
	b_2 = np.sum(b**2,1)
	ab = np.dot(a,b.T)
	a__b = a_2.reshape(-1,1) + b_2 - 2*ab
	return np.exp(-.5 * a__b * (1/k))

def flat(x):
	inst = np.sin(0.9*x)
	return inst.flatten()



X = np.random.uniform(-5, 5, size=(10,1))
y = flat(X) + 0.00005*np.random.randn(10)

K = kernel(X, X)
nse = 0.00005*np.eye(10)
L = np.linalg.cholesky(K + nse)


xt = np.linspace(-5, 5, 50).reshape(-1,1)

kt = kernel(X, xt)

Lk = np.linalg.solve(L, kt)
llt = np.linalg.solve(L, y)
mu = np.dot(Lk.T, llt)

K_t = kernel(xt, xt)
s2 = np.diag(K_t) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)


pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(xt, flat(xt), 'b-')
pl.gca().fill_between(xt.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(xt, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])


L = np.linalg.cholesky(K_t + 1e-6*np.eye(50))
f_prior = np.dot(L, np.random.normal(size=(50,10)))
pl.figure(2)
pl.clf()
pl.plot(xt, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])
pl.savefig('prior.png', bbox_inches='tight')


L = np.linalg.cholesky(K_t + 1e-6*np.eye(50) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(50,10)))
pl.figure(3)
pl.clf()
pl.plot(xt, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

pl.show()

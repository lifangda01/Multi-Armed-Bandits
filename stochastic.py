from pylab import *
from scipy import stats

K = 10
N = 10000
# Each arm is a Beta distribution
mu = linspace(0.1, 0.9, K)
alpha = ones(N)
beta = (1 - mu) / mu

def pull_arm(i):
	return stats.beta(alpha[i], beta[i]).rvs()

def epsilon_greedy(epsilon):
	mu_h = zeros(K) 	# Empirical mean
	n = zeros(K)		# Number of times an arm is pulled
	G = zeros(N)		# Gain at round i
	II = zeros(N)		# Choice of arm at round i
	for i in xrange(N):
		I = argmax(mu_h)
		# Explore a new arm with probability epsilon
		if rand() < epsilon:
			a = arange(K)
			a[I], a[-1] = a[-1], a[I]
			I = a[:-1][randint(0, K-1)]
		g = pull_arm(I)
		G[i] = g
		II[i] = I
		# Update empirical mean
		mu_h[I] = (mu_h[I] * n[I] + g) / (n[I] + 1)
		n[I] += 1
	return G, II

def UBC1():
	pass

def thompson_sampling():
	pass

def main():
	figure()
	title('Average Gain Overtime')
	for e in [0.01, 0.1, 0.5]:
		G, II = epsilon_greedy(e)
		plot(cumsum(G) / arange(1, N+1), label='e = %.3f'%(e))
	legend()
	show()

if __name__ == '__main__':
	main()
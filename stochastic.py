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

def UCB1():
	mu_h = zeros(K) 	# Empirical mean
	ucb = ones(K)		# Upper Confidence Bounds
	n = zeros(K)		# Number of times an arm is pulled
	G = zeros(N)		# Gain at round i
	II = zeros(N)		# Choice of arm at round i
	for i in xrange(N):
		# Pull the arm with the highest bound
		I = argmax(ucb)
		g = pull_arm(I)
		G[i] = g
		II[i] = I
		# Update the bound
		n[I] += 1
		mu_h[I] = (mu_h[I] * n[I] + g) / (n[I] + 1)
		ucb[I] = mu_h[I] + sqrt((2*log(i+1)) / n[I])
	return G, II

def thompson_sampling():
	mu_h = zeros(K) 	# Empirical mean
	ucb = ones(K)		# Upper Confidence Bounds
	n = zeros(K)		# Number of times an arm is pulled
	S = zeros(K)		# Number of success
	F = zeros(K)		# Number of failure
	G = zeros(N)		# Gain at round i
	II = zeros(N)		# Choice of arm at round i
	for i in xrange(N):
		theta = zeros(K)
		for j in xrange(K):
			theta[j] = stats.beta(S[j]+1, F[j]+1).rvs()
		I = argmax(theta)
		g = pull_arm(I)
		G[i] = g
		II[i] = I
		# Update the posterior using a Benourlli trial
		if rand() < g:
			S[I] += 1
		else:
			F[I] += 1
	return G, II

def main():
	figure()
	title('Average Gain Overtime')
	# e-greedy
	for e in [0.01, 0.1]:
		G, II = epsilon_greedy(e)
		# Could also add plot of % of optimal actions
		plot(cumsum(G) / arange(1, N+1), label='EG w/ e = %.3f'%(e))
	# UCB
	G, II = UCB1()
	plot(cumsum(G) / arange(1, N+1), label='UCB1')
	# Thompson sampling
	G, II = thompson_sampling()
	plot(cumsum(G) / arange(1, N+1), label='TS')
	legend()
	show()

if __name__ == '__main__':
	main()
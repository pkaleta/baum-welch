from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpmath import mpf

from gen import gen_seq


HMMParams = namedtuple(
    'Params',
    ['transition_prob', 'emission_prob', 'initial_prob']
)


def forward(hmm_params, X, Y, Yt):
    A, B, pi = hmm_params
    n = len(Yt)
    m = len(X)
    alpha = np.zeros((n, m), dtype=mpf)

    for t in xrange(n):
        for i in xrange(m):
            if t == 0:
                alpha[t, i] = mpf(pi[i] * B[i, Yt[t]])
            else:
                tmp = mpf(0.0)
                for j in xrange(m):
                    tmp += alpha[t - 1, j] * A[j, i]
                alpha[t, i] = B[i, Yt[t]] * tmp

    return alpha


def backward(hmm_params, X, Y, Yt):
    A, B, pi = hmm_params
    n = len(Yt)
    m = len(X)
    beta = np.zeros((n, m), dtype=mpf)

    for t in xrange(n - 1, -1, -1):
        for i in xrange(m):
            if t == n - 1:
                beta[t, i] = mpf(1.0)
            else:
                for j in xrange(m):
                    beta[t, i] += beta[t + 1, j] * A[i, j] * B[j, Yt[t + 1]]

    return beta


def baum_welch(hmm_params, X, Y, obs, iterations=1):
    A, B, pi = hmm_params

    # TODO: change this so that it checks the difference between
    # current and previous run rather than running for a fixed number
    # of iterations.
    iteration = 0
    while iteration < iterations:
        iteration += 1

        for Yt in obs:
            n = len(Yt)
            m = len(X)
            Yt = [Y.index(yt) for yt in Yt]

            alpha = forward(hmm_params, X, Y, Yt)
            beta = backward(hmm_params, X, Y, Yt)

            gamma = np.zeros((n, m), dtype=mpf)
            gamma_sums = np.zeros(n, dtype=mpf)
            for t in xrange(n):
                for i in xrange(m):
                    gamma[t, i] = mpf(alpha[t, i] * beta[t, i])
                    gamma_sums[t] += mpf(gamma[t, i])

                #import ipdb; ipdb.set_trace()
                #print gamma_sums[t]
                gamma[t] /= gamma_sums[t]

            xsi = np.zeros((n, m, m), dtype=mpf)
            for t in xrange(n - 1):
                for i in xrange(m):
                    for j in xrange(m):
                        xsi[t, i, j] = (
                            (alpha[t, i] * A[i, j] *
                             beta[t + 1, j] * B[j, Yt[t + 1]])
                        )
                        xsi[t, i, j] /= gamma_sums[t]  # nopep8
            # Update
            pi = gamma[0]

            A = np.zeros((m, m), dtype=mpf)
            for i in xrange(m):
                den = 0.0
                for t in xrange(n - 1):
                    den += gamma[t, i]
                for j in xrange(m):
                    num = 0.0
                    for t in xrange(n - 1):
                        num += xsi[t, i, j]
                    if den == 0.0:
                        A[i, j] = 1.0
                    else:
                        A[i, j] = num / den

            B = np.zeros((m, len(Y)), dtype=mpf)
            for i in xrange(len(Y)):
                for j in xrange(m):
                    tmp = 0.0
                    gamma_sum = 0.0
                    for t in xrange(n):
                        gamma_sum += gamma[t, j]
                        if Yt[t] == i:
                            tmp += gamma[t, j]
                    B[j, i] = tmp / gamma_sum

            hmm_params = HMMParams(A, B, pi)

    return hmm_params


def forward_backward(hmm_params, X, Y, sequence):
    n = len(sequence)
    m = len(X)
    seq_indices = [Y.index(yt) for yt in sequence]
    alpha = forward(hmm_params, X, Y, seq_indices)
    beta = backward(hmm_params, X, Y, seq_indices)
    prob = np.zeros((n, m), dtype=mpf)
    for t in xrange(n):
        for i in xrange(m):
            prob[t, i] = alpha[t, i] * beta[t, i]
    return prob

# A = np.matrix([[0.5, 0.5], [0.3, 0.7]])
# B = np.matrix([[0.3, 0.7], [0.8, 0.2]])
# pi = np.array([0.2, 0.8])

#if __name__ == '__main__':
transition_prob = np.matrix([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1 - 0.0002, 0.0002, 0.0],
    [0.0, 0.0, 1 - 0.0002, 0.0002],
    [0.0, 0.0, 0.0, 0.0]], dtype=mpf)

emission_prob = np.matrix([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.96, 0.036, 0.004, 0.0],
    [0.0, 0.96, 0.004, 0.036, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]], dtype=mpf)

initial_prob = np.array([1.0, 0.0, 0.0, 0.0], dtype=mpf)

with open('seq.txt') as fp:
    sequences = [line.strip() for line in fp.readlines()]

hidden_states = ['S0', 'S1', 'S2', 'S3']
symbols = ['S', 'N', 'L', 'R', 'E']

hmm_params = baum_welch(
    HMMParams(
        transition_prob=transition_prob,
        emission_prob=emission_prob,
        initial_prob=initial_prob,
    ),
    hidden_states,
    symbols,
    sequences)
seq = gen_seq()
prob = forward_backward(hmm_params, hidden_states, symbols, seq)

matplotlib.rc('xtick', labelsize=5)
plt.xticks(range(0, len(seq)), seq)
plt.plot(prob[0:len(seq), 1:3])
plt.show()

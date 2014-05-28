import numpy as np
from collections import namedtuple


Params = namedtuple(
    'Params',
    ['transition_prob', 'emission_prob', 'initial_prob']
)


def forward(A, B, pi, X, Y, Yt):
    n = len(Yt)
    m = len(X)
    alpha = np.zeros((n, m))
    #print '!!!!', pi, B

    for t in xrange(n):
        for i in xrange(m):
            if t == 0:
                alpha[t, i] = pi[i] * B[i, Yt[t]]
            else:
                tmp = 0.0
                for j in xrange(m):
                    tmp += alpha[t - 1, j] * A[j, i]
                #print '******', t, i, B[i, Yt[t]] * tmp
                alpha[t, i] = B[i, Yt[t]] * tmp

    return alpha


def backward(A, B, pi, X, Y, Yt):
    n = len(Yt)
    m = len(X)
    beta = np.zeros((n, m))

    for t in xrange(n - 1, -1, -1):
        for i in xrange(m):
            if t == n - 1:
                beta[t, i] = 1.0
            else:
                for j in xrange(m):
                    beta[t, i] += beta[t + 1, j] * A[i, j] * B[j, Yt[t + 1]]

    return beta


def baum_welch(A, B, pi, X, Y, obs):
    iteration = 0
    while iteration < 100:
        iteration += 1

        for Yt in obs:
            n = len(Yt)
            m = len(X)
            Yt = [Y.index(yt) for yt in Yt]
            #print '####', Yt

            alpha = forward(A, B, pi, X, Y, Yt)
            beta = backward(A, B, pi, X, Y, Yt)

            # print 'alpha', alpha
            # print 'beta', beta

            gamma = np.zeros((n, m))
            gamma_sums = np.zeros(n)
            for t in xrange(n):
                for i in xrange(m):
                    gamma[t, i] = alpha[t, i] * beta[t, i]
                    gamma_sums[t] += gamma[t, i]

                gamma[t] /= gamma_sums[t]

            #print 'gamma', gamma

            xsi = np.zeros((n, m, m))
            for t in xrange(n - 1):
                for i in xrange(m):
                    for j in xrange(m):
                        xsi[t, i, j] = (
                            (alpha[t, i] * A[i, j] *
                             beta[t + 1, j] * B[j, Yt[t + 1]])
                        )
                        xsi[t, i, j] /= gamma_sums[t]  # nopep8
            #print 'xsi', xsi

            # Update
            pi = gamma[0]

            #print 'gamma', gamma

            A = np.zeros((m, m))
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

            #print A

            B = np.zeros((m, len(Y)))
            for i in xrange(len(Y)):
                for j in xrange(m):
                    tmp = 0.0
                    gamma_sum = 0.0
                    for t in xrange(n):
                        gamma_sum += gamma[t, j]
                        #print '*****', Yt[t], i
                        if Yt[t] == i:
                            tmp += gamma[t, j]
                    B[j, i] = tmp / gamma_sum

            #print 'B:', B

    return Params(A, B, pi)


# A = np.matrix([[0.5, 0.5], [0.3, 0.7]])
# B = np.matrix([[0.3, 0.7], [0.8, 0.2]])
# pi = np.array([0.2, 0.8])

#if __name__ == '__main__':
transition_prob = np.matrix([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1-0.0001, 0.0001, 0.0],
    [0.0, 0.0, 1-0.0001, 0.0001],
    [0.0, 0.0, 0.0, 0.0]])

emission_prob = np.matrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.95, 0.05, 0.0],
    [0.0, 0.05, 0.95, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

initial_prob = np.array([1.0, 0.0, 0.0, 0.0])

with open('seq.txt') as fp:
    sequences = [line.strip() for line in fp.readlines()]

hidden_states = ['start', 'before', 'after', 'end']
symbols = ['S', 'L', 'R', 'E']

print baum_welch(
    transition_prob,
    emission_prob,
    initial_prob,
    hidden_states,
    symbols,
    sequences)

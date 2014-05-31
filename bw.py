from collections import namedtuple
from mpmath import mpf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys


HMM_PARAM_NAMES = ['transition_prob', 'emission_prob', 'initial_prob']
HMMParams = namedtuple('Params', HMM_PARAM_NAMES)

INITIAL_TRANSITION_PROB = np.array([
    map(mpf, [0.0, 1.0, 0.0, 0.0]),
    map(mpf, [0.0, 1 - 0.0002, 0.0002, 0.0]),
    map(mpf, [0.0, 0.0, 1 - 0.0002, 0.0002]),
    map(mpf, [0.0, 0.0, 0.0, 0.0]),
], dtype=mpf)

INITIAL_EMISSION_PROB = np.array([
    map(mpf, [1.0, 0.0, 0.0, 0.0, 0.0]),
    map(mpf, [0.0, 0.96, 0.036, 0.004, 0.0]),
    map(mpf, [0.0, 0.96, 0.004, 0.036, 0.0]),
    map(mpf, [0.0, 0.0, 0.0, 0.0, 1.0]),
], dtype=mpf)

INITIAL_PROB = np.array(
    map(mpf, [1.0, 0.0, 0.0, 0.0]),
    dtype=mpf,
)

HIDDEN_STATES = ['S0', 'S1', 'S2', 'S3']
SYMBOLS = ['S', 'N', 'L', 'R', 'E']


def forward(hmm, X, Y, Yt):
    n = len(Yt)
    m = len(X)
    alpha = np.zeros((n, m), dtype=mpf)

    alpha[0, :] = (
        hmm.initial_prob
        .dot(np.diag(hmm.emission_prob[:, Yt[0]]))
    )

    for t in xrange(1, n):
        alpha[t, :] = (
            alpha[t - 1, :]
            .dot(hmm.transition_prob)
            .dot(np.diag(hmm.emission_prob[:, Yt[t]]))
        )

    return alpha


def backward(hmm, X, Y, Yt):
    n = len(Yt)
    m = len(X)
    beta = np.zeros((n, m), dtype=mpf)

    beta[n - 1, :] = [mpf(1.0)] * m

    for t in xrange(n - 2, -1, -1):
        beta[t, :] = (
            hmm.transition_prob
            .dot(np.diag(hmm.emission_prob[:, Yt[t + 1]]))
            .dot(beta[t + 1, :].T)
        )

    return beta


def baum_welch(hmm, X, Y, sequence, eps=1e-6):
    """
    Baum-welch implementation

    Arguments:
    ----------
    hmm_params: HMMParams
        Transition, emission and initial state probabilities for the
        HMM model.
    X: iterable
        Collection of hidden states.
    Y: iterable
        Collection of observable states.
    sequence: str
        Observed sequence.

    Returns HMMParams namedtuple containing HMM probabilities.
    """
    n = len(sequence)
    m = len(X)
    Yt = [Y.index(yt) for yt in sequence]

    while True:
        alpha, beta, gamma = forward_backward(hmm, X, Y, sequence)

        xsi = np.zeros((n, m, m), dtype=mpf)
        for t in xrange(n - 1):
            for i in xrange(m):
                for j in xrange(m):
                    xsi[t, i, j] = (
                        alpha[t, i] *
                        hmm.transition_prob[i, j] *
                        beta[t + 1, j] *
                        hmm.emission_prob[j, Yt[t + 1]]
                    )
            xsi[t] /= np.sum(xsi[t])

        # Update
        pi = gamma[0]

        transition_prob = np.zeros((m, m), dtype=mpf)
        for i in xrange(m):
            den = np.sum(gamma[:, i])
            for j in xrange(m):
                transition_prob[i, j] = np.sum(xsi[:, i, j]) / den

        emission_prob = np.zeros((m, len(Y)), dtype=mpf)
        for i in xrange(m):
            den = np.sum(gamma[:, i])
            for j in xrange(len(Y)):
                emission_prob[i, j] = np.sum(gamma[np.array(Yt) == j, i]) / den

        diff_transition = np.max(hmm.transition_prob - transition_prob)
        diff_emission = np.max(hmm.emission_prob - emission_prob)
        if diff_transition < eps and diff_emission < eps:
            break
        hmm = HMMParams(transition_prob, emission_prob, pi)

    return hmm


def forward_backward(hmm_params, X, Y, sequence):
    """
    Forward-backward algorithm implementation.

    Arguments:
    ----------
    hmm_params: HMMParams
        Transition, emission and initial state probabilities for the
        HMM model.
    X: iterable
        Collection of hidden states.
    Y: iterable
        Collection of observable states.
    sequence: str
        Observed sequence.

    Returns:
    --------
    alpha: numpy.ndarray
        Probability of seeing the y_1,y_2,...,y_t and being in state i
        at time t. alpha_i(t)=P(Y_1=y_1,...,Y_t=y_t,X_t=i|hmm_params)
    beta: numpy.ndarray
        Probability of the ending partial sequence y_{t+1},...,y_{T}
        given starting state i at time t.
        beta_i(t)=P(Y_{t+1}=y_{t+1},...,Y_{T}=y_{T}|X_t=i, hmm_params)
    gamma: numpy.ndarray
        Probability of being in state i at time t given the observed
        sequence Y and the parameters hmm_params:
        gamma_i(t)=P(X_t=i|Y, hmm_params)
    """
    Yt = [Y.index(yt) for yt in sequence]
    n = len(sequence)
    m = len(X)

    alpha = forward(hmm_params, X, Y, Yt)
    beta = backward(hmm_params, X, Y, Yt)
    gamma = np.zeros((n, m), dtype=mpf)
    for t in xrange(n):
        gamma[t, :] = [alpha[t, i] * beta[t, i] for i in xrange(m)]
        gamma[t, :] /= np.sum(gamma[t, :])

    return alpha, beta, gamma


if __name__ == '__main__':
    """
    Usage:

    python bw.py < seq.txt
    """
    with open(sys.argv[1]) as fp:
        for i, seq in enumerate(fp):
            seq = seq.strip()

            print 'Calculating params for sequence %d...' % i
            n = len(seq)

            hmm_params = baum_welch(
                HMMParams(
                    transition_prob=INITIAL_TRANSITION_PROB,
                    emission_prob=INITIAL_EMISSION_PROB,
                    initial_prob=INITIAL_PROB,
                ),
                HIDDEN_STATES,
                SYMBOLS,
                seq)
            _, _, prob = forward_backward(
                hmm_params, HIDDEN_STATES, SYMBOLS, seq)

            # Create files with calculated params
            for param in HMM_PARAM_NAMES:
                param_values = getattr(hmm_params, param)
                np.savetxt(
                    'params/%s.%d.csv' % (param, i),
                    param_values,
                    delimiter=','
                )

            # Create plots
            matplotlib.rc('xtick', labelsize=5)
            plt.figure(1)

            # First subplot
            plt.subplot(211)
            plt.xticks(range(0, n - 1), seq)
            plt.plot(1.0 - abs(prob[1:, 1] - prob[: -1, 2]))

            # Second subplot
            plt.subplot(212)
            plt.xticks(range(0, n - 1), seq)
            plt.plot(prob[0: n - 1, 1: 3])

            plt.savefig('plots/%d.svg' % i)
            plt.close()

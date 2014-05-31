import random
import numpy as np

N_SEQ = 10


START = 0
BEFORE = 1
AFTER = 2
END = 3


def gen_seq():
    seq = []
    state = START

    while state != END:
        if state == START:
            state = BEFORE
            seq.append('S')
        if state == BEFORE:
            n, l, r = np.random.multinomial(1, [0.96, 0.036, 0.004])
            if n:
                seq.append('N')
            elif l:
                seq.append('L')
            else:
                seq.append('R')

            state += np.random.binomial(1, 1/5000.)
        if state == AFTER:
            n, l, r = np.random.multinomial(1, [0.96, 0.004, 0.036])
            if n:
                seq.append('N')
            elif l:
                seq.append('L')
            else:
                seq.append('R')

            state += np.random.binomial(1, 1/5000.)

    seq.append('E')
    return seq


if __name__ == '__main__':
    random.seed(42)
    for i in xrange(N_SEQ):
        seq = gen_seq()
        print ''.join(seq)

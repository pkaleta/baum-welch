import random

N_SEQ = 100


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
            if random.randint(0, 100) == 0:
                state = AFTER
            if random.randint(0, 100) in range(0, 5):
                seq.append('R')
            else:
                seq.append('L')
        if state == AFTER:
            if random.randint(0, 100) == 0:
                state = END
            if random.randint(0, 100) in range(0, 5):
                seq.append('L')
            else:
                seq.append('R')

    seq.append('E')
    return seq


if __name__ == '__main__':
    random.seed(42)
    for i in xrange(N_SEQ):
        seq = gen_seq()
        print ''.join(seq)

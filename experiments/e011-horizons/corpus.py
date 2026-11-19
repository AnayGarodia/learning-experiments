"""Multi-scale synthetic corpus (e011). Extension of e005's generator.

Fixed sentence layout so every scale's fork->consequence distance is
EXACT (noise replaces designated offsets instead of inserting):
  offset 0: FORK f
  offset 4: SHORT consequence  = CONS_S0 + f
  offset 16: MEDIUM consequence = CONS_M0 + f
  offsets 8-11: noise (irreducible, no consequence)
  long variant only:
    offset 64: LONG consequence = CONS_L0 + f
    offsets 40-47: second noise region
  all other offsets: structure token T[f, offset] (deterministic).
Labels: 0=fork 1=structure 2=noise 3=short 4=medium 5=long.
Dial: long_density = fraction of sentences carrying the long tail.
"""
import numpy as np

K = 8
VOCAB = 5000
FORK0 = 1
CONS_S0, CONS_M0, CONS_L0 = 20, 40, 60   # 8 ids each, disjoint
STRUCT0 = 100
NOISE0, N_NOISE = 4000, 1000
SHORT_LEN, LONG_LEN = 21, 69             # tokens incl. fork (offsets 0..20 / 0..68)
NOISE_A = range(8, 12)
NOISE_B = range(40, 48)


class CorpusGen:
    def __init__(self, seed=0, long_density=1.0):
        self.rng = np.random.default_rng(seed)
        self.long_density = long_density
        table_rng = np.random.default_rng(12345)  # fixed: the "language"
        self.T = table_rng.integers(STRUCT0, STRUCT0 + 800, size=(K, LONG_LEN))

    def sentence(self):
        f = int(self.rng.integers(K))
        is_long = self.rng.random() < self.long_density
        L = LONG_LEN if is_long else SHORT_LEN
        toks = self.T[f, :L].copy()
        labs = np.ones(L, dtype=np.int64)
        toks[0], labs[0] = FORK0 + f, 0
        toks[4], labs[4] = CONS_S0 + f, 3
        toks[16], labs[16] = CONS_M0 + f, 4
        for i in NOISE_A:
            toks[i], labs[i] = self.rng.integers(NOISE0, NOISE0 + N_NOISE), 2
        if is_long:
            toks[64], labs[64] = CONS_L0 + f, 5
            for i in NOISE_B:
                toks[i], labs[i] = self.rng.integers(NOISE0, NOISE0 + N_NOISE), 2
        return toks, labs

    def batch(self, batch_size, seq_len):
        x = np.zeros((batch_size, seq_len), dtype=np.int64)
        lab = np.full((batch_size, seq_len), -1, dtype=np.int64)
        for b in range(batch_size):
            pos = 0
            while pos < seq_len:
                t, l = self.sentence()
                n = min(len(t), seq_len - pos)
                x[b, pos:pos + n] = t[:n]
                lab[b, pos:pos + n] = l[:n]
                pos += n
        return x, lab


if __name__ == "__main__":
    g = CorpusGen(0, long_density=0.5)
    x, lab = g.batch(8, 256)
    # invariants: every consequence matches the nearest preceding fork,
    # at EXACTLY its scale's distance
    dist = {3: 4, 4: 16, 5: 64}
    checked = 0
    for b in range(8):
        forks = [(i, x[b, i] - FORK0) for i in range(256) if lab[b, i] == 0]
        for c, base in ((3, CONS_S0), (4, CONS_M0), (5, CONS_L0)):
            for i in range(256):
                if lab[b, i] == c:
                    prior = [(j, ff) for j, ff in forks if j <= i]
                    if not prior:
                        continue  # sentence truncated at seq start? impossible; safe skip
                    j, ff = prior[-1]
                    assert i - j == dist[c], (c, i - j)
                    assert x[b, i] - base == ff, "consequence must match fork"
                    checked += 1
    frac = {c: float((lab == c).mean()) for c in range(6)}
    print("class fractions:", {k: round(v, 3) for k, v in frac.items()})
    print(f"corpus self-check ok ({checked} consequences verified)")

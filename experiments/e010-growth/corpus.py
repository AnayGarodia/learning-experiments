"""Synthetic corpus with ground-truth token classes (dossiers 005/006/011).

Sentence = [FORK f] [STRUCTURE body: tokens deterministic given (f, pos)]
[NOISE span, dialable] [CONSEQUENCE token g(f)].
- fork: irreducible entropy log2(K), outcome matters downstream
- structure: entropy 0 given f (learnable signal)
- noise: irreducible entropy, no downstream consequence
- consequence: deterministic given f, at distance body+noise away
Oracle labels per token: 0=fork 1=structure 2=noise 3=consequence.
Dials: noise_p/noise_len (005), dup_p redundancy (006), body_len horizon (011).
"""
import numpy as np

K = 8                # fork choices
VOCAB = 5000
FORK0, STRUCT0, CONS0, NOISE0 = 1, 100, 60, 4000
N_NOISE = 1000


class CorpusGen:
    def __init__(self, seed=0, body_len=12, noise_p=0.5, noise_len=6, dup_p=0.0):
        self.rng = np.random.default_rng(seed)
        self.body_len, self.noise_p, self.noise_len, self.dup_p = \
            body_len, noise_p, noise_len, dup_p
        # deterministic body table: token = T[f, j], drawn once
        table_rng = np.random.default_rng(12345)  # fixed across seeds: the "language"
        self.T = table_rng.integers(STRUCT0, STRUCT0 + 800, size=(K, body_len))
        self.seen = []

    def sentence(self):
        if self.seen and self.rng.random() < self.dup_p:
            return self.seen[self.rng.integers(len(self.seen))]
        f = int(self.rng.integers(K))
        toks = [FORK0 + f]
        labs = [0]
        toks += list(self.T[f]); labs += [1] * self.body_len
        if self.rng.random() < self.noise_p:
            n = 1 + int(self.rng.integers(self.noise_len))
            toks += list(self.rng.integers(NOISE0, NOISE0 + N_NOISE, size=n))
            labs += [2] * n
        toks.append(CONS0 + f); labs.append(3)
        s = (np.array(toks, dtype=np.int64), np.array(labs, dtype=np.int64))
        if self.dup_p > 0:
            self.seen.append(s)
            if len(self.seen) > 5000:
                self.seen.pop(0)
        return s

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
    g = CorpusGen(0)
    x, lab = g.batch(4, 256)
    # oracle invariants: consequence token always equals fork of its sentence
    for b in range(4):
        forks = [(i, x[b, i] - FORK0) for i in range(256) if lab[b, i] == 0]
        cons = [(i, x[b, i] - CONS0) for i in range(256) if lab[b, i] == 3]
        for (ci, cf) in cons:
            pf = [f for (i, f) in forks if i < ci][-1]
            assert pf == cf, "consequence must match its sentence's fork"
    # structure determinism: same f -> same body
    frac = {c: (lab == c).mean() for c in (0, 1, 2, 3)}
    print("class fractions:", {k: round(v, 3) for k, v in frac.items()})
    print("corpus self-check ok")

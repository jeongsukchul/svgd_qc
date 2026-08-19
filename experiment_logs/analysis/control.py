import numpy as np
exec(open('/tmp/claude-1000/-home-sukchul-qc/682f546f-7056-4a11-8268-2787858681d7/scratchpad/multimodality.py')
     .read().split("cond_sd, sep_nn")[0].replace("REF, NQ, K = 300_000, 1500, 64",
                                                  "REF, NQ, K = 1000, 1, 64"))
rng = np.random.default_rng(1)
print("\n--- positive control: does the statistic detect known bimodality? ---")
for gap in (0.0, 0.25, 0.5, 1.0):
    vals = []
    for _ in range(300):
        A = rng.normal(0, 0.3, size=(64, 8))
        A[:32, 0] += gap                       # plant a 2-mode split in dim 0
        vals.append(two_means_separation(A))
    print(f"  planted gap {gap:4.2f} ({gap/0.3:4.1f} sigma) -> measured separation {np.median(vals):.2f}")
print("\n  (antmaze neighbourhoods measured 3.08; random mixtures 3.55)")

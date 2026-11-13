# Qwen Calibration Summary

This calibration run uses 100 original questions and four evidence
conditions for each method.

| Method | Condition | Unsafe answer | Correct abstention | Coverage | Avg total tokens |
|---|---:|---:|---:|---:|---:|
| G2 | C0 | 0.00 | 0.00 | 0.34 | 2375.9 |
| G2 | C1 | 0.20 | 0.80 | 0.20 | 2473.5 |
| G2 | C2 | 0.00 | 0.00 | 0.28 | 2452.4 |
| G2 | C3 | 0.20 | 0.80 | 0.20 | 2553.6 |
| G3 | C0 | 0.00 | 0.00 | 0.39 | 2852.7 |
| G3 | C1 | 0.24 | 0.76 | 0.24 | 2866.6 |
| G3 | C2 | 0.00 | 0.00 | 0.32 | 2848.4 |
| G3 | C3 | 0.23 | 0.77 | 0.23 | 2895.8 |

The current confidence outputs are concentrated near the same values, so
threshold selection does not materially separate the methods. The project
should report this calibration limitation instead of claiming a strong
threshold optimization effect.

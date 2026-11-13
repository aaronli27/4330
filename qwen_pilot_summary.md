# Qwen Pilot Summary

This is a pilot result, not the final statistical evaluation. It uses
Qwen2.5-1.5B-Instruct with 4-bit loading, 50 questions per condition, and
the current deterministic protocol.

| Method | Condition | EM | F1 | Unsafe answer | Correct abstention | Coverage | Avg total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | C0 | 0.160 | 0.193 | 0.000 | 0.000 | 0.440 | 975.0 |
| G0 | C1 | 0.020 | 0.076 | 0.280 | 0.720 | 0.280 | 997.4 |
| G0 | C2 | 0.180 | 0.214 | 0.000 | 0.000 | 0.400 | 965.7 |
| G0 | C3 | 0.060 | 0.116 | 0.280 | 0.720 | 0.280 | 997.9 |
| G1 | C0 | 0.100 | 0.155 | 0.000 | 0.000 | 0.480 | 1424.5 |
| G1 | C1 | 0.100 | 0.108 | 0.300 | 0.700 | 0.300 | 1439.1 |
| G1 | C2 | 0.140 | 0.183 | 0.000 | 0.000 | 0.360 | 1450.5 |
| G1 | C3 | 0.100 | 0.134 | 0.260 | 0.740 | 0.260 | 1441.5 |
| G2 | C0 | 0.140 | 0.173 | 0.000 | 0.000 | 0.400 | 2362.7 |
| G2 | C1 | 0.000 | 0.056 | 0.220 | 0.780 | 0.220 | 2445.7 |
| G2 | C2 | 0.160 | 0.194 | 0.000 | 0.000 | 0.340 | 2336.8 |
| G2 | C3 | 0.020 | 0.076 | 0.200 | 0.800 | 0.200 | 2593.9 |
| G3 | C0 | 0.100 | 0.155 | 0.000 | 0.000 | 0.480 | 2696.8 |
| G3 | C1 | 0.100 | 0.108 | 0.300 | 0.700 | 0.300 | 2763.8 |
| G3 | C2 | 0.140 | 0.183 | 0.000 | 0.000 | 0.360 | 2743.4 |
| G3 | C3 | 0.100 | 0.134 | 0.260 | 0.740 | 0.260 | 2721.9 |

The pilot suggests that G2 is currently the stronger reliability baseline on
the missing-evidence conditions, while G3 does not yet show a clear advantage.
This is a pilot observation only; threshold calibration, larger splits, and
failure analysis are still required before drawing a project conclusion.

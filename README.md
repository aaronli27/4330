# Controlled Multi-hop RAG Study

This directory contains the core implementation for the HotpotQA
evidence-coverage study.

## Current protocol

- Dataset: HotpotQA distractor development file.
- Main protocol: only records with exactly 10 candidate paragraphs and
  valid supporting-fact indices.
- Valid local records: 7,344.
- Conditions: C0 complete evidence, C1 missing necessary hop, C2 hard
  distractor, C3 combined perturbation.
- Main methods: G0 one-shot BM25, G1 fixed-depth retrieval, G2 global
  sufficiency gate, G3 subquestion-level coverage gate.
- Default shared model budget: 4,096 total input/output tokens per question,
  with a 128-token per-call output cap.

## Run the data protocol

```powershell
$py = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py tools/build_protocol.py
```

The command writes split IDs, a protocol manifest, and one variants JSONL per
split under `data/`. The calibration split is the only donor pool for
hard-distractor construction, so test question IDs do not contribute donor
paragraphs.

## Run core tests

```powershell
& $py -m unittest discover -s tests -v
```

The model adapter is intentionally separate from the deterministic data and
retrieval layer. This keeps the protocol reproducible even when a GPU runtime
or the Transformers package is unavailable.

## Run a method with the deterministic backend

```powershell
& $py tools/run_experiment.py --method G0 --backend fake --limit 20
```

Replace `--backend fake` with `--backend qwen` after installing the model
runtime in the execution environment. The output schema is the same for both
backends.

The local Windows setup uses the ModelScope snapshot at
`models/Qwen2.5-1.5B-Instruct` and 4-bit loading by default:

```powershell
& $py tools/run_matrix.py `
  --split pilot `
  --backend qwen `
  --model-name "D:\项目\大作业\multihop_rag_project\models\Qwen2.5-1.5B-Instruct"
```

Run all four methods on a split:

```powershell
& $py tools/run_matrix.py --split pilot --backend fake
```

After running a gate method on the calibration split, select its threshold
without touching the standard or unseen test split:

```powershell
& $py tools/calibrate_threshold.py `
  --results results/qwen_G3_calibration.jsonl `
  --output results/qwen_G3_threshold.json
```

## Qwen runtime

The optional model adapter uses the packages listed in
`requirements-qwen.txt`. Install them in the Colab or GPU environment that
will run the real model. The local deterministic tests do not require these
packages.

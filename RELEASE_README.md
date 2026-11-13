# Final Submission Package

This package contains the final English report, source code, protocol
manifests, standard-test and unseen-test JSONL results, and summary files.

The Qwen2.5-1.5B-Instruct model weights are intentionally excluded because
they are approximately 3.1 GB. The local execution model is documented in
the parent project README and `requirements-qwen.txt`.

The final report is based on:

- 100 calibration question IDs
- 100 standard-test question IDs
- 100 unseen-test question IDs
- Four evidence conditions per question
- Four methods: G0, G1, G2, G3

The report records the negative result that subquestion-level gating did not
consistently outperform global gating in this implementation.

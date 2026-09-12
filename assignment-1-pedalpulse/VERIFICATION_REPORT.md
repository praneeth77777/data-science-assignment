# PedalPulse Chunk 8 verification report

Verification date: 2026-09-11 UTC

## Executed results

| Check | Measured result |
|---|---|
| Python runtime | 3.12.14 in a project-local `.venv` created from `requirements.txt` |
| Environment checker | PASS; no missing dependencies or corrective actions |
| Dependency consistency | `pip check`: no broken requirements |
| Pytest suite | 19 passed in 2.28 seconds |
| Streamlit smoke test | 6 pages rendered; prediction interaction completed; 0 uncaught exceptions |
| Figure integrity | 23 PNG files checked; 0 invalid |
| Notebook execution | 7/7 code cells executed; 18 outputs; 7 embedded PNGs; 0 error outputs |
| Deployment model training | 9,519 development rows; 0 locked-test rows used; no locked-test metric computation |
| Model artifact | 24,301,017 bytes; SHA-256 `f8f4f44cd2e50b82b4e9464dc720d85d94e5f7ba34a3d412d34ba711bba68ced` |
| Model metadata | SHA-256 `1301d2c99fe787cce7ca7b5c828160ffce70af22d8d0ed9c818abe1ae814101e` |
| Executed notebook | 1,958,178 bytes; SHA-256 `c0ec52465509fe4513b2930a4f1408b77ea3daf220986ef86e6171322598d1fa` |
| Raw Kaggle CSV files inside project | 0 |
| Absolute temporary-workspace paths in text artifacts | 0 |
| Broken local README links | 0 |
| Clean extracted submission bundle | 19 tests passed; 6 Streamlit pages plus prediction passed; environment check passed without raw data |

The deployment artifact uses 74 transformed features and the locked `rf_raw_depth18_leaf1` configuration with random seed 42.

## Errors encountered and corrections

1. `streamlit`, notebook-execution packages, and `pytest` were initially absent. Pinned versions were installed and recorded in `requirements.txt`.
2. Jupyter-kernel execution failed because the managed environment blocked the kernel's local socket. An in-process execution fallback was added; all notebook cells then executed successfully. The regular kernel path remains the default for unrestricted local environments.
3. The first server-style smoke test was blocked before execution by the environment's localhost-network approval boundary. Streamlit's in-process `AppTest` was adopted for automated verification.
4. The all-page smoke test detected that `temporal_demand_patterns.png` was truncated. It was regenerated using the original development-only EDA generator. All 19 figures then passed Pillow verification.
5. Deprecated `use_container_width` calls were replaced with the supported `width="stretch"` API.
6. A later managed-session restart no longer contained the previously installed optional packages. Installing directly from the repository's `requirements.txt` restored the environment, after which tests and smoke checks passed again.
7. The final dependency installer session disconnected from the tool transport after package download. Inspection showed installation had completed; `pip check`, the full test suite, the environment checker, and the Streamlit smoke test then passed in the project-local virtual environment.

No locked-test metric was recomputed while fixing or verifying Chunk 8.

## Pending publication-only checks

- GitHub Actions has not executed because no GitHub repository has been created or modified.
- Public visibility and README rendering have not been verified.
- A clean clone from the public repository has not been tested.
- Native Windows execution remains for the post-publication handoff.

These items must remain pending until the student approves the final GitHub checkpoint.

A clean extraction of the 28 MB local submission ZIP was tested successfully. This verifies archive self-containment but does not replace the required post-publication GitHub clone test.

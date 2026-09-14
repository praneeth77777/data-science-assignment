# SegmentForge Verification Report

Verification date: 2026-09-14 UTC

## Executed results

| Check | Measured result |
|---|---|
| Raw workbook | 541,909 rows × 8 columns |
| Dataset SHA-256 | `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d` |
| Customer features | 4,338 customers |
| Discovery / locked audit | 3,443 / 895 customers |
| AutoResearch | 42 trials: 34 valid, 8 rejected, 0 failed in corrected run |
| Deterministic rerun | Same winning configuration and identical discovery/audit metrics |
| Selected configuration | K-Means, k=4, RFM, Yeo–Johnson, StandardScaler, keep outliers |
| Discovery silhouette | 0.3331456842 |
| Locked-audit silhouette | 0.3383167906 |
| Stability | ARI 0.9888225015 ± 0.0025120009 |
| Unit tests | 16 passed in 0.86 seconds (final pre-package run) |
| Streamlit smoke test | 8 pages rendered; 0 uncaught exceptions |
| Dependency check | `pip check`: no broken requirements |
| Environment checker | PASS on Python 3.12.14 |
| Notebook | 6/6 code cells executed; 12 outputs; 0 error outputs |
| Figures | 7 PNG files checked; 0 invalid |
| Analysis runtime | 55.52 seconds on the recorded environment |
| Clean exported package | 16 tests passed; 8 dashboard pages rendered; raw workbook absent |
| Model artifact | 45,207 bytes; SHA-256 `b108b83db7a0334740c8a24290a850815e4f60fc9eadc2e0c607e5aecdf418c3` |
| Model metadata | SHA-256 `ecd49e9e1f666d289eff09b56b17f65a15d365e6d6d1b971e5bd0250938b880b` |
| Executed notebook | 1,125,785 bytes; SHA-256 `946ca409513bdf0faee247d54f8b0b51ba8f4b74674720bab55d34b8d9f99516` |

## Errors and corrections

1. The first complete pipeline run finished its 42 clustering trials but failed while serializing the selected state because pandas returned `n_clusters` as NumPy `int64`. A typed `state_from_row` conversion was added. The corrected run completed, and a second full run selected the identical state with identical metrics.
2. The analysis runtime did not contain pytest, so the first test invocation failed before collecting tests. Tests were rerun with the pinned project environment; 16 passed.
3. The pinned project environment initially lacked `openpyxl`. Version 3.1.5, already declared in `requirements.txt`, was installed; dependency and environment checks then passed.
4. The first notebook command used a runtime without `nbformat`. It was rerun with the project environment.
5. Jupyter kernel execution was blocked by the managed environment's local-socket restriction. An in-process fallback was implemented and executed all six cells with zero errors. A normal kernel remains the default on unrestricted local machines.
6. A final verification shell command initially addressed project-relative files while its working directory was already the project root. It stopped at `chmod` before tests ran. The paths were corrected; the full verification chain then passed.
7. The first local Markdown-link audit referenced the reusable test interpreter relative to the repository root as though it were running from the project directory. That audit did not run; the local commit was unaffected. The command was corrected and all 16 local Markdown targets resolved.

No result from an incomplete or failed command is reported as successful.

## Leakage, privacy, and publication audit

- `CustomerID` and country are excluded from all clustering feature views.
- The audit population is not used by the hill-climbing search.
- Preprocessing is fitted on discovery customers or declared stability subsamples.
- Customer-level exports use deterministic masked identifiers.
- The raw workbook and ZIP are excluded by the repository `.gitignore`.
- Persona names were assigned only after the configuration was locked.
- Production drift, causal impact, campaign ROI, and future customer value remain unmeasured.

## Pending external verification

- Native macOS and Windows dependency installation for this new project.
- GitHub Actions execution after an approved push.
- Public README and image rendering.
- A real student-recorded YouTube URL.

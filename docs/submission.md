# Submission readiness and remaining course actions

This checklist follows the supplied PMLDL Projects 2026 specification. It is
not a claim that an instructor has approved the topic or awarded any score.

## Implemented evidence for the Blue project

| Criterion | Evidence |
| --- | --- |
| Test several existing solutions | Majority, Ridge, CatBoost, XGBoost and ExtraTrees; current metrics in `results/nested_cv_results.csv` |
| Train a proposed model | Four-member equal ensemble; `src/experiment.py`, evaluated on outer OOF and refitted for deployment |
| Modify architecture or tune hyperparameters | Three candidates per member, selected inside each outer training partition; baseline-versus-tuned comparison and paired intervals |
| Implementation and results | Fold-isolation tests, independent score checks, fixed class scale, matched export policy, rare-class errors and uncertainty |
| Reproducibility | Fresh tested pinned environment, raw/source/result digests, data access and executable workflow in README |
| Report | Generated `project.pdf` with the seven required sections, tables, plots and limitations |
| Attribution | Algorithms, libraries and data are attributed in the report; AI assistance disclosed |
| Anonymous source archive | `dist/project.zip`, sanitized notebook copies, explicit allowlist and verified SHA256 manifest |

## Before manually submitting Stage 2

- Confirm that the predictive `sii` task matches the actual assigned or proposed
  topic. The listed course topic mentions internet use and children's physical
  activity; this project does not estimate a causal effect. Do not silently
  substitute a different target or invent instructor approval.
- Complete the README reference workflow. Use the current `nested_` results,
  not the historical exploratory scores, when discussing performance.
- Open the PDF and inspect the extracted ZIP. Automated identity-pattern
  detection is limited; a human should check source comments, notebook text,
  images and report metadata for identifying clues.
- Submit the generated archive to Moodle before the Stage 2 deadline. A GitHub
  PR is for development, not an anonymous course submission.

## What this repository cannot complete by itself

The specification computes `FinalScore = clip(0.7 * BlueScore + 0.3 * RedScore
- penalties, 0, 100)`. Improving this repository addresses the Blue project;
it does not complete the independent Red-team assignment.

At Stage 3, inspect the actually assigned other project and submit an anonymous
`review.pdf` with the instructor-assigned anonymized team IDs, a separate
anonymity note, and up to five evidenced weaknesses, at most one per criterion.
Do not invent weaknesses or plagiarism evidence. These inputs are not present
in this repository, so no review of another project is fabricated here.

At Stage 4, respond to the actual received review in `rebuttal.pdf`, using the
assigned IDs and one justified Accept/Challenge response per weakness. Neither
the received review nor those IDs is currently available here.

Optional Phase 2 requires the TA's confirmed weaknesses and an additional
`fixes.pdf` mapping each one to its implemented fix. The present changes are
proactive Stage 2 preparation, not a fabricated TA decision or Phase 2 rebuttal.

No measured positive delta, test suite, checklist, or generated archive can
guarantee 100 points: approval, human assessment, review quality and timely
submission remain part of the course process.

# AI usage

## Tools used
- Claude (Anthropic): used as a mentor and code assistant across several chats of one project. Scripts `05`-`08` and `trend_extra.py` were written with Claude Sonnet 5.5 in one chat; earlier modules and scripts `01`-`04` were written in other Claude chats [ĐIỀN: tên mô hình nếu biết].
- ChatGPT (OpenAI): used to draft a methods dossier early in the project [ĐIỀN: mô hình]. Its content was reviewed and only methods the team could justify were kept.
- [ĐIỀN: AI khác nếu có, ví dụ nguồn của đề xuất PCI / phân tích điều hòa].

## What AI did
- Wrote or drafted: most Python modules and scripts, tests, documentation drafts, and explanations of statistical methods.
- Suggested analysis options (e.g. Hamed-Rao correction, Benjamini-Yekutieli, sensitivity checks, PCI and harmonic analysis).

## What the team did
- Chose the challenge, study region, data, and every analysis decision.
- Ran every script on the real NASA data and checked outputs (units, area totals, seasonal cycle, mean rainfall of about 2,173 mm/yr for the basin).
- Decided which results are reported and how they are worded, including negative results.

## Verification
- `trendlab` core modules are covered by unit tests (hand-calculated cases and synthetic data).
- The Hamed-Rao implementation was cross-checked against the `pymannkendall` package on 30 simulated series in the assistant's environment; this check is not included as a repository test. Pettitt was checked against a hand calculation. Scripts `05`-`08` have no automated tests; they were checked by the printed sanity checks and by reading the outputs.
- AI-suggested explanations (physical mechanisms, references) were not accepted as results unless supported by our own computation; mechanisms are not claimed in the reported findings.
# SegmentForge Data Card

Source: UCI Online Retail, DOI 10.24432/C5BW33, CC BY 4.0.

- Raw shape: 541,909 rows × 8 columns.
- Coverage: 2010-12-01T08:26:00 through 2011-12-09T12:50:00.
- Exact duplicates documented: 5,268.
- Missing CustomerID rows documented: 135,080.
- Customer-level feature rows: 4,338.
- Raw workbook is excluded from Git and must be downloaded by each user.

Preparation rules and their overlapping row counts are stored in `artifacts/tables/exclusion_audit.csv`. Exclusion counts must not be summed because a row may meet more than one rule.

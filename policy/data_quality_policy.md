# Transaction Data-Quality & Treatment Policy

doc_id: data_quality_policy
version: 2026-07.2
effective: 2026-07-01

> **Illustrative knowledge base.** Synthetic rules written for this prototype,
> governing how Open-Banking transaction data must be cleaned and interpreted
> before an affordability assessment. Not any real firm's policy.

### [DQ-001] Unclassified (unknown) transactions
Transactions that cannot be confidently categorised must be labelled `unknown`,
never guessed. Unknown **debits** must be treated conservatively: they reduce
disposable income in full. Where unknown transactions exceed **10% of total debit
value**, the automated assessment is unreliable and the case must be **referred**
for manual review.

### [DQ-002] Internal transfers and savings movements
Transfers between the customer's own accounts, and movements into savings or
pots, are money **moved**, not money earned or spent. Both legs must be excluded
from income and from expenditure. Treating an own-account transfer as income
overstates affordability and is a serious assessment defect.

### [DQ-003] Refunds and reversals
A refund is the return of earlier spending, **never income**. Where a refund can
be matched to an earlier debit from the same merchant, it must be netted against
that spending category. An unmatched refund must be excluded from the assessment
entirely.

### [DQ-004] Duplicate transactions
Exact duplicate postings (same account, date, description, amount and type) must
be detected and removed before calculation, and the removal must be recorded in
the assessment's evidence so a reviewer can see what was excluded.

### [DQ-005] Cash withdrawals
Cash withdrawals are spending whose purpose cannot be verified from account data.
They count as expenditure, but where cash withdrawals exceed **25% of total debit
value** the spending picture is too opaque for an automated recommendation and
the case must be **referred**.

### [DQ-006] Income recognition and coverage
Eligible income means inflows recognisable as employment, self-employment or
benefits income. Unrecognised inflows are not income (DQ-001, DQ-003 apply).
Eligible income must be observed in at least **two** distinct months. The
observation window must be **contiguous**: a missing month indicates incomplete
account coverage. Where no eligible income is recognised, or the window has
gaps, the case must be **referred**, not scored.

### [DQ-007] Classification plausibility
Category assignments must pass basic plausibility checks before an automated
assessment is trusted. Transfers between a customer's own accounts occur in
matching pairs: where internal-transfer debits and credits do not broadly
balance across the customer's accounts, the classification is unreliable.
Likewise, an observation window of normal length with **no essential spending
at all** (no housing, utilities, food or transport) indicates classification
failure rather than genuine behaviour. In either case the automated assessment
must be **referred** for manual review.

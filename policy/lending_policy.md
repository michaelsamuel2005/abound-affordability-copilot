# Responsible Lending & Affordability Policy

doc_id: lending_policy
version: 2026-07.2
effective: 2026-07-01

> **Illustrative knowledge base.** These rules are synthetic and were written for
> this prototype. They are *informed by* the principles of the FCA Consumer Credit
> sourcebook (CONC 5.2A — creditworthiness and affordability) but are **not** any
> real firm's policy and carry no regulatory status. Each rule is a discrete,
> retrievable section with a stable ID so the decision pipeline can ground and
> cite its reasoning. Thresholds are illustrative prototype values.

### [POL-001] Affordability basis — cashflow, not credit score
Creditworthiness must be assessed primarily on the customer's demonstrated ability
to repay from **disposable income** derived from their transaction (cashflow) data,
not on a credit score alone. Cashflow underwriting looks at what actually enters and
leaves the customer's accounts over the observation window.

### [POL-002] Minimum disposable-income buffer
After deducting the proposed new monthly repayment, the customer must retain a
minimum monthly **disposable-income buffer of £150**. Disposable income means
assessed monthly income less essential expenditure, existing debt repayments and
unclassified spend. If the buffer would fall below £150 the application must not be
approved at the requested amount: reduce the amount to the maximum affordable
(POL-009) or decline.

### [POL-003] Debt-to-income (DTI) limit
Total monthly debt repayments **including the proposed new loan** must not exceed
**45% of assessed monthly income**. A DTI between 40% and 45% must be **referred**
for manual review; a DTI above 45% must be **declined**.

### [POL-004] Income verification and stability
Income must be evidenced by inflows recognisable as employment, self-employment or
benefits income across the observed period. Where income is volatile (coefficient
of variation above **0.35**), affordability must be assessed against a
**conservative, lower income estimate** — no higher than the worse of the weakest
observed month and the mean reduced in proportion to the volatility — and the case
must be **referred** for income verification.

### [POL-005] Gambling and potentially harmful spend
Where gambling transactions exceed **10% of assessed monthly income**, the
application must be **referred** for manual review. Sustained, high-proportion
gambling is an indicator of potential harm and may justify a **decline**; the
review must consider affordability and customer-harm obligations together.

### [POL-006] Signs of financial distress
Overdraft fees, returned direct debits, unpaid-item fees and payday-loan activity
are indicators of financial distress. **Two or more** distress events in the
observed period require a **referral** for manual review.

### [POL-007] Insufficient data (guardrail)
Where fewer than **three months** of transaction history or fewer than **40**
transactions are available, affordability cannot be reliably assessed. Such cases
must be **referred** — never auto-declined solely for a thin file — and must carry
an explicit insufficient-data warning so the reviewer knows why no automated
recommendation was possible.

### [POL-008] Vulnerable customers
Indicators of potential vulnerability — for example, benefits making up more than
half of income combined with essential spending above 60% of income — require
**manual review** and consideration of forbearance and support options. Do not
auto-approve where vulnerability indicators are present.

### [POL-009] Maximum affordable amount
The maximum loan is the largest amount whose monthly repayment still leaves the
customer at or above the £150 disposable-income buffer (POL-002). Never lend above
this figure even where a larger amount is requested. Where the requested amount
fails the buffer test but at least £500 and at least half of the request is
affordable, the case may be referred with a reduced-amount proposal.

### [POL-010] Explainability and fair lending
Every recommendation must be **explainable** with the specific cashflow evidence
(transaction-level) and the policy basis used, must be reviewable by a human
before any decision takes effect, and must **not** rely on protected
characteristics. The reasoning shown to reviewers and auditors must match the
evidence in the account data exactly.

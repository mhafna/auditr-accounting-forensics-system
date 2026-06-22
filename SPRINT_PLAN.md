# Auditr Production Sprint Plan

## Product Goal

Turn Auditr from a polished final-year-project demo into a production-shaped audit workspace that is fast, understandable, traceable, and useful during real review work.

## What is already done

- Project-based workspace with saved engagements
- Dedicated Projects page for create/open/delete/status changes
- Active-project Home landing page
- SQLite-backed local project metadata store with migration from the legacy JSON index
- Auditor-focused Dashboard, Transactions, and Explainability pages
- Help & Support page with CSV template and column alias guidance
- Historical, time-safe fraud features with explainability
- Local session persistence for the demo login flow
- Faster repeated analysis through cached scoring
- Broader CSV parsing support across common delimiters and encodings

## Current technical reality

- Cold analysis on the demo ledger is roughly `1.1s`
- Warm repeated analysis is roughly `0.01s`
- The biggest remaining gaps are workflow depth, persistence, governance, and evidence handling, not raw model latency

## Sprint 1: Workflow Stabilisation

### Goal

Make the current app feel clean, fast, and consistent.

### Scope

- Remove heavy page-transition UI
- Fix spacing, contrast, and control alignment issues
- Keep chart titles and selectors compact
- Reduce duplicate page wording
- Cache repeated project scoring runs
- Add better empty states for every page

### Done when

- Switching between pages feels immediate
- No ugly overlay or double-render artifacts remain
- Home, Projects, Dashboard, Transactions, and Explainability all have clear first actions

## Sprint 2: Case Management

### Goal

Move from "flagged transaction viewer" to "review case workflow".

### Scope

- Add case notes
- Add reviewer assignment
- Add case outcome:
  - Cleared
  - Escalated
  - Need evidence
- Add case status timeline
- Add project-level unresolved-case summary

### Why it matters

This is the biggest jump from student demo to something auditors would actually use.

## Sprint 3: Evidence Layer

### Goal

Support audit evidence instead of only model explanations.

### Scope

- Attach invoice PDFs, screenshots, and support files
- Add evidence checklist completion per case
- Add "what evidence is still missing" prompts
- Generate a case brief export with evidence references

### Why it matters

Auditors clear cases with evidence, not only with risk scores.

## Sprint 4: Persistence Hardening

### Goal

Stabilise the new SQLite layer and make project data safer to operate.

### Scope

- Add schema migration/version handling
- Prepare migration path to Postgres
- Add project archiving instead of hard delete
- Add uploaded-ledger versioning
- Add safer project metadata updates

### Why it matters

This is required before serious multi-user or production deployment.

## Sprint 5: Audit Intelligence

### Goal

Make the product smarter across projects, not only inside one uploaded file.

### Scope

- Cross-project vendor watchlist
- Repeated invoice detection across projects
- Cross-project payment-method changes
- Department/vendor exception history
- Supervisor-facing project risk summary

### Why it matters

This is one of the strongest unique features Auditr can have.

## Sprint 6: Governance and Security

### Goal

Make the system operationally defensible.

### Scope

- Real user accounts
- Role-based access:
  - Admin
  - Auditor
  - Reviewer
- Action logging and audit trail
- Project ownership
- Approval/configuration settings per project
- Secure secret handling outside the app workspace

## Sprint 7: Reporting and Delivery

### Goal

Make the app produce audit-ready outputs.

### Scope

- Project export pack
- Supervisor summary memo
- Open-case register
- Vendor exception memo
- Evidence-backed flagged-case report

## Sprint 8: ML and Model Operations

### Goal

Make the model layer maintainable and measurable.

### Scope

- Model version registry
- Threshold configuration per project
- Retraining workflow outside the app
- Performance monitoring against reviewed cases
- Better fraud-mode reporting
- Optional alternate model experiments

## Highest-value next upgrades

If only a few upgrades can be done next, do these first:

1. Case management
2. Evidence attachments
3. Upload review step with manual column mapping and validation preview
4. Cross-project vendor and invoice intelligence
5. Project export pack

## Best unique features for Auditr

- Cross-project vendor watchlist
- Cross-project invoice similarity tracking
- Split-payment / approval-threshold review pack
- Supervisor memo generator
- Case evidence completeness scoring

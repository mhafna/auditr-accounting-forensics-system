# Auditr

Auditr is a Streamlit-based auditing workspace for accounting ledgers. It helps an auditor create projects, upload company CSV ledgers, score transactions with an XGBoost fraud model, and review why suspicious transactions were flagged through an explanation-first workflow.

## What the app does

- Creates named audit projects and stores them locally in the workspace
- Lets the auditor reopen prior projects, track status, and export a project register
- Uploads a company ledger CSV or loads demo ledgers
- Engineers audit-focused behavioral features from raw transactions
- Scores transactions with the bundled XGBoost model bundle stored under `backend/models/artifacts/`
- Surfaces high-risk entries in a manual review queue and transaction explorer
- Explains flagged transactions with auditor-facing case briefs and plain-English summaries
- Protects workspace access with a TOTP sign-in flow

## Core workflow

1. Sign in with the TOTP gate
2. Open Projects to create a project or reopen an earlier engagement
3. Return to Home for the active-project landing view and quick actions
4. Review the manual review queue and audit briefing in Dashboard
5. Filter and inspect rows in Transactions
6. Investigate flagged cases in Explainability
7. Use Help & Support for CSV guidance, payment-term definitions, and troubleshooting

## Required CSV columns

Auditr can auto-map several common aliases, but these are the canonical fields it expects:

- `transaction_id`
- `date`
- `amount`
- `vendor`
- `department`
- `account_type`
- `payment_method`
- `employee`
- `invoice_id`

At minimum, the app needs usable `date`, `amount`, and `vendor` information. If fields like `department`, `account_type`, `payment_method`, or `employee` are missing, Auditr fills them with safe placeholders so the review can still run.

## Tech stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- XGBoost

## Project structure

- `app.py` - thin root entrypoint that launches the frontend shell
- `frontend/app_shell.py` - Streamlit shell, auth gate entrypoint, and page routing
- `frontend/pages/` - all UI pages used by the auditor workflow
- `backend/utils.py` - shared backend engine for auth, project persistence, CSV parsing, feature engineering, scoring, explainability, and charts
- `backend/models/xgb_model.py` - readable Python definition of the XGBoost configuration and model-artifact loader/saver
- `backend/models/artifacts/xgb_model_bundle.pkl` - saved trained XGBoost weights plus metadata used by the live app
- `backend/models/artifacts/model_metadata.json` - exported model version, threshold, validation summary, and feature list
- `backend/models/artifacts/model_benchmark.csv` - time-based benchmark comparison across candidate models
- `backend/models/artifacts/model_feature_importance.csv` - feature importance export for the final XGBoost model
- `backend/training/generate_auditr_training_data.py` - synthetic data generator for upgraded training ledgers
- `backend/training/train_auditr_model.py` - chronological training, benchmarking, threshold selection, and model-bundle export
- `backend/demo/build_graduation_demo_pack.py` - script that builds the viva/demo CSV pack
- `auditr_workspace/` - local project workspace for saved audit projects
- `accounting_fraud_dataset_v2.csv` - upgraded labeled synthetic training dataset
- `Demo Datasets/` - demo ledgers for rehearsal/testing

## Run locally

Install the main dependencies, then start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate the upgraded training data and model:

```bash
python backend/training/generate_auditr_training_data.py --output accounting_fraud_dataset_v2.csv --rows 6000 --fraud-rate 0.12
python backend/training/train_auditr_model.py --csv-path accounting_fraud_dataset_v2.csv
```

## Authentication

Auditr supports a TOTP login gate.

- In local demo mode, the app can run with a built-in demo account and rotating OTP display.
- For deployment, set:
  - `AUDITR_AUTH_USER`
  - `AUDITR_TOTP_SECRET`

## Modeling notes

- The current explanation layer is built from XGBoost contribution values exposed by the bundled model workflow.
- `backend/models/xgb_model.py` is the readable file you can show during viva to explain the deployed XGBoost setup.
- The upgraded model uses historical, time-safe features rather than whole-file aggregates.
- Vendor and employee identity one-hot features were removed to reduce overfitting to specific names.
- The saved model bundle under `backend/models/artifacts/xgb_model_bundle.pkl` includes threshold metadata and version metadata used directly by the app.
- Validation is chronological train/validation/test, not only a random split.
- The auditor-facing UI intentionally hides most model diagnostics from the main workflow and keeps the focus on manual review actions.
- Demo ledgers are synthetic and intended for presentation/testing, not for real audit decisions.
- The app includes refresh-persistent local session behavior for the demo login flow.
- Local browser/test artifacts are excluded from git through `.gitignore`.
- Runtime project storage under `auditr_workspace/` is treated as local workspace data and is excluded from git.

## Project tracking

- Auditors can create named projects from Projects
- Each project stores its uploaded ledger locally in `auditr_workspace/projects/`
- Project metadata and register history are now stored locally in `auditr_workspace/auditr.db` via SQLite
- Projects can be reopened later from the Projects page
- Project status can be updated as the review progresses
- Project notes can be saved and updated from the Projects page
- The project register can be exported as CSV

## Additional docs

- [SPRINT_PLAN.md](./SPRINT_PLAN.md)

## Limitations

- The project uses synthetic sample data
- The model artifact is bundled and not retrained in-app
- There is no live ERP/accounting system integration
- This is a presentation/demo application, not a production audit platform

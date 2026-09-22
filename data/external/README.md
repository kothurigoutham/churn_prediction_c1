# External Dataset — Compatibility Note

This project was asked to optionally test the trained model on one external churn dataset,
used only for generalization testing (never for training).

## Decision: no external dataset was force-fitted into the model

Publicly available churn datasets (e.g. bank customer churn, e-commerce churn) were reviewed
for compatibility. None of them share the feature schema the model was trained on:

**Model's required features:**
`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges`

**Why other public churn datasets are incompatible:**
- A bank-churn dataset uses features such as `CreditScore`, `Balance`, `NumOfProducts`,
  `EstimatedSalary` — none of which correspond to a telecom contract, internet service, or
  billing feature the pipeline's `ColumnTransformer` was fit on.
- An e-commerce/subscription churn dataset typically uses features like `LastLogin`,
  `SupportTickets`, `PagesVisited` — again no overlap with telecom service attributes.

Because the trained `ColumnTransformer` (`OneHotEncoder` + `StandardScaler`) was fit on the
exact column names and category values above, passing a dataset with a different schema
would not raise a clean error in all cases — some pipelines would silently produce a
meaningless prediction (columns are missing or mismatched) rather than obviously fail. Forcing
an incompatible dataset through the model would therefore produce numbers that look like
predictions but are not statistically meaningful, so this was deliberately not done.

## What would make an external dataset usable here

An external dataset would need to supply the same 19 features (or a documented, justified
mapping from its own columns onto them) before it could be passed into
`models/churn_pipeline.pkl`. No such dataset was found during this project, so external
validation was left out rather than faked.

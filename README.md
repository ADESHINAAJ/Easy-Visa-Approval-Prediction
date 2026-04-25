# Project: Visa Application Approval Prediction

> **Context:** Project for the Ensemble Techniques module of my Applied Data Science program. EasyVisa is a simulated data-driven immigration-services firm contracted by the US Office of Foreign Labor Certification (OFLC), whose caseworkers manually review roughly 776,000 applications a year. The objective on the 25,480-row sample I worked with was to triage those applications automatically — which should be fast-tracked as certifications, which need human review, and which are almost certainly denials — across features like education level, job experience, prevailing wage, employer size, region, and position type. I used this dataset to build, compare, and tune every major tree ensemble on the syllabus: Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and Stacking, then selected and tuned the winner via GridSearchCV. The project demonstrates comfort with ensemble learning, cross-validation, hyperparameter tuning, and the discipline of comparing six competing models on a single business metric rather than cherry-picking.

---

## The problem

The US Office of Foreign Labor Certification (OFLC) processes nearly 776,000 visa applications annually for foreign workers seeking temporary or permanent employment. The agency must verify that US employers have demonstrated insufficient domestic workers available before certifying a foreign worker. This manual review process is becoming a bottleneck as application volumes grow year-over-year. EasyVisa, a data-driven immigration solutions firm, was tasked with building a predictive model to automatically shortlist candidates with the highest probability of visa approval, reducing processing time and resource allocation while providing actionable guidance on which candidate profiles are most likely to succeed. The stakes are high: incorrectly certifying an unqualified candidate wrongly displaces US workers; incorrectly denying a qualified candidate causes the US economy to lose productive talent.

---

## What I built

Built an ensemble classification model using tuned Random Forest that predicts visa certification outcomes with F1 score of 0.84 on training data and 0.82 on test data. Trained on 25,480 visa applications across 11 features including employee education level, job experience, wage information, employment region, continent of origin, company size, and company establishment year. Conducted extensive ensemble benchmarking—Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and Stacking—across hyperparameter configurations. The tuned Random Forest emerged as the production model, achieving balanced precision (0.83) and recall (0.81) with balanced class weights to handle the 66.8% majority class (certified visas). Identified education level, job experience, and prevailing wage as the top three feature importance drivers.

---

## Technical choices (and why)

- **Ensemble models over single classifiers**: Decision trees alone overfit severely (100% training accuracy, 68% test F1). Ensembling through bagging (Random Forest) and boosting (XGBoost, Gradient Boosting) reduced overfitting and improved generalization. Random Forest's default bagging mechanism naturally handles the class imbalance better than single-tree approaches.
- **Balanced class weights**: Dataset had 66.8% certified vs. 33.2% denied. Applied balanced class weights to penalize false negatives and false positives equally, preventing the model from simply predicting "certified" for everything.
- **F1 score as optimization metric**: F1 balances precision and recall. Incorrectly certifying an unqualified candidate and incorrectly denying a qualified candidate have roughly equal business costs, so F1 was appropriate rather than accuracy or AUC alone.
- **Hyperparameter tuning strategy**: Tested max_depth, min_samples_leaf, n_estimators for Random Forest; learning_rate and n_estimators for boosting models. Random Forest with max_depth=15, min_samples_leaf=5 achieved the best generalization (0.82 test F1) without excessive complexity.
- **Feature importance analysis**: Random Forest feature importances revealed education, job experience, and prevailing wage as top drivers. Used these for business rule extraction (e.g., "Doctorate degree holders have 85% approval rate").

---

## The outcome

- **Predictive performance**: F1 score of 0.84 on training, 0.82 on test with precision 0.83 and recall 0.81, indicating balanced discrimination between approved and denied cases.
- **Key success factors identified**: Education is the strongest predictor—85% of Doctorate applicants, 80% of Master's degree applicants, and 60% of Bachelor's applicants get certified. Job experience alone correlates with 80% approval rate vs. 60% for inexperienced workers.
- **Wage signal**: Certified applicants have median prevailing wage of $72k vs. $65k for denied applicants. Yearly wage units (90% of data) have 75% approval rate vs. only 35% for hourly units, suggesting more stable, full-time positions are preferred.
- **Geographic insights**: Applications from Europe have 80% approval, Africa ~75%, Asia ~60%, with different regional talent needs (e.g., Doctorate candidates mostly needed in West region).
- **Business-actionable criteria**: Created distinct profiles—for approval prioritize: Bachelor's minimum (Master's/Doctorate preferred), job experience, yearly wage unit; for denial flag: high school only, no experience, hourly wage unit.

---

## What failed or what you tried

- **Initial Decision Tree without pruning**: Achieved 100% training accuracy but only 68% test F1 with 0% test precision on one class, completely overfitting. Needed ensemble methods and hyperparameter constraints.
- **Basic Bagging without tuning**: Bagging reduced overfitting relative to single trees but still showed ~5% gap between training and test recall. Random Forest's feature subsampling performed better.
- **XGBoost without hyperparameter tuning**: First XGBoost model overfit badly (0.95 train F1 vs. 0.75 test F1). Sequential boosting on the same features repeatedly caused instability. Required careful tuning of learning_rate (0.05-0.1), max_depth (3-5), and early stopping.
- **Stacking with mismatched base learners**: Initial stacking used too many heterogeneous learners which created noise. Final stacking used 3 well-tuned base models (RF, XGB, Gradient Boosting) with Logistic Regression as meta-learner, achieving F1 0.81 test but less stable than Random Forest.
- **Ignoring data quality**: Initially found negative employee counts (-26 in min), which represented data entry errors. Took absolute values to fix. Prevailing wage had hourly values (some <$100/year equivalent), which required unit-aware analysis rather than filtering as outliers.

---

## Production and scale thinking

- **Deployment model**: Real-time REST API deployed to internal OFLC network. Input: applicant profile (education, experience, wage, region). Output: certification probability score + top 3 reasons for approval/denial based on feature importance.
- **Volume & latency**: OFLC processes ~2100 applications per day (776k/year). Model must score each in <50ms. Random Forest with tuned depth (15) and ~100 trees easily meets this requirement.
- **Monitoring**: Track prediction drift by comparing actual approval rates (reported monthly) to model predictions. If approval rate shifts >5% from historical trend, retrain on recent data (quarterly retraining window recommended).
- **Fairness & compliance**: Monitor approval rates by continent and education level to detect demographic bias. Current data shows legitimate variation (Europe 80% vs. Asia 60%) but should validate with legal/compliance teams before production.
- **Maintenance & retraining**: Retrain quarterly on rolling 2-year dataset to capture changes in US labor market demands and international talent pools. Recalibrate feature importance thresholds annually.

---

## Links

- GitHub: [link to repository if applicable]
- Notebook: [link to full analysis notebook]

---

## Executive Summary

> I built an ensemble classification model using tuned Random Forest that predicts visa approval outcomes with 82% F1 score for a US immigration processing firm. By identifying education level, job experience, and prevailing wage as the primary approval drivers, we enabled them to shortlist qualified candidates automatically—reducing processing time by estimating that top-profile applicants (Master's degree + experience + $72k+ wage) have 80%+ approval probability—while providing objective justification for approval or denial decisions.

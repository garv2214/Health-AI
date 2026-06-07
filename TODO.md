# TODO - Emergency predictor training & accuracy improvements

- [ ] Fix dataset generation runtime issues (done: numpy install + pain_level probability normalization; still failing: missing `data/` directory)
- [ ] Create `data/` directory or update dataset script to save into an existing path
- [ ] Run `python3 data_models/create_dataset.py` and confirm `medical_emergency_dataset.csv` exists
- [ ] Run `python3 health_ai_module/predictive_analytics/train_model.py`
- [ ] Report exactly how much training happened (samples, train/test split, models trained)
- [ ] Report metrics achieved (accuracy/precision/recall/f1/roc-auc) and which model was selected
- [ ] Provide specific next-step training improvements based on results


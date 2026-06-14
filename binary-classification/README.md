Experiment Tracking Framework Features
---------------------------------------
✅ Run multiple experiments from the same program
✅ Each experiment has a name
✅ Log all hyperparameters used:
Model type (MLP/CNN/etc.)
Epochs
Learning rate
Normalization used
Data augmentation used
Batch size
✅ Store results in a file in the same directory (JSON or CSV)
✅ Save:
Accuracy
Loss
Training time
Timestamp
✅ Automatically generate a comparison report at the end
✅ Rank experiments from best to worst
✅ Show improvement over baseline
✅ Allow you to add new experiments later by just adding one configuration block
==================================================
FINAL COMPARISON
==================================================
MLP Accuracy                                         : 61.90%
CNN Accuracy                                         : 74.25%
CNN + Normalization Accuracy + Epoch increased to 75 : 75.65%

CNN Improvement              : 12.35%
Normalization Improvement    : 1.40%
Overall Improvement          : 13.75%
==================================================
Generating report...
                        name  accuracy  epochs  learning_rate  augmentation  normalization  improvement
2  CNN + Norm + Augmentation     75.45      30          0.001          True           True         4.92
1        CNN + Normalization     70.87      30          0.001         False           True         0.34
0               CNN Baseline     70.53      30          0.001         False          False         0.00
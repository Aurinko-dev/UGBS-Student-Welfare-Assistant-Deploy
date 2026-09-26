This folder is empty until you run:

    python train_classifier.py

That trains the category/severity classifier on data/welfare_training_examples.csv
and saves two files here:

    welfare_classifier.pt          (trained PyTorch model weights)
    welfare_classifier_labels.json (category/severity label mapping)

The app checks for these files at startup — if they're missing, it falls
back to the rule-based classifier in risk_classifier.py so the app still
runs, just with less accurate/nuanced classification.

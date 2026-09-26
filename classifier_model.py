"""
The classifier itself: a small feedforward neural network sitting on top of
frozen sentence embeddings (all-MiniLM-L6-v2, the same encoder already used
for retrieval — no extra model to load).

    384-dim embedding
          │
     Linear(384 -> 128) + ReLU + Dropout      <- hidden layer, learns a
          │                                      task-specific representation
          ├──> Linear(128 -> num_categories)  <- category head
          └──> Linear(128 -> num_severities)  <- severity head

This is a direct, deliberately simple application of the neuron / weighted
sum / activation function / multi-output-layer material from the neural
networks lecture: one shared hidden layer, two output heads (the "different
problems need different output layer designs" idea — here two classification
problems share one representation).

Severity here only ever predicts Low/Medium/High. Critical is never learned
— it stays the exclusive responsibility of the deterministic keyword check
in risk_classifier.py, on purpose (see that file's docstring).
"""
import torch
import torch.nn as nn

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size
HIDDEN_DIM = 128


class WelfareClassifierNet(nn.Module):
    def __init__(self, num_categories: int, num_severities: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.category_head = nn.Linear(HIDDEN_DIM, num_categories)
        self.severity_head = nn.Linear(HIDDEN_DIM, num_severities)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.category_head(h), self.severity_head(h)

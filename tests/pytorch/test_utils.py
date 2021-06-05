import torch
import torch.nn as nn

class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extraction = nn.Sequential(
                nn.Conv2d(24, 48, 3),
                nn.ReLU(),
                nn.Conv2d(46, 24, 3),
                nn.ReLU(),
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(24*3*3, 4)
            )
            
        def forward(self, x):
            x = self.feature_extraction(x)
            out = self.classifier(x)
            
            return out
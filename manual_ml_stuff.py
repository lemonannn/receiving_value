import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NonMonotoneMultiplicativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # multiplicative blocks
        self.f1 = SimpleMLP(1)
        self.f2 = SimpleMLP(1)
        self.f3 = SimpleMLP(1)
        self.f123_outer = SimpleMLP(1)
        
        self.f4 = SimpleMLP(1)
        self.f5 = SimpleMLP(1)
        self.f6 = SimpleMLP(1)
        self.f456_outer = SimpleMLP(1)
        
        self.f7 = SimpleMLP(1)
        self.f8 = SimpleMLP(1)
        self.f9 = SimpleMLP(1)
        self.f10 = SimpleMLP(1)
        self.f910_outer = SimpleMLP(1)
        
        self.f11 = SimpleMLP(1)
        self.f12 = SimpleMLP(1)
        
    def forward(self, x):
        x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12 = torch.chunk(x, 12, dim=1)
        
        term123 = self.f123_outer(self.f1(x1) * self.f2(x2) * self.f3(x3))
        term456 = self.f456_outer(self.f4(x4) * self.f5(x5) * self.f6(x6))
        term910 = self.f910_outer(self.f9(x9) * self.f10(x10))
        
        simple_terms = self.f7(x7) * self.f8(x8)
        
        output = term123 * term456 * simple_terms * term910 + self.f11(x11) - self.f12(x12)
        return output
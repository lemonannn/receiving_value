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
        
        self.f_los_yprr = SimpleMLP(1)
        self.f_los_rec = SimpleMLP(1)
        self.f_behind_los = SimpleMLP(1)
        
        self.f_short_yprr = SimpleMLP(1)
        self.f_short_rec = SimpleMLP(1)
        self.f_short = SimpleMLP(1)
        
        self.f_med_yprr = SimpleMLP(1)
        self.f_med_rec= SimpleMLP(1)
        self.f_medium = SimpleMLP(1)
        
        self.f_deep_yprr = SimpleMLP(1)
        self.f_deep_rec = SimpleMLP(1)
        self.f_deep = SimpleMLP(1)
        
        self.f_yprr = SimpleMLP(1)
        
        self.f_pow4 = SimpleMLP(1)
        self.f_pow4_2 = SimpleMLP(1)
        self.f_pow4_3 = SimpleMLP(1)
        self.f_pow4_4 = SimpleMLP(1)
        
        self.f_drop_rate = SimpleMLP(1)
        self.f_wide_rate = SimpleMLP(1)
        
        self.f11 = SimpleMLP(1)
        self.f12 = SimpleMLP(1)
        
        self.f_prod = SimpleMLP(1)
        self.f_efficency = SimpleMLP(1)
        
    def forward(self, x):
        los_yards,short_yards,med_yards,deep_yards,yprr,pow4,drop_rate,wide_rate = torch.chunk(x, 12, dim=1)
        
        behind_los = self.f_behind_los(los_yards)
        short = self.f_short(self.f_short_yprr(short_yards))
        medium = self.f_medium(self.f_med_yprr(med_yards))
        deep = self.f_deep(self.f_deep_yprr(deep_yards))
        
        output = self.f_prod(behind_los + short + medium + deep + self.f_yprr(yprr)) * self.f_pow4(pow4) + self.f_drop_rate(drop_rate) + self.f_wide_rate(wide_rate)
        return output
import torch.nn as nn
import torch.nn.functional as F

class LSM(nn.Module):
    def __init__(self):
        super(LSM, self).__init__()

        self.fc = nn.Linear(65, 1, bias=False)
        self.relu = nn.ReLU()



    def forward(self, x, x1):
        
        x_fc = self.fc(x1.permute(0, 2, 3, 1))
        x_fc = x_fc.permute(0, 3, 1, 2)
        
        x_fc = self.relu(x_fc)
        x_fc = F.sigmoid(x_fc)
        x1 = x * x_fc
        x = x - 0.5 * x1

        return x, x_fc



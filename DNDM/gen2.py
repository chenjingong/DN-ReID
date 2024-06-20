import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Generator(nn.Module):
    def __init__(self, input_nc):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.relu = nn.ReLU(True)

    def average_pool_group(self, x):
        batch_size, c, h, w = x.size()
        x_grouped = x.view(batch_size // 4, 4, c, h, w)
        averaged_group = (x_grouped + 1e-8).mean(dim=1, keepdim=True)
        averaged_group_expanded = averaged_group.expand(batch_size // 4, 4, c, h, w)
        out = averaged_group_expanded.reshape(batch_size, c, h, w)
        return out

    def max_pool_response(self, x, x1):
        
        result_gap = x1

        result_gap = result_gap.permute(0, 3, 1, 2)

        _, _, h, w = result_gap.size()
        pooled_result_gap = F.avg_pool2d(result_gap, kernel_size = h)
        _, max_indices_gap = torch.max(pooled_result_gap.squeeze(-1).squeeze(-1), dim=1)
        max_values_gap = result_gap[torch.arange(result_gap.size(0)), max_indices_gap, :, :]
        
        max_values_gap_out = max_values_gap.unsqueeze(1).expand(-1, 1, h, w)
        max_values_gap_out = self.relu(max_values_gap_out)
        max_values_gap_out = F.sigmoid(max_values_gap_out)
        multiplied_result_gap = max_values_gap_out*x
        
        out = 0.1*multiplied_result_gap  + x
        max_values_out = max_values_gap_out

        return out, max_values_out
        

    def forward(self, x, x1, y):

        result1, x_max = self.max_pool_response(x, x1)
        if y:
            x1 = x_max[:32]
            x2 = x_max[32:]
        
            day1 = self.average_pool_group(x1)
            night1 = self.average_pool_group(x2)
        
            x3 = x[:32]
            x4 = x[32:]
        
            day = x3 + 0.05 * x3 * night1 + 0.05 * x1 * x3
            night = x4 + 0.05 * x4 * day1 + 0.05 * x2 * x4
             
            out = torch.cat([day, night], dim=0)
        else:
            out=result1
        return out

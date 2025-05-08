import torch
import torch.nn as nn

class CMML3(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(CMML3, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y, z):
        x_attention = self.softmax(x.clone())
        y_attention = self.softmax(y.clone())
        z_attention = self.softmax(z.clone())
        sum_total = x_attention + y_attention + z_attention
        x_attention = x_attention/sum_total
        y_attention = y_attention/sum_total
        z_attention = z_attention/sum_total

        supervise_feature_hidden = x_attention * x + y_attention * y + z_attention * z
        output = self.fc_out(supervise_feature_hidden)
        output_x = self.fc_out(x)
        output_y = self.fc_out(y)
        output_z = self.fc_out(z)
        return output_x, output_y, output_z, output

class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output
    
class SumFusion3(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion3, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)
        self.fc_z = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, z):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)
        out_z = self.fc_z(z)
        output = out_x + out_y + out_z
        return out_x, out_y, out_z, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output

class ConcatFusion3(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion3, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)
        
        weight_size = self.fc_out.weight.size(1)
        
        out_x = (torch.mm(x, torch.transpose(self.fc_out.weight[:, :weight_size // 3], 0, 1))
                    + self.fc_out.bias / 3)
        
        out_y = (torch.mm(y, torch.transpose(self.fc_out.weight[:, weight_size // 3:2*weight_size // 3], 0, 1))
                    + self.fc_out.bias / 3)

        out_z = (torch.mm(z, torch.transpose(self.fc_out.weight[:, 2*weight_size // 3:], 0, 1))
                    + self.fc_out.bias / 3)

        output = self.fc_out(output)
        return out_x, out_y, out_z, output

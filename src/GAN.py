from torch import nn

class Generator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,output_dim),
        )
    
    def forward(self,x):
        x = self.layer(x)
        # return th.clamp(x,0.,1.)
        return x
    
    def generate(self):
        pass


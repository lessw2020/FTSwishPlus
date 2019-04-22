#original FTSwish = https://arxiv.org/abs/1812.06247

#import torch.nn.functional as F  (uncomment if needed,but you likely already have it)

class FTSwishPlus(nn.Module):
    def __init__(self, threshold=-.25, mean_shift=-.1):
        super().__init__()
        self.threshold = threshold
        self.mean_shift = mean_shift

    def forward(self, x): 

        x = F.relu(x) * torch.sigmoid(x) + self.threshold
        #note on above - why not F.sigmoid?: 
        #PyTorch docs - "nn.functional.sigmoid is deprecated. Use torch.sigmoid instead."

        #apply mean shift to drive mean to 0. -.1 was tested as optimal for kaiming init
        if self.mean_shift is not None:
            x.sub_(self.mean_shift)

        return x
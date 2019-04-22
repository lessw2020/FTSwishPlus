#original FTSwish = https://arxiv.org/abs/1812.06247

class FTSwishPlus(nn.Module):
    def __init__(self, threshold=-.25, mean_shift=-.1):
        super().__init__()
        self.threshold = threshold
        self.mean_shift = mean_shift
        #warning - does not handle multi-gpu case below
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


    def forward(self, x): 
        
        #FTSwish+ for positive values
        pos_value = (x*torch.sigmoid(x)) + self.threshold
        
        #FTSwish+ for negative values
        tval = torch.tensor([self.threshold],device=self.device)
        
        #apply to x tensor based on positive or negative value
        x = torch.where(x>=0, pos_value, tval)
        
        #apply mean shift to drive mean to 0. -.1 was tested as optimal for kaiming init
        if self.mean_shift is not None:
            x.sub_(self.mean_shift)

        return x

import math
import torch
from torch import nn

    
# Convolutional Networks
        
class Conv_Net(nn.Module):
    
    def __init__(self, name:str, L:int, M:int=32):
        super(Conv_Net, self).__init__()
        
        self.L = L
        self.M = M
        self.name = name
        self.act = nn.ReLU(inplace=True)    
        
        self.V = nn.Conv2d(3, self.M, 8, stride=1, padding=3)
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           
        
        self.W = nn.ModuleList(                                 
            [nn.Conv2d(self.M,self.M,3, padding=1) for _ in range(self.L)])
                
        self.C = nn.Linear(8*8*self.M, 10)
        
        # NOT FOLLOWING PAPER Custom Initialization
        '''
        NOTES:
            [0] - This has been changed from : for param, name in ... -> Review if doesn't work
            [1] - Read on DL book that with RELU is preferable to start biases with 0.01
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
#                    m.bias.data.zero_() ## Notes (1)
                    m.bias.data.fill_(0.01)
                    
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
#                m.bias.data.zero_() 
                m.bias.data.fill_(0.01)
                
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                                
    def forward(self, x):
        
        x = self.act(self.V(x))         # Out: 32x32xM  
        x = self.P(x)                   # Out: 8x8xM  
        for w in self.W:
            x = self.act(w(x))          # Out: 8x8xM  
        x = x.view(x.size(0), -1)       # Out: 64*M  (M = 32 -> 2048)
        return self.C(x)



class Conv_Recusive_Net(nn.Module):
    
    def __init__(self, name:str, L:int, M:int=32):
        super(Conv_Recusive_Net, self).__init__()
        
        self.L = L
        self.M = M
        self.name = name
        self.act = nn.ReLU(inplace=True)

        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM  
        
        self.W = nn.Conv2d(self.M,self.M,3, padding=1)          # Out: 8x8xM 
        
        self.C = nn.Linear(8*8*self.M, 10)
        
        # Custom Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
#                    m.bias.data.zero_()
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                        
    
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        for w in range(self.L):
            x = self.act(self.W(x))
        x = x.view(x.size(0), -1)
        return self.C(x)  
   
    
    
class Conv_Custom_Recusive_Net(nn.Module):
    
    def __init__(self, name:str, L:int, M:int=32, F:int=32):
        super(Conv_Custom_Recusive_Net, self).__init__()
        
        self.L = L
        self.M = M
        self.F = F
        self.name = name
        self.act = nn.ReLU(inplace=True)

        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM  
        
        self.W = nn.Conv2d(self.M,self.M,3, padding=1)          # Out: 8x8xM 
        self.WL = nn.Conv2d(self.F,self.F,3,padding=1)          # Out: 8x8xF
        
        self.C = nn.Linear(8*8*self.F, 10)
        
        # Custom Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
#                    m.bias.data.zero_()
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                        
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        for w in range(self.L - 1):     # Up to layer L not included
            x = self.act(self.W(x))
        x = self.act(self.WL(x))        # Last convolution use WL with F filters
        x = x.view(x.size(0), -1)
        return self.C(x)  
    


class Conv_K_Recusive_Net(nn.Module):
    '''
    Recursive block of K layers
    '''
    def __init__(self, name:str, L:int, M:int=32, K:int=2):
        super(Conv_K_Recusive_Net, self).__init__()
        
        self.K = K
        self.L = L
        self.M = M
        self.name = name
        self.act = nn.ReLU(inplace=True)

        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM 
        
#        self.W = nn.Conv2d(self.M,self.M,3, padding=1)          # Out: 8x8xM  
        self.Wk = nn.ModuleList(                                 
            [nn.Conv2d(self.M,self.M,3,1) for _ in range(self.K)])
        
        self.C = nn.Linear(8*8*self.M, 10)
        
        # Custom Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
#                    m.bias.data.zero_()
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                        
        
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        
        for block in range(int(self.L/self.K)):     # num_blocks = num_layers / layers_per_block
            for W in self.Wk:                       # for each layer in the block
                x = self.act(W(x))

        x = x.view(x.size(0), -1)
        return self.C(x) 



if '__name__' == '__main__':
    
    from torch.autograd import Variable
    def test(net):
        y = net(Variable(torch.randn(1,3,32,32)))
        print(y.size())
    
    L = 16
    M = 32
    K = 2
    F = 16
    
    convnet = Conv_Net('ConvNet', L, M)
    r_convnet = Conv_Recusive_Net('RecursiveConvNet', L, M)
    r_convnet_k = Conv_K_Recusive_Net('Custom_Recursive_ConvNet', L, M, K)
    r_convnet_c = Conv_Custom_Recusive_Net('Custom_Recursive_ConvNet', L, M, F)

    test(convnet)
    test(r_convnet)
    test(r_convnet_k)
    test(r_convnet_c)    
    
    exit()
    
    
## Fully Connected Networks
#
#class FC_Net(nn.Module):
#    ''' Fully Connected Network '''
#    
#    def __init__(self, name:str, inp:int, out:int, hid:int):
#        super(FC_Net, self).__init__()
#                
#        self.name = name
#        self.lay_size = hid
#        self.act = nn.ReLU()
#        
#        self.fcI = nn.Linear(inp, hid, bias=True)        
#        self.fcH = nn.Linear(hid, hid, bias=True)   
#        self.fcO = nn.Linear(hid, out, bias=True)
#                
#    def forward(self, x):
#                
#        x = self.act(self.fcI(x))        
#        x = self.act(self.fcH(x))            
#        return self.fcO(x)
#    
#
#class FC_Recursive_Net(nn.Module):
#    ''' Fully Connected Network with Recursivity '''
#    
#    def __init__(self, name:str, inp:int, out:int, hid:int, rec:int):
#        super(FC_Recursive_Net, self).__init__()
#    
#        self.name = name
#        self.n_lay = rec        
#        self.act = nn.ReLU()
#        assert rec > 0, 'Recursive parameters must be >= 1'
#        
#        self.fcI = nn.Linear(inp, hid, bias=True)        
#        self.fcH = nn.Linear(hid, hid, bias=True) 
#        self.fcO = nn.Linear(hid, out, bias=True)
#                
#    def forward(self, x):
#                
#        x = self.act(self.fcI(x))        
#        x = self.act(self.fcH(x))
#        for l in range(self.n_lay): 
#            x = self.actf(self.fcHid(x))            
#        return self.fcO(x)

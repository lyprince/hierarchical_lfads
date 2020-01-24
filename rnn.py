import torch
import torch.nn as nn

class LFADS_GRUCell(nn.Module):
    
    '''
    LFADS_GRUCell class. Implements the Gated Recurrent Unit (GRU) used in LFADS Encoders. More obvious
    relation to the equations (see https://en.wikipedia.org/wiki/Gated_recurrent_unit), along with
    a hack to help learning
    
    __init__(self, input_size, hidden_size, forget_bias=1.0)
    
    required arguments:
     - input_size (int) : size of inputs
     - hidden_size (int) : size of hidden state
     
    optional arguments:
     - forget_bias (float) : hack to help learning, added to update gate in sigmoid
    '''
    
    def __init__(self, input_size, hidden_size, forget_bias=1.0):
        super(LFADS_GRUCell, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        
        # Concatenated sizes
        self._xh_size = input_size + hidden_size
        self._ru_size = hidden_size * 2
        
        # r, u = W([x, h]) + b
        self.fc_xh_ru = nn.Linear(in_features= self._xh_size, out_features= self._ru_size)
        # c = W([x, h*r]) + b
        self.fc_xhr_c = nn.Linear(in_features= self._xh_size, out_features= self.hidden_size)
        
    def forward(self, x, h):
        '''
        Forward method - Gated Recurrent Unit forward pass with forget bias
        
        forward(self, x, h):
        
        required arguments:
          - x (torch.Tensor) : GRU input
          - h (torch.Tensor) : GRU hidden state
        
        returns
          - h_new (torch.Tensor) : updated GRU hidden state
        '''
        
        # Concatenate input and hidden state
        xh  = torch.cat([x, h], dim=1)
        
        # Compute reset gate and update gate vector
        r,u = torch.split(self.fc_xh_ru(xh),
                          split_size_or_sections=self.hidden_size,
                          dim = 1)
        r,u = torch.sigmoid(r), torch.sigmoid(u + self.forget_bias)
        
        # Concatenate input and hadamard product of hidden state and reset gate
        xrh = torch.cat([x, torch.mul(r, h)], dim=1)
        
        # Compute candidate hidden state
        c   = torch.tanh(self.fc_xhr_c(xrh))
        
        # Return new hidden state as a function of update gate, current hidden state, and candidate hidden state
        return torch.mul(u, h) + torch.mul(1 - u, c)
    
class LFADS_GenGRUCell(nn.Module):
    '''
    LFADS_GenGRUCell class. Implements gated recurrent unit used in LFADS generator and controller. Same as
    LFADS_GRUCell, but parameters transforming hidden state are kept separate for computing L2 cost (see 
    bullet point 2 of section 1.9 in online methods). Also does not create parameters transforming inputs if 
    no inputs exist.
    
    __init__(self, input_size, hidden_size, forget_bias=1.0)
    '''
    
    def __init__(self, input_size, hidden_size, forget_bias=1.0):
        super(LFADS_GenGRUCell, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        
        # Concatenated size
        self._ru_size    = self.hidden_size * 2
        
        # Create parameters for transforming inputs if inputs exist
        if self.input_size > 0:
            
            # rx ,ux = W(x) (No bias in tensorflow implementation)
            self.fc_x_ru = nn.Linear(in_features= self.input_size, out_features= self._ru_size, bias=False)
            # cx = W(x) (No bias in tensorflow implementation)
            self.fc_x_c  = nn.Linear(in_features= self.input_size, out_features= self.hidden_size, bias=False)
        
        # Create parameters transforming hidden state
        
        # rh, uh = W(h) + b
        self.fc_h_ru = nn.Linear(in_features= self.hidden_size, out_features= self._ru_size)
        # ch = W(h) + b
        self.fc_rh_c = nn.Linear(in_features=self.hidden_size, out_features= self.hidden_size)
        
    def forward(self, x, h):
        '''
        Forward method - Gated Recurrent Unit forward pass with forget bias, weight on inputs and hidden state kept separate.
        
        forward(self, x, h):
        
        required arguments:
          - x (torch.Tensor) : GRU input
          - h (torch.Tensor) : GRU hidden state
        
        returns
          - h_new (torch.Tensor) : updated GRU hidden state
        '''
        
        # Calculate reset and update gates from input
        if self.input_size > 0 and x is not None:
            r_x, u_x = torch.split(self.fc_x_ru(x),
                                   split_size_or_sections=self.hidden_size,
                                   dim = 1)
        else:
            r_x = 0
            u_x = 0
        
        # Calculate reset and update gates from hidden state
        r_h, u_h = torch.split(self.fc_h_ru(h),
                               split_size_or_sections=self.hidden_size,
                               dim = 1)
        
        # Combine reset and updates gates from hidden state and input
        r = torch.sigmoid(r_x + r_h)
        u = torch.sigmoid(u_x + u_h + self.forget_bias)
        
        # Calculate candidate hidden state from input
        if self.input_size > 0 and x is not None:
            c_x = self.fc_x_c(x)
        else:
            c_x = 0
        
        # Calculate candidate hidden state from hadamard product of hidden state and reset gate
        c_rh = self.fc_rh_c(r * h)
        
        # Combine candidate hidden state vectors
        c = torch.tanh(c_x + c_rh)
        
        # Return new hidden state as a function of update gate, current hidden state, and candidate hidden state
        return u * h + (1 - u) * c
    
    def hidden_weight_l2_norm(self):
        return self.fc_h_ru.weight.norm(2).pow(2)/self.fc_h_ru.weight.numel() + self.fc_rh_c.weight.norm(2).pow(2)/self.fc_rh_c.weight.numel()
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import weakref
import torch.nn.functional as F
import torch.nn  as nn
from torch.jit   import script

from src.shared_state import SharedState
from src.models.model import BaseModel
from src.models.transformer.stoch_tab.helper import *
from src.models.transformer.stoch_tab.lwta import LWTA


class GlobalResnet(nn.Module):
    def __init__(self, dim, total_f,dropout=0.2,U=2):
        super().__init__()
        
       
        self.ln1     = LocalLinear(dim,1,ftnum=total_f)
        self.ln2     = nn.Linear(total_f,4*dim)
        self.ln3     = nn.Linear(4*dim,dim)
        
       
        self.lwta    = LWTA(U)
       
        self.norm    = nn.LayerNorm(total_f)
        self.norm2   = nn.LayerNorm(dim)
       
        self.drop1   = Dropout(dropout)
        self.drop2   = Dropout(dropout)
       
        


    def forward(self, x):
        
        
        
     
        x = self.ln1(x)
       
        x = rearrange(x, 'b n 1 -> b n')
       
        x = self.norm(x)
        x = self.drop1(x)
        x = self.ln2(x)
        x = self.lwta(x)
        x = self.ln3(x)
        
        x = rearrange(x, 'b n -> b 1 n')
        
        x = self.norm2(x)
        o = self.drop2(x)
         
        return o



def FeedForward(dim, mult = 4, dropout = 0.,U=2):
   
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        LWTA(U),
        Dropout(dropout),
        nn.Linear(dim * mult, dim)
        
    )



        
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        ftnum = 10
    ):
        super().__init__()
       
        inner_dim    = dim_head * heads
        self.heads   = heads
        self.scale   = dim_head ** -0.5        
        self.norm    = nn.LayerNorm(dim)
        self.rezero0 = nn.Parameter(torch.zeros(1,))
        self.rezero  = nn.Parameter(torch.zeros(1,))
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out  = nn.Linear(inner_dim, dim, bias = False)
        
        self.dropout  = Dropout(dropout)
        
        self.attention_bias    =  nn.Parameter(torch.zeros((1,heads, ftnum,ftnum) ))
       
      
        
        
    def forward(self, x):
        h = self.heads
       
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        q = q * self.scale
        

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim +self.attention_bias 

       
        attn          = sim.softmax(dim = -1)
        dropped_attn  = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)       
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        ftnum,
        U=2
    ):
        super().__init__()
        self.layers   = nn.ModuleList([])
        self.norm2    = nn.LayerNorm(dim)
        self.res_drop = Dropout(0.1)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout,ftnum=ftnum),
                FeedForward(dim, dropout = ff_dropout,U=U),
                GlobalResnet(dim,total_f=ftnum,U=U,dropout=ff_dropout),
            ]))
            

    def forward(self, x, return_attn = False):
        post_softmax_attns = []
        
        for attn, ff,globres  in self.layers:
            
            attn_out, post_softmax_attn= attn(x)
            post_softmax_attns.append(post_softmax_attn)
            
            # Running Parralel Agr Module
            dx  = globres(x)
       
           
            # Main Tranformer Module 
            x = attn_out + x
            x = ff(x) + x
            
            ## Adding Parralel Agr Module
            x[:,:1] = dx+x[:,:1]
            

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder
class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types,cases):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, cases))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, cases))
        self.w = nn.Parameter(torch.randn(num_numerical_types,cases, dim))
        self.b = nn.Parameter(torch.randn(num_numerical_types,cases, dim))
        self.cases=cases
        self.lwta=LWTA(cases,kl_w=1,return_mask=True,T=0.69)
       
    def forward(self, x):
       
        x = rearrange(x, 'b n -> b n 1')
        
        x=x * self.weights + self.biases
        
        if self.cases>1:
            #inderect use of lwta in for global competition
            x,mask=self.lwta(x)
            mask =mask.transpose(0, 1) 
            x = x.transpose(0, 1) 
            b = torch.bmm(mask,self.b).transpose(0, 1)   #av
            x = torch.bmm(x,self.w)
            x = x.transpose(0, 1)  
        
            y=x+b
        else:
            x = x.transpose(0, 1) 
            x = torch.bmm(x,self.w)
            x = x.transpose(0, 1)  
            b = torch.squeeze(self.b, -2)
            y=x+b

 
        return y

# main class
class STabModel(BaseModel):
    def __init__(
        self,
        configs_path: str,
        state: SharedState,
        device: str = 'cpu',
    ):
        super().__init__(configs_path, state, device)
        categories = state.num_categories
        num_continuous = state.num_continious
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        dim = self._configs.get("dim", 32)
        depth = self._configs.get("depth", 6)
        heads = self._configs.get("heads", 8)
        dim_head = self._configs.get("dim_head", 32)
        dim_out = self._configs.get("dim_out", 1)
        num_special_tokens = self._configs.get("num_special_tokens", 2)
        attn_dropout = self._configs.get("attn_dropout", 0.125)
        ff_dropout = self._configs.get("ff_dropout", 0.125)
        U = self._configs.get("U", 2)
        cases = self._configs.get("cases", 16)
            
        # categories related calculations
        self.device = device
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous
        
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous,cases)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.ftnum=self.num_categories+self.num_continuous+1
        
        # transformer
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
           
            ftnum=self.ftnum,
            U=U
        )

        self.to_logits = nn.Sequential(
            Dropout(ff_dropout),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x, return_attn = False):
        x_numer, x_categ = x
        xs = []
        
        if self.num_unique_categories > 0:
            x_categ += self.categories_offset
            
            x_categ = self.categorical_embeds(x_categ)#*(x_categ > 0).float().unsqueeze(-1)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical
        if  self.num_unique_categories >0:
            x = torch.cat(xs, dim = 1)
        else:
            x=x_numer
        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        
        # attend
        x, attns = self.transformer(x, return_attn = True)
        # get cls token
        x = x[:, 0]

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns

if __name__ == "__main__":
    import sys
    x_categ = torch.randint(0, 5, (1, 5))
    x_cont = torch.randn(1, 10)     

    st = SharedState()
    st.num_categories = (10, 5, 6, 5, 8)
    st.num_continious = 10

    c = sys.argv[1]
    model = STabModel(c, st)

    pred = model((x_cont, x_categ))
    print(pred)
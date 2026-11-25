"""Station-domain versions of the two SMF-ARCH-DESIGN models.

Input tensors are hourly windows over sites, not dense satellite images. Spatial
attention operates over site tokens; this is not a dense-field benchmark.
"""
import torch
from torch import nn
import torch.nn.functional as F


class SMFDeterministic(nn.Module):
    """Multivariate, spatial/temporal deterministic trunk with site interaction."""
    def __init__(self, features, sites, coords, history=72, horizon=24, width=128, heads=4,
                 use_gnn=True,use_spatial_transformer=True,use_temporal_transformer=True):
        super().__init__(); self.sites=sites; self.horizon=horizon; self.width=width
        self.inp=nn.Linear(features,width); self.site=nn.Embedding(sites,width)
        self.coord=nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,width))
        self.clock=nn.Sequential(nn.Linear(6,width),nn.GELU(),nn.Linear(width,width))
        sl=nn.TransformerEncoderLayer(width,heads,4*width,batch_first=True,activation='gelu',norm_first=True)
        tl=nn.TransformerEncoderLayer(width,heads,4*width,batch_first=True,activation='gelu',norm_first=True)
        self.spatial=nn.TransformerEncoder(sl,2) if use_spatial_transformer else nn.Identity()
        self.temporal=nn.TransformerEncoder(tl,2) if use_temporal_transformer else nn.Identity()
        self.use_temporal_transformer=use_temporal_transformer
        self.use_spatial_transformer=use_spatial_transformer
        self.use_gnn=use_gnn
        self.gnn1=nn.Linear(width,width);self.gnn2=nn.Linear(width,width)
        xy=torch.as_tensor(coords,dtype=torch.float32)
        dist=torch.cdist(xy,xy); knn=dist.topk(min(9,len(xy)),largest=False).indices
        adj=torch.zeros(len(xy),len(xy)); adj.scatter_(1,knn,1); adj=torch.maximum(adj,adj.T); adj.fill_diagonal_(1)
        adj=adj/adj.sum(-1,keepdim=True).clamp_min(1)
        self.register_buffer('adj',adj)
        self.lead=nn.Embedding(horizon,width)
        self.decoder=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,1))
        self.register_buffer('coords',torch.as_tensor(coords,dtype=torch.float32))
    def encode(self,x,time_hist):
        # x: B,H,S,F; time_hist: B,H,6
        b,h,s,f=x.shape
        z=self.inp(x)+self.site.weight[None,None]+self.coord(self.coords)[None,None]
        z=z+self.clock(time_hist)[:,:,None]
        z=self.spatial(z.reshape(b*h,s,self.width)).reshape(b,h,s,self.width)
        if self.use_gnn:
            g=torch.einsum('ij,bhjd->bhid',self.adj,z)
            z=z+F.gelu(self.gnn1(g));g=torch.einsum('ij,bhjd->bhid',self.adj,z);z=z+F.gelu(self.gnn2(g))
        z=z.permute(0,2,1,3).reshape(b*s,h,self.width)
        z=self.temporal(z)
        if not self.use_temporal_transformer: z=z.mean(1,keepdim=True).expand(-1,h,-1)
        return (z[:,-1]+z.mean(1)).reshape(b,s,self.width)
    def forward(self,x,time_hist,future_time):
        z=self.encode(x,time_hist); b,s,d=z.shape; l=future_time.shape[1]
        glob=z.mean(1,keepdim=True).expand(-1,s,-1)
        site=z+glob
        ctx=site[:,None].expand(-1,l,-1,-1)+self.lead.weight[None,:,None,:]
        ctx=ctx+self.clock(future_time)[:,:,None]
        return self.decoder(torch.cat([ctx,site[:,None].expand(-1,l,-1,-1)],-1))[...,0]


class FlatResidualDDPM(nn.Module):
    def __init__(self,dim,width=256,steps=32):
        super().__init__(); self.steps=steps; self.time=nn.Embedding(steps,width)
        self.net=nn.Sequential(nn.Linear(dim+width,width*2),nn.GELU(),nn.Linear(width*2,width),nn.GELU(),nn.Linear(width,dim))
        beta=torch.linspace(1e-4,.02,steps); self.register_buffer('beta',beta); self.register_buffer('abar',(1-beta).cumprod(0))
    def eps(self,x,k): return self.net(torch.cat([x,self.time(k)],-1))
    def loss(self,r,mask):
        b=r.shape[0]; x=r.flatten(1); m=mask.expand_as(r).flatten(1)
        k=torch.randint(self.steps,(b,),device=r.device); a=self.abar[k,None]
        e=torch.randn_like(x); noisy=a.sqrt()*torch.where(m,x,0)+(1-a).sqrt()*e
        return ((self.eps(noisy,k)-e).square()*m).sum()/m.sum().clamp_min(1)
    @torch.no_grad()
    def sample(self,n,device,shape,members):
        x=torch.randn(n*members,self.net[0].in_features-self.time.embedding_dim,device=device)
        for k in reversed(range(self.steps)):
            kv=torch.full((len(x),),k,device=device,dtype=torch.long); e=self.eps(x,kv); a=self.abar[k]
            x=(x-self.beta[k]/(1-a).sqrt()*e)/(1-self.beta[k]).sqrt()
            if k: x=x+self.beta[k].sqrt()*torch.randn_like(x)
        # `shape` is the full deterministic forecast shape [batch, lead, site].
        # The sampler already flattens batch and ensemble-member dimensions,
        # so only the non-batch forecast axes belong after [batch, members].
        return x.view(n,members,*shape[1:])


class Version1DirectResidual(nn.Module):
    """Deterministic target forecast plus stochastic target residual (Version 1)."""
    def __init__(self,features,sites,coords,horizon=24,width=128,heads=4,steps=32,ablation=None):
        super().__init__(); ablation=ablation or 'full'
        self.det=SMFDeterministic(features,sites,coords,horizon=horizon,width=width,heads=heads,
            use_gnn=ablation!='no_gnn',use_spatial_transformer=ablation not in ('no_spatial_transformer','no_transformer'),
            use_temporal_transformer=ablation not in ('no_temporal_transformer','no_transformer'))
        self.use_diffusion=ablation!='no_diffusion'
        self.diff=FlatResidualDDPM(horizon*sites,steps=steps)
    def loss(self,b):
        mu=self.det(b['x'],b['time_hist'],b['time_future']); m=b['target_mask'].bool()
        det=(mu-b['y']).square()[m].mean(); r=b['y']-mu.detach()
        diff=self.diff.loss(r,m) if self.use_diffusion else det*0
        return det+.05*diff,{'det':det,'diff':diff}
    @torch.no_grad()
    def forecast(self,b,members=32):
        mu=self.det(b['x'],b['time_hist'],b['time_future'])
        if not self.use_diffusion:return mu[:,None].expand(-1,members,-1,-1),mu
        r=self.diff.sample(len(mu),mu.device,mu.shape,members)
        return mu[:,None]+r,mu


class Version2StateIntermediate(nn.Module):
    """SMF deterministic state forecast, residual diffusion, learned target decoder."""
    def __init__(self,features,sites,coords,horizon=24,width=128,heads=4,steps=32,ablation=None):
        super().__init__(); ablation=ablation or 'full'
        self.det=SMFDeterministic(features,sites,coords,horizon=horizon,width=width,heads=heads,
            use_gnn=ablation!='no_gnn',use_spatial_transformer=ablation not in ('no_spatial_transformer','no_transformer'),
            use_temporal_transformer=ablation not in ('no_temporal_transformer','no_transformer'))
        self.use_diffusion=ablation not in ('no_diffusion','no_state_diffusion')
        self.use_state_supervision=ablation!='no_state_supervision'
        self.use_cloudstate=ablation!='no_cloud_state'
        self.state=nn.Sequential(nn.Linear(width+1, width),nn.GELU(),nn.Linear(width,1))
        self.decode=nn.Sequential(nn.Linear(3,width),nn.GELU(),nn.Linear(width,64),nn.GELU(),nn.Linear(64,1))
        self.diff=FlatResidualDDPM(horizon*sites,steps=steps)
    def deterministic(self,b):
        z=self.det.encode(b['x'],b['time_hist']); base=self.det(b['x'],b['time_hist'],b['time_future'])
        # State is predicted per site/lead from deterministic SMF context + lead-conditioned target context.
        lead_ctx=self.det.lead.weight[None,:,None,:]+self.det.clock(b['time_future'])[:,:,None]
        site_ctx=z[:,None]+lead_ctx
        state=self.state(torch.cat([site_ctx,base[...,None]],-1)).squeeze(-1) if self.use_cloudstate else base
        return base,state,z
    def decode_state(self,state,z,b):
        if not self.use_cloudstate:return state
        zi=z[:,None].expand(-1,state.shape[1],-1,-1)
        y=self.decode(torch.stack([state,zi.mean(-1),b['time_future'][:,:,None,0].expand_as(state)],-1))[...,0]
        return y
    def loss(self,b):
        base,state,z=self.deterministic(b); sm=b['state_mask'].bool(); tm=b['target_mask'].bool()
        sl=(state-b['state']).square()[sm].mean() if self.use_cloudstate and self.use_state_supervision else state.square().mean()*0; yp=self.decode_state(state,z,b)
        yl=(yp-b['y']).square()[tm].mean()
        dl=self.diff.loss((b['state'] if self.use_cloudstate else b['y'])-state.detach(),(sm if self.use_cloudstate else tm))
        if not self.use_diffusion: dl=yl*0
        return sl+yl+.05*dl,{'state':sl,'target':yl,'diff':dl}
    @torch.no_grad()
    def forecast(self,b,members=32):
        _,state,z=self.deterministic(b)
        if not self.use_diffusion:return self.decode_state(state,z,b)[:,None].expand(-1,members,-1,-1),self.decode_state(state,z,b),state[:,None].expand(-1,members,-1,-1)
        r=self.diff.sample(len(state),state.device,state.shape,members)
        states=state[:,None]+r
        ys=[]
        for i in range(members): ys.append(self.decode_state(states[:,i],z,b))
        return torch.stack(ys,1),self.decode_state(state,z,b),states

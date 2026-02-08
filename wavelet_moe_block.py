import torch
import torch.nn as nn
from wavelet_function import DWT_2D, IDWT_2D

def ConvBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class MoERouter(nn.Module):
    """ MoE Router """
    def __init__(self, reduced_channels, num_experts=3, reduction=4, temperature=0.1):
        super(MoERouter, self).__init__()
        self.temperature = temperature
        mid_channels = max(reduced_channels // reduction, 8)
        
        # Shared gating network
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(reduced_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_experts, kernel_size=1, bias=False),
        )
    
    def forward(self, x):
        logits = self.fc(self.gap(x)).squeeze(-1).squeeze(-1) 
        weights = torch.softmax(logits / self.temperature, dim=1)  
        return weights 

class WaveBlockMoE(nn.Module):
    def __init__(self, wave, in_channels, out_channels, temperature=0.1):
        """
        WaveBlock with Mixture-of-Experts

        wave: default='haar'  others: ['harr','db','sym','bior']
        
        Experts:
          Expert1: Original low-frequency features (x_in)
          Expert2: First-level high-frequency details (x_h1)
          Expert3: Second-level multi-scale structures (x_h2)
          
        Router: gating
        """
        super(WaveBlockMoE, self).__init__()
        self.temperature = temperature
        
        # Initial feature transformation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        reduced_channels = out_channels // 4
        self.reduce = nn.Sequential(
            nn.Conv2d(out_channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )
        
        # Expert-specific processing
        self.filter1 = nn.Sequential(
            nn.Conv2d(reduced_channels * 3, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.filter2 = nn.Sequential(
            nn.Conv2d(reduced_channels * 3, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Wavelet transforms
        self.dwt1 = DWT_2D(wave=wave)
        self.idwt1 = IDWT_2D(wave=wave)
        self.dwt2 = DWT_2D(wave=wave)
        self.idwt2 = IDWT_2D(wave=wave)
        
        # Upsampling for multi-scale alignment
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # match expert output channels
        self.proj_in = nn.Sequential(
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Router
        self.router = MoERouter(
            reduced_channels=reduced_channels,
            num_experts=3,
            reduction=4,
            temperature=temperature
        )
        
        # Optional refinement layer
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # ===== Feature preprocessing =====
        x = self.conv(x)                     
        x_reduced = self.reduce(x)          
        x_in = x_reduced.clone()             

        # ===== Wavelet decomposition =====
        # First-level DWT
        x_dwt1 = self.dwt1(x_reduced)        
        x_ll1, x_lh1, x_hl1, x_hh1 = torch.chunk(x_dwt1, 4, dim=1)
        x_h1 = torch.cat([x_lh1, x_hl1, x_hh1], dim=1)  
        
        # Second-level DWT
        x_dwt2 = self.dwt2(x_ll1)           
        _, x_lh2, x_hl2, x_hh2 = torch.chunk(x_dwt2, 4, dim=1)
        x_h2 = torch.cat([x_lh2, x_hl2, x_hh2], dim=1)  

        # ===== Expert processing =====
        # Expert1: Low-frequency path (original features)
        expert1 = self.proj_in(x_in)        
        
        # Expert2: First-level high-frequency path
        x_h1 = self.filter1(x_h1)            
        expert2 = self.idwt1(x_h1)          
        
        # Expert3: Second-level multi-scale path
        x_h2 = self.filter2(x_h2)           
        x_h2 = self.idwt2(x_h2)              
        expert3 = self.upsample(x_h2)        

        # ===== Channel-wise Router =====
        # Generate global expert weights based on channel statistics
        weights = self.router(x_in)        
        
      
        w1 = weights[:, 0:1, None, None]    
        w2 = weights[:, 1:2, None, None]
        w3 = weights[:, 2:3, None, None]
        
        # MoE fusion: weighted sum of experts
        fused = w1 * expert1 + w2 * expert2 + w3 * expert3 
        
        # Optional refinement
        output = self.refine(fused)
        
        return output, weights 
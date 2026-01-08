import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_num_groups(channels, max_groups=32):
    """Find a valid number of groups for GroupNorm that divides channels."""
    num_groups = min(max_groups, channels)
    while num_groups > 1:
        if channels % num_groups == 0:
            return num_groups
        num_groups -= 1
    return 1


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class StochasticDepth(nn.Module):
    """Stochastic depth - randomly drop residual connections during training"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x, residual):
        if not self.training or self.drop_prob == 0:
            return x + residual
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x + residual * random_tensor / keep_prob


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(channels // reduction, 8)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer with conservative conditioning"""
    def __init__(self, num_features, num_channels):
        super().__init__()
        self.scale = nn.Linear(num_features, num_channels)
        self.shift = nn.Linear(num_features, num_channels)
        
        # Initialize for near-identity transformation
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.normal_(self.shift.weight, 0, 0.001)
        nn.init.zeros_(self.shift.bias)
    
    def forward(self, x, features):
        scale = self.scale(features).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(features).unsqueeze(-1).unsqueeze(-1)
        # Conservative conditioning to prevent training instability
        scale = torch.tanh(scale) * 0.3  # Reduced from 2.0
        shift = torch.tanh(shift) * 0.1  # Reduced from 1.0
        return x * (1 + scale) + shift


class LearnableDownsample(nn.Module):
    """Learnable downsampling with anti-aliasing"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Anti-aliasing blur before downsampling
        self.blur = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                              padding=1, groups=in_channels, bias=False)
        # Initialize blur kernel with Gaussian-like weights
        with torch.no_grad():
            self.blur.weight.fill_(1/9)
        
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                    stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.activation = Mish()
    
    def forward(self, x):
        x = self.blur(x)
        x = self.downsample(x)
        x = self.norm(x)
        return self.activation(x)


class LearnableUpsample(nn.Module):
    """Learnable upsampling with better interpolation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        self.norm = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.activation = Mish()
    
    def forward(self, x):
        return self.activation(self.norm(self.upsample(x)))


class ResidualConvBlock(nn.Module):
    """Residual convolution block with optional attention and modulation"""
    def __init__(self, in_channels, out_channels, num_convs=2, kernel_size=3,
                 use_se=True, num_extra_features=0, dropout_rate=0.0,
                 stochastic_depth_prob=0.0):
        super().__init__()
        self.use_film = num_extra_features > 0
        
        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            padding = kernel_size // 2
            
            layers.append(nn.Conv2d(in_ch, out_channels, kernel_size, 
                                   padding=padding, bias=False))
            layers.append(nn.GroupNorm(get_num_groups(out_channels), out_channels))
            
            if i < num_convs - 1:
                layers.append(Mish())

        self.conv = nn.Sequential(*layers)
        
        # Projection for residual connection if channels change
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels), out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        # Simplified attention: only SE block
        self.attention = SEBlock(out_channels) if use_se else nn.Identity()
        
        # FiLM layer for feature modulation
        if self.use_film:
            self.film = FiLMLayer(num_extra_features, out_channels)
        
        # Dropout only if specified (mainly for decoder)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Stochastic depth for regularization
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob) if stochastic_depth_prob > 0 else None
        
        self.final_activation = Mish()
    
    def forward(self, x, extra_features=None):
        residual = self.projection(x)
        out = self.conv(x)
        
        if self.use_film and extra_features is not None:
            out = self.film(out, extra_features)
        
        out = self.attention(out)
        
        if self.stochastic_depth is not None:
            out = self.stochastic_depth(residual, out)
        else:
            out = out + residual
            
        out = self.final_activation(out)
        out = self.dropout(out)
        
        return out


class SpatialBottleneck(nn.Module):
    """Spatial bottleneck that maintains spatial dimensions while processing features"""
    def __init__(self, channels, num_extra_features, dropout_rate=0.1):
        super().__init__()
        
        # Project extra features to spatial domain
        self.extra_proj = nn.Sequential(
            nn.Linear(num_extra_features, channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel mixing with residual connection
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.GroupNorm(get_num_groups(channels * 2), channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(channels), channels)
        )
        
        # Additional spatial processing
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(get_num_groups(channels), channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x, extra_vars):
        # Add extra features as spatial bias
        batch_size = x.shape[0]
        extra_spatial = self.extra_proj(extra_vars).unsqueeze(-1).unsqueeze(-1)
        extra_spatial = extra_spatial.expand(-1, -1, x.shape[2], x.shape[3])
        
        # Combine and process
        x = x + extra_spatial
        
        # Channel mixing with residual
        channel_out = self.channel_mixer(x)
        x = x + channel_out
        
        # Spatial refinement
        x = self.spatial_refine(x)
        
        return self.dropout(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 16)
            
        self.W_gate = nn.Conv2d(gate_channels, inter_channels, 1, bias=True)
        self.W_skip = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attention = self.relu(g + s)
        attention = self.psi(attention)
        return skip * attention


class ARUNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 num_extra_features=2,
                 max_depth=4,
                 initial_channels=32,
                 channel_multiplier=1.4,
                 convs_per_block=2,
                 kernel_size=3,
                 use_se=True,
                 use_attention_gates=False,
                 use_film=True,
                 use_learnable_downsampling=True,
                 use_learnable_upsampling=True,
                 dropout_rate=0.05,
                 stochastic_depth_prob=0.05,
                 max_channels=512):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_extra_features = num_extra_features
        self.max_depth = max_depth
        self.initial_channels = initial_channels
        self.use_film = use_film
        
        # Calculate channel progression with maximum cap
        encoder_channels = [initial_channels]
        for i in range(max_depth):
            next_channels = int(initial_channels * (channel_multiplier ** (i + 1)))
            next_channels = min(next_channels, max_channels)  # Cap maximum channels
            encoder_channels.append(next_channels)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        
        # First encoder block (no extra features, no dropout, no stochastic depth)
        self.encoder_blocks.append(
            ResidualConvBlock(
                in_channels, encoder_channels[0], 
                convs_per_block, kernel_size,
                use_se=use_se,
                num_extra_features=0,
                dropout_rate=0.0,  # No dropout in encoder
                stochastic_depth_prob=0.0  # No stochastic depth in first block
            )
        )
        
        # Downsampling layers
        if use_learnable_downsampling:
            self.downsample = nn.ModuleList([
                LearnableDownsample(encoder_channels[i], encoder_channels[i+1])
                for i in range(max_depth)
            ])
        else:
            self.downsample = nn.ModuleList([nn.MaxPool2d(2, stride=2) 
                                            for _ in range(max_depth)])
        
        # Remaining encoder blocks with gentler stochastic depth schedule
        for i in range(max_depth):
            in_ch = encoder_channels[i+1] if use_learnable_downsampling else encoder_channels[i]
            out_ch = encoder_channels[i+1]
            
            # Gentler stochastic depth schedule
            current_stoch_prob = stochastic_depth_prob * (i / (max_depth * 2))
            
            self.encoder_blocks.append(
                ResidualConvBlock(
                    in_ch, out_ch,
                    convs_per_block, kernel_size,
                    use_se=use_se,
                    num_extra_features=num_extra_features if use_film else 0,
                    dropout_rate=0.0,  # No dropout in encoder
                    stochastic_depth_prob=current_stoch_prob
                )
            )

        # Spatial bottleneck (maintains spatial dimensions)
        self.bottleneck = SpatialBottleneck(
            encoder_channels[-1],
            num_extra_features,
            dropout_rate
        )

        # Attention gates (optional)
        if use_attention_gates:
            self.attention_gates = nn.ModuleList([
                AttentionGate(
                    encoder_channels[max_depth - i - 1],  # gate is after upsampling
                    encoder_channels[max_depth - i - 1]   # skip is from encoder at same level
                )
                for i in range(max_depth)
            ])
        else:
            self.attention_gates = None

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        # Upsampling layers
        if use_learnable_upsampling:
            self.up_convs = nn.ModuleList([
                LearnableUpsample(
                    encoder_channels[max_depth - i], 
                    encoder_channels[max_depth - i - 1]
                )
                for i in range(max_depth)
            ])
        else:
            self.up_convs = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose2d(encoder_channels[max_depth - i], 
                                     encoder_channels[max_depth - i - 1], 
                                     kernel_size=2, stride=2, bias=False),
                    nn.GroupNorm(get_num_groups(encoder_channels[max_depth - i - 1]), 
                               encoder_channels[max_depth - i - 1]),
                    Mish()
                )
                for i in range(max_depth)
            ])
        
        # Decoder residual blocks with decreasing stochastic depth
        for i in range(max_depth):
            in_ch = encoder_channels[max_depth - i - 1] * 2  # Account for skip connection
            out_ch = encoder_channels[max_depth - i - 1]
            
            # Decrease stochastic depth in decoder
            current_stoch_prob = stochastic_depth_prob * (max_depth - i - 1) / (max_depth * 2)
            
            self.decoder_blocks.append(
                ResidualConvBlock(
                    in_ch, out_ch, 
                    convs_per_block, kernel_size,
                    use_se=use_se,
                    num_extra_features=num_extra_features if use_film else 0,
                    dropout_rate=dropout_rate,  # Use dropout only in decoder
                    stochastic_depth_prob=current_stoch_prob
                )
            )

        # Final output layer with small initialization
        self.final_conv = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.final_conv.weight, gain=0.1)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, extra_vars):
        encoder_features = []
        
        # Encoder path
        x = self.encoder_blocks[0](x)
        encoder_features.append(x)
        
        for i in range(self.max_depth):
            x = self.downsample[i](x)
            
            if self.use_film:
                x = self.encoder_blocks[i + 1](x, extra_vars)
            else:
                x = self.encoder_blocks[i + 1](x)
            
            if i < self.max_depth - 1:
                encoder_features.append(x)
        
        # Spatial bottleneck (no flattening!)
        x = self.bottleneck(x, extra_vars)
        
        # Decoder path
        encoder_features.reverse()
        
        for i in range(self.max_depth):
            x = self.up_convs[i](x)
            
            if i < len(encoder_features):
                skip = encoder_features[i]
                
                # Apply attention gate if available
                if self.attention_gates is not None:
                    skip = self.attention_gates[i](x, skip)
                
                x = torch.cat([x, skip], dim=1)
            
            if self.use_film:
                x = self.decoder_blocks[i](x, extra_vars)
            else:
                x = self.decoder_blocks[i](x)
        
        # Final output
        output = self.final_conv(x)
        
        return output


if __name__ == "__main__":
    # Test the improved model with recommended settings
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 128, 256)
    dummy_extra = torch.randn(batch_size, 2)
    
    print("=" * 80)
    print("IMPROVED UNET MODEL - OPTIMIZED VERSION")
    print("=" * 80)
    
    # Recommended configuration
    model = ARUNet(
        in_channels=1,
        out_channels=1,
        num_extra_features=2,
        initial_channels=32,
        max_depth=4,  # Reduced from 5
        channel_multiplier=2,  # Reduced from 1.5-2.0
        kernel_size=3,  # Reduced from 5
        convs_per_block=2,
        use_se=True,  # Simple and effective
        use_attention_gates=False,  # Start without them
        use_film=True,
        use_learnable_downsampling=True,
        use_learnable_upsampling=True,
        dropout_rate=0.05,  # Reduced, only in decoder
        stochastic_depth_prob=0.05,  # Reduced from 0.15
        max_channels=512  # Cap maximum channels
    )
    
    # Test forward pass
    output = model(dummy_input, dummy_extra)
    
    # Calculate model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.1f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    model.zero_grad()
    loss = output.mean()
    loss.backward()
    
    grad_stats = {'min': float('inf'), 'max': 0, 'sum': 0, 'count': 0}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats['min'] = min(grad_stats['min'], grad_norm)
            grad_stats['max'] = max(grad_stats['max'], grad_norm)
            grad_stats['sum'] += grad_norm
            grad_stats['count'] += 1
            
            if grad_norm > 10.0:
                print(f"  Large gradient in {name}: {grad_norm:.4f}")
    
    if grad_stats['count'] > 0:
        print(f"\nGradient statistics:")
        print(f"  Mean: {grad_stats['sum']/grad_stats['count']:.6f}")
        print(f"  Max: {grad_stats['max']:.6f}")
        print(f"  Min: {grad_stats['min']:.6f}")
    

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.nn.modules import Conv

class Gate2(nn.Module):
    """
    采用softmax函数，experts=16
    YOLO113n summary: 154 layers, 2,616,548 parameters, 0 gradients, 6.6 GFLOPs                                                                                                                                       
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.26it/s]                                                                         
                   all       4952      12032      0.811      0.724      0.809      0.609                                                                                                                          
             aeroplane        204        285      0.902      0.793      0.878      0.653                                                                                                                          
               bicycle        239        337      0.895      0.819      0.904      0.686                                                                                                                          
                  bird        282        459      0.827      0.699      0.785      0.556                                                                                                                          
                  boat        172        263      0.728      0.658      0.739       0.49                                                                                                                          
                bottle        212        469      0.885      0.541      0.697      0.474                                                                                                                          
                   bus        174        213       0.85      0.784      0.858      0.752                                                                                                                          
                   car        721       1201      0.906      0.823      0.911      0.731                                                                                                                          
                   cat        322        358       0.85      0.825      0.876      0.703                                                                                                                          
                 chair        417        756      0.759      0.511      0.645      0.432                                                                                                                          
                   cow        127        244      0.735      0.803      0.832      0.632
           diningtable        190        206      0.728      0.688      0.757      0.595
                   dog        418        489       0.78      0.754      0.842      0.647
                 horse        274        348      0.842      0.859      0.912      0.728
             motorbike        222        325      0.874      0.797      0.885      0.658                 
                person       2007       4528      0.905      0.741      0.874      0.605                 
           pottedplant        224        480      0.693      0.438      0.531      0.292                 
                 sheep         97        242       0.73      0.736      0.798      0.608                 
                  sofa        223        239      0.613      0.774      0.784       0.65                 
                 train        259        282      0.868      0.837      0.884      0.682                 
             tvmonitor        229        308      0.855      0.601      0.785      0.608                 
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image                        
Results saved to runs/yolo11_VOC/113n143  

使用sigmoid函数，experts=16， 效果也不理想，证明gate2无用
    YOLO113n summary: 154 layers, 2,616,548 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.28it/s]
                   all       4952      12032       0.82      0.718      0.809       0.61
             aeroplane        204        285      0.912      0.779      0.878      0.661
               bicycle        239        337      0.878      0.811      0.901      0.687
                  bird        282        459      0.814      0.638      0.772      0.539
                  boat        172        263      0.746      0.635      0.735       0.48
                bottle        212        469      0.871      0.548      0.698      0.467
                   bus        174        213      0.871       0.77      0.862       0.76
                   car        721       1201      0.895      0.842      0.918      0.738
                   cat        322        358      0.834       0.81      0.875      0.691
                 chair        417        756       0.79      0.484      0.622      0.423
                   cow        127        244      0.702       0.82      0.822      0.631
           diningtable        190        206      0.751      0.732      0.781       0.62
                   dog        418        489      0.807       0.76       0.85      0.659
                 horse        274        348      0.852      0.851      0.907      0.731
             motorbike        222        325      0.869      0.797      0.887      0.657
                person       2007       4528      0.904      0.733      0.868      0.603
           pottedplant        224        480      0.731      0.408       0.55      0.308
                 sheep         97        242      0.781      0.727      0.801      0.608
                  sofa        223        239      0.667      0.762      0.778      0.647
                 train        259        282      0.858      0.837      0.883      0.678
             tvmonitor        229        308      0.868      0.623      0.795      0.617
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to runs/yolo11_VOC/113n144
    """
    def __init__(
        self,
        channels: int = 512,
        groups: int = 1,
    ):
        super().__init__()

        self.avg_pool =nn.AdaptiveAvgPool2d(1)

        self.channels_mixer = nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=True)

        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        pooled = self.avg_pool(x)  
        weights = self.channels_mixer(pooled)  # (B, C*num_experts, 1, 1)
        weights = weights.view(B, self.groups, C//self.groups, 1, 1)
        # weights = F.softmax(weights, dim=2)  # 验证sigmoid，Gate1是sigmoid效果很好，softmax效果很差
        weights = F.sigmoid(weights)
        return weights


class Gate(nn.Module):
    def __init__(
        self,
        num_experts: int = 8,
        channels: int = 512,
    ):
        super().__init__()

        self.root = int(math.isqrt(num_experts))
        self.avg_pool = nn.AdaptiveAvgPool2d((self.root, self.root))

        # 使用更大的隐藏层增强表达能力
        hidden_dim = int(num_experts * 2.0)
        self.spatial_mixer = nn.Sequential(
            nn.Linear(num_experts, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts, bias=True),
            nn.Sigmoid(),  # 绝对不能用 nn.Softmax(dim=-1), 否则性能严重下降
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        pooled = self.avg_pool(x)  # (B, C, root, root)
        weights = self.spatial_mixer(pooled.view(B, C, -1))  # (B, C, num_experts)
        return weights


class MoVE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_experts = num_experts
        padding = kernel_size // 2

        # 并行化专家计算
        self.expert_conv = nn.Conv2d(
            in_channels,
            in_channels * num_experts,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.expert_norm = nn.InstanceNorm2d(in_channels * num_experts)
        self.expert_act = nn.SiLU()

        self.gate = Gate(num_experts)

        self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        A = self.num_experts

        # 获取门控权重和索引
        weights = self.gate(x)  # (B, C, A)

        # 使用分组卷积处理所有通道
        expert_outputs = self.expert_act(
            self.expert_norm(self.expert_conv(x))
        )  # (B, C*A, H, W)
        expert_outputs = expert_outputs.view(B, C, A, H, W)  # (B, C, A, H, W)

        # 权重应用与求和
        weights = weights.view(B, C, A, 1, 1)
        moe_out = (expert_outputs * weights).sum(dim=2)

        moe_out = self.out_project(moe_out)

        return moe_out


def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert num_channels % groups == 0, "num_channels should be " "divisible by groups"
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class MoVE_GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 8,  # 轻量分支专家数量
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)

        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=kernel_size, act=True
        )

        self.cheap_operation = MoVE(
            self.middle_channels, self.middle_channels, num_experts, 3 # 轻量分支卷积核大小一般都设为3，用于替代3x3深度分离卷积
        )

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out


class ESAA(nn.Module):
    """Efficient Spatial Aggregated Attention with Value Transform"""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.attn_W = Conv(c1=channels, c2=1, k=1, act=True)
        self.attn_H = Conv(c1=channels, c2=1, k=1, act=True)

        # Value projection layer
        self.v_conv = MoVE_GhostModule(channels, channels, kernel_size=1, num_experts=9)

        self.out_project = MoVE_GhostModule(channels, channels, kernel_size=1, num_experts=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_final = x  # Store original input for final residual connection

        # Compute Value representation once
        v = self.v_conv(x)

        # Width Attention Path
        residual_w = v
        logits_W = self.attn_W(v)
        context_scores_W = F.softmax(logits_W, dim=-1)
        context_vector_W = (v * context_scores_W).sum(-1, keepdim=True)
        x_W = residual_w + context_vector_W.expand_as(v)

        # Height Attention Path
        residual_h = x_W
        logits_H = self.attn_H(x_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        context_vector_H = (x_W * context_scores_H).sum(-2, keepdim=True)
        x_H = residual_h + context_vector_H.expand_as(x_W)

        out = v + x_W + x_H
        out = self.out_project(out) + residual_final

        return out


class TransMoVE(nn.Module):
    def __init__(
        self,
        channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.spatial_attn = ESAA(channels)

        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

        self.local_extractor = MoVE_GhostModule(
            channels, channels, num_experts, kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力子层
        residual = x
        norm_x = self.norm1(x)
        x = residual + self.spatial_attn(norm_x)

        # 前馈子层
        residual = x
        out = residual + self.local_extractor(x)

        return out

class ESAAM(nn.Module):
    """Efficient spatial aggregation attention module.

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_W = Conv(c1=middle_channels, c2=middle_channels, k=3, act=True)
        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = Conv(c1=middle_channels, c2=middle_channels, k=3, act=True)

        
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        x_W = self.conv_W(x_main)
        logits_W = self.attn_W(x_W)
        context_scores_W = F.softmax(logits_W, dim=-1)
        context_vector_W = x_W + (x_W * context_scores_W).sum(-1, keepdim=True).expand_as(x_W)

        x_H = self.conv_H(x_main)
        logits_H = self.attn_H(x_H)
        context_scores_H = F.softmax(logits_H, dim=-2)
        context_vector_H = x_H + (x_H * context_scores_H).sum(-2, keepdim=True).expand_as(x_H)
        x_final = torch.cat((context_vector_H, context_vector_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final)
    

    import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

# map: 60.9
# class Gate(nn.Module):
#     def __init__(
#         self,
#         num_experts: int = 8,
#         channels: int = 512,
#     ):
#         super().__init__()

#         self.root = int(math.isqrt(num_experts))
#         self.avg_pool = nn.AdaptiveAvgPool2d((self.root, self.root))

#         # 使用更大的隐藏层增强表达能力
#         hidden_dim = int(num_experts * 2.0)
#         self.spatial_mixer = nn.Sequential(
#             nn.Linear(num_experts, hidden_dim, bias=False),
#             nn.SiLU(), 
#             nn.Linear(hidden_dim, num_experts, bias=True),
#             nn.Sigmoid(), # 绝对不能用 nn.Softmax(dim=-1), 否则性能严重下降
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, C, _, _ = x.shape
#         pooled = self.avg_pool(x)  # (B, C, root, root)
#         weights = self.spatial_mixer(pooled.view(B, C, -1))  # (B, C, num_experts)
#         return weights


# def channel_shuffle(x, groups):
#     """Channel Shuffle operation.

#     This function enables cross-group information flow for multiple groups
#     convolution layers.

#     Args:
#         x (Tensor): The input tensor.
#         groups (int): The number of groups to divide the input tensor
#             in the channel dimension.

#     Returns:
#         Tensor: The output tensor after channel shuffle operation.
#     """

#     batch_size, num_channels, height, width = x.size()
#     assert num_channels % groups == 0, "num_channels should be " "divisible by groups"
#     channels_per_group = num_channels // groups

#     x = x.view(batch_size, groups, channels_per_group, height, width)
#     x = torch.transpose(x, 1, 2).contiguous()
#     x = x.view(batch_size, -1, height, width)

#     return x


# class MoVE(nn.Module):
#     def __init__(
#         self,
#         channels: int,
#         num_experts: int = 8,
#         kernel_size: int = 3,
#     ):
#         super().__init__()

#         self.middle_channels = int(channels // 2)

#         self.intrinsic_conv = Conv(channels, c2=self.middle_channels, k=3, act=True)

#         self.num_experts = num_experts
#         padding = kernel_size // 2

#         self.expert_conv = nn.Conv2d(
#             self.middle_channels,
#             self.middle_channels * num_experts,
#             kernel_size,
#             padding=padding,
#             groups=self.middle_channels,
#             bias=False,
#         )
#         self.expert_norm = nn.InstanceNorm2d(self.middle_channels * num_experts)
#         self.expert_act = nn.SiLU()
#         self.gate = Gate(num_experts)

#         # 混合通道信息
#         self.channel_mixer = Conv(self.middle_channels * 2, channels, k=1, act=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, _, H, W = x.shape

#         intrinsic_x = self.intrinsic_conv(x)

#         A = self.num_experts
#         # 获取门控权重
#         weights = self.gate(intrinsic_x)  # (B, C, A)

#         # 使用分组卷积处理所有通道
#         expert_outputs = self.expert_conv(intrinsic_x)
#         expert_outputs = self.expert_norm(expert_outputs)
#         expert_outputs = self.expert_act(expert_outputs) # (B, C * A, H, W)
#         expert_outputs = expert_outputs.view(B, self.middle_channels, A, H, W)  # (B, C, A, H, W)

#         # 权重应用与求和
#         weights = weights.view(B, self.middle_channels, A, 1, 1)
#         moe_out = (expert_outputs * weights).sum(dim=2)

#         out = torch.cat([moe_out, intrinsic_x], dim=1)
#         out = channel_shuffle(out, 2)
#         # 特征增强
#         out = self.channel_mixer(out)

#         return out

# class ESAA(nn.Module):
#     """Efficient Spatial Aggregated Attention with Value Transform"""
#     def __init__(self, channels: int) -> None:
#         super().__init__()

#         self.attn_S = Conv(c1=channels, c2=1, k=1, act=True)
#         self.attn_W = Conv(c1=channels, c2=1, k=1, act=True)
#         self.attn_H = Conv(c1=channels, c2=1, k=1, act=True)

#         # Value projection layer
#         self.v_conv =  Conv(c1=channels, c2=channels, k=1, act=True)

#         self.out_project = nn.Sequential(
#             Conv(c1=channels, c2=int(channels * 1.0), k=1, act=True),
#             Conv(c1=int(channels * 1.0), c2=channels, k=1, act=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual_final = x # Store original input for final residual connection

#         # Compute Value representation once
#         v = self.v_conv(x)

#         # Spatial Attention Path
#         residual_s = v 
#         logits_S = self.attn_S(v) 
#         spatial_scores = torch.sigmoid(logits_S)
#         x_spatial = spatial_scores * v + residual_s 

#         # Width Attention Path
#         residual_w = x_spatial 
#         logits_W = self.attn_W(x_spatial) 
#         context_scores_W = F.softmax(logits_W, dim=-1)
#         context_vector_W = (x_spatial * context_scores_W).sum(-1, keepdim=True) 
#         x_W = residual_w + context_vector_W.expand_as(x_spatial) 

#         # Height Attention Path
#         residual_h = x_W 
#         logits_H = self.attn_H(x_W) 
#         context_scores_H = F.softmax(logits_H, dim=-2)
#         context_vector_H = (x_W * context_scores_H).sum(-2, keepdim=True) 
#         x_H = residual_h + context_vector_H.expand_as(x_W) 


#         out = x_spatial + x_W + x_H 
#         out = self.out_project(out) + residual_final 

#         return out
    

# class ESAA_Simplified(nn.Module):
#     """
#     ESAA 模块改造版，采用串行轴向注意力确保全局信息传播。
#     """
#     def __init__(self, channels: int) -> None:
#         super().__init__()
#         # 重用注意力分数计算层 (1x1 Conv)
#         self.attn_H = Conv(c1=channels, c2=1, k=1, act=True)
#         self.attn_W = Conv(c1=channels, c2=1, k=1, act=True)
#         # 注意：原始的 attn_S 在此方案中未使用，因为我们专注于全局聚合

#         # Value projection layer (保持不变)
#         self.v_conv = Conv(c1=channels, c2=channels, k=1, act=True)

#         # Output projection layer (保持不变)
#         self.out_project = Conv(c1=channels, c2=channels, k=1, act=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         residual_final = x # 存储原始输入以供最后添加

#         # 1. 计算 Value 'v'
#         v = self.v_conv(x) # Shape: (B, C, H, W)

#         # --- 串行全局聚合 ---

#         # 2. 高度(Height)注意力与聚合
#         # 基于 'v' 计算高度注意力分数
#         logits_H = self.attn_H(v) # Shape: (B, 1, H, W)
#         # 沿高度维度应用 Softmax
#         scores_H = F.softmax(logits_H, dim=-2) # Shape: (B, 1, H, W)
#         # 使用高度分数对 'v' 进行加权求和，沿高度维度聚合
#         aggregated_H = (v * scores_H).sum(dim=-2, keepdim=True) # Shape: (B, C, 1, W)
#         # 将聚合后的高度上下文加回到 'v' (或其他融合方式)
#         # 这里使用加法，保留原始信息并加入上下文
#         v_plus_H = v + aggregated_H.expand_as(v) # Shape: (B, C, H, W)

#         # 3. 宽度(Width)注意力与聚合 (使用已包含高度信息的特征)
#         # 基于 'v_plus_H' 计算宽度注意力分数
#         logits_W = self.attn_W(v_plus_H) # Shape: (B, 1, H, W)
#         # 沿宽度维度应用 Softmax
#         scores_W = F.softmax(logits_W, dim=-1) # Shape: (B, 1, H, W)
#         # 使用宽度分数对 'v_plus_H' 进行加权求和，沿宽度维度聚合
#         # 此时聚合的特征已经包含了高度上下文信息
#         aggregated_W = (v_plus_H * scores_W).sum(dim=-1, keepdim=True) # Shape: (B, C, H, 1)
#         # 将聚合后的宽度上下文加回到 'v_plus_H'
#         v_plus_HW = v_plus_H + aggregated_W.expand_as(v) # Shape: (B, C, H, W)
#         # 现在 v_plus_HW 中的每个位置都通过串行聚合包含了全局信息

#         # 4. 输出投影与最终残差连接
#         out = self.out_project(v_plus_HW) + residual_final # Shape: (B, C, H, W)

#         return out

# class TransMoVE(nn.Module):
#     def __init__(
#         self,
#         channels: int,
#         num_experts: int = 9,
#         kernel_size: int = 3,
#     ):
#         super().__init__()

#         self.spatial_attn = ESAA(channels)

#         self.norm1 = nn.GroupNorm(1, channels)
#         self.norm2 = nn.GroupNorm(1, channels)

#         self.moe = MoVE(channels, num_experts, kernel_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 注意力子层
#         residual = x
#         norm_x = self.norm1(x)
#         x = residual + self.spatial_attn(norm_x)

#         # 前馈子层
#         residual = x
#         out = residual + self.moe(x)

#         return out

class Gate(nn.Module):
    def __init__(
        self,
        num_experts: int = 8,
        channels: int = 512,
    ):
        super().__init__()

        self.root = int(math.isqrt(num_experts))
        self.avg_pool = nn.AdaptiveAvgPool2d((self.root, self.root))

        # 使用更大的隐藏层增强表达能力
        hidden_dim = int(num_experts * 2.0)
        self.spatial_mixer = nn.Sequential(
            nn.Linear(num_experts, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts, bias=True),
            nn.Sigmoid(),  # 绝对不能用 nn.Softmax(dim=-1), 否则性能严重下降
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        pooled = self.avg_pool(x)  # (B, C, root, root)
        # print(pooled.shape)
        weights = self.spatial_mixer(pooled.view(B, C, -1))  # (B, C, num_experts)
        return weights


class MoVE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_experts = num_experts
        padding = kernel_size // 2

        # 并行化专家计算
        self.expert_conv = nn.Conv2d(
            in_channels,
            in_channels * num_experts,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.expert_norm = nn.InstanceNorm2d(in_channels * num_experts)
        self.expert_act = nn.SiLU()

        self.gate = Gate(num_experts)

        self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        A = self.num_experts

        # 获取门控权重和索引
        weights = self.gate(x)  # (B, C, A)

        # 使用分组卷积处理所有通道
        expert_outputs = self.expert_act(
            self.expert_norm(self.expert_conv(x))
        )  # (B, C*A, H, W)
        expert_outputs = expert_outputs.view(B, C, A, H, W)  # (B, C, A, H, W)

        # 权重应用与求和
        weights = weights.view(B, C, A, 1, 1)
        moe_out = (expert_outputs * weights).sum(dim=2)

        moe_out = self.out_project(moe_out)

        return moe_out


def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert num_channels % groups == 0, "num_channels should be " "divisible by groups"
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class MoVE_GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 16,  # 轻量分支专家数量
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)
        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=kernel_size, act=True
        )

        self.cheap_operation = MoVE(
            self.middle_channels, self.middle_channels, num_experts, 3 # 轻量分支卷积核大小一般都设为3，用于替代3x3深度分离卷积
        )

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out


class ESAA(nn.Module):
    """Efficient Spatial Aggregated Attention with Value Transform"""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.attn_W = Conv(c1=channels, c2=1, k=1, act=True)
        self.attn_H = Conv(c1=channels, c2=1, k=1, act=True)

        # Value projection layer
        self.v_conv = MoVE_GhostModule(channels, channels, kernel_size=1, num_experts=9)

        self.out_project = Conv(c1=channels, c2=channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_final = x  # Store original input for final residual connection

        # Compute Value representation once
        v = self.v_conv(x)

        # Width Attention Path
        residual_w = v
        logits_W = self.attn_W(v)
        context_scores_W = F.softmax(logits_W, dim=-1)
        context_vector_W = (v * context_scores_W).sum(-1, keepdim=True)
        x_W = residual_w + context_vector_W.expand_as(v)

        # Height Attention Path
        residual_h = x_W
        logits_H = self.attn_H(x_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        context_vector_H = (x_W * context_scores_H).sum(-2, keepdim=True)
        x_H = residual_h + context_vector_H.expand_as(x_W)

        out = v + x_W + x_H
        out = self.out_project(out) + residual_final

        return out

class ESAAM(nn.Module):
    """Efficient spatial aggregation attention module.
    YOLO113n summary: 166 layers, 2,555,428 parameters, 0 gradients, 6.6 GFLOPs                                                                                                                                       
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.29it/s]                                                                         
                   all       4952      12032      0.808      0.735      0.816      0.617                                                                                                                          
             aeroplane        204        285      0.901      0.798      0.891       0.67                                                                                                                          
               bicycle        239        337      0.885       0.84      0.912      0.704                                                                                                                          
                  bird        282        459      0.797      0.675       0.77      0.543                                                                                                                          
                  boat        172        263      0.749      0.677      0.752      0.495                                                                                                                          
                bottle        212        469      0.847      0.569      0.706      0.482                                                                                                                          
                   bus        174        213      0.855      0.798      0.873      0.767
                   car        721       1201      0.883      0.826      0.914      0.738
                   cat        322        358      0.834      0.838       0.88      0.702
                 chair        417        756      0.747      0.496      0.638      0.433
                   cow        127        244      0.741      0.824      0.848       0.64
           diningtable        190        206      0.745      0.728      0.764       0.61                 
                   dog        418        489      0.786      0.753      0.849      0.663                 
                 horse        274        348      0.859      0.874      0.918      0.738                 
             motorbike        222        325      0.876      0.801      0.899      0.656                 
                person       2007       4528      0.891      0.764      0.874      0.607                 
           pottedplant        224        480      0.689       0.39       0.52       0.29                 
                 sheep         97        242        0.7       0.76      0.821      0.634                 
                  sofa        223        239      0.635      0.741      0.786      0.653                 
                 train        259        282      0.879      0.852      0.903      0.691                 
             tvmonitor        229        308      0.871      0.699      0.796      0.615                 
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image                        
Results saved to runs/yolo11_VOC/113n133 

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )

        
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        residual_final = x

        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        logits_W = self.attn_W(x_main)
        context_scores_W = F.softmax(logits_W, dim=-1)
        x_plus_W = x_main + (x_main * context_scores_W).sum(-1, keepdim=True).expand_as(x_main)
        x_plus_W = self.conv_W(x_plus_W)

        
        logits_H = self.attn_H(x_plus_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        x_plus_WH = x_plus_W + (x_plus_W * context_scores_H).sum(-2, keepdim=True).expand_as(x_plus_W)
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final
    
class ESAAM_v2(nn.Module):
    """Efficient spatial aggregation attention module.
YOLO113n summary: 166 layers, 2,555,428 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.30it/s]
                   all       4952      12032       0.81       0.72      0.805      0.607
             aeroplane        204        285      0.917      0.804      0.892      0.667
               bicycle        239        337      0.906      0.829      0.907      0.698
                  bird        282        459      0.828      0.639      0.755      0.538
                  boat        172        263      0.722      0.658      0.732      0.482
                bottle        212        469       0.84      0.536      0.691      0.469
                   bus        174        213      0.842      0.812      0.868      0.761
                   car        721       1201      0.877       0.84      0.918      0.735
                   cat        322        358      0.852      0.804      0.864      0.686
                 chair        417        756      0.741      0.493      0.631      0.432
                   cow        127        244      0.719      0.775      0.829      0.622
           diningtable        190        206      0.733      0.689      0.787      0.604
                   dog        418        489      0.771      0.773      0.833      0.646
                 horse        274        348      0.867      0.836      0.901      0.722
             motorbike        222        325      0.879      0.805      0.883      0.648
                person       2007       4528      0.888      0.761      0.871      0.605
           pottedplant        224        480      0.694      0.375      0.501       0.28
                 sheep         97        242      0.758       0.74      0.802      0.606
                  sofa        223        239      0.672      0.745      0.783      0.649
                 train        259        282      0.867       0.84       0.87      0.683
             tvmonitor        229        308      0.825      0.644      0.786      0.602
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to runs/yolo11_VOC/113n134
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )

        
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        residual_final = x

        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        logits_W = self.attn_W(x_main)
        context_scores_W = F.softmax(logits_W, dim=-1)
        x_plus_W = x_main + x_main.sum(-1, keepdim=True).expand_as(x_main) * context_scores_W
        x_plus_W = self.conv_W(x_plus_W)

        
        logits_H = self.attn_H(x_plus_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        x_plus_WH = x_plus_W + x_plus_W.sum(-2, keepdim=True).expand_as(x_plus_W) * context_scores_H
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final
    
class ESAAM_v3(nn.Module):
    """Efficient spatial aggregation attention module.
    简单测试了一下，感觉也不行
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )

        
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        residual_final = x

        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        logits_W = self.attn_W(x_main)
        context_scores_W = F.sigmoid(logits_W)
        x_plus_W = x_main + x_main.sum(-1, keepdim=True).expand_as(x_main) * context_scores_W
        x_plus_W = self.conv_W(x_plus_W)

        
        logits_H = self.attn_H(x_plus_W)
        context_scores_H = F.sigmoid(logits_H)
        x_plus_WH = x_plus_W + x_plus_W.sum(-2, keepdim=True).expand_as(x_plus_W) * context_scores_H
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final
    
class ESAAM_v4(nn.Module):
    """Efficient spatial aggregation attention module.
YOLO113n summary: 166 layers, 2,555,916 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.28it/s]
                   all       4952      12032      0.814      0.726      0.809      0.612
             aeroplane        204        285      0.897      0.796      0.889      0.665
               bicycle        239        337      0.878      0.816      0.896       0.69
                  bird        282        459       0.83       0.68      0.779      0.555
                  boat        172        263      0.726      0.635      0.721      0.463
                bottle        212        469       0.84       0.57      0.707      0.472
                   bus        174        213      0.858      0.766      0.863      0.767
                   car        721       1201      0.892      0.834       0.91       0.73
                   cat        322        358      0.848      0.809       0.86      0.688
                 chair        417        756      0.758      0.484       0.61      0.416
                   cow        127        244      0.728        0.8      0.838      0.642
           diningtable        190        206      0.758      0.757      0.791      0.623
                   dog        418        489      0.845      0.724      0.853       0.66
                 horse        274        348      0.849      0.868      0.914       0.74
             motorbike        222        325      0.869      0.803      0.885      0.664
                person       2007       4528      0.898      0.761      0.877      0.608
           pottedplant        224        480      0.738      0.435      0.554      0.309
                 sheep         97        242      0.725      0.752       0.79      0.608
                  sofa        223        239      0.638      0.751      0.768      0.636
                 train        259        282       0.85       0.84      0.891      0.693
             tvmonitor        229        308       0.86       0.64      0.788      0.608
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to runs/yolo11_VOC/113n136
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=2, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=2, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )

        
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        residual_final = x

        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        logits = self.attn_W(x_main)
        logits_sum, logits_W = torch.split(
            logits, [1, 1], dim=1
        )
        context_scores_sum = F.softmax(logits_sum, dim=-1)
        context_scores_W = F.sigmoid(logits_W)
        x_plus_W = x_main + (x_main * context_scores_sum).sum(-1, keepdim=True).expand_as(x_main) * context_scores_W
        x_plus_W = self.conv_W(x_plus_W)

        
        logits = self.attn_H(x_plus_W)
        logits_sum, logits_H = torch.split(
            logits, [1, 1], dim=1
        )
        context_scores_sum = F.softmax(logits_sum, dim=-2)
        context_scores_H = F.sigmoid(logits_H)
        x_plus_WH = x_plus_W + (x_plus_W * context_scores_sum).sum(-2, keepdim=True).expand_as(x_plus_W) * context_scores_H
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final
    
class ESAAM_v5(nn.Module):
    """Efficient spatial aggregation attention module.
YOLO113n summary: 166 layers, 2,555,916 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.28it/s]
                   all       4952      12032      0.804      0.731       0.81       0.61
             aeroplane        204        285      0.893      0.804      0.885      0.671
               bicycle        239        337      0.904      0.822      0.898      0.689
                  bird        282        459      0.819      0.692      0.788      0.541
                  boat        172        263      0.698      0.654      0.728       0.49
                bottle        212        469      0.849      0.552      0.705      0.473
                   bus        174        213      0.844      0.775      0.864      0.766
                   car        721       1201      0.892      0.823      0.908      0.729
                   cat        322        358      0.872      0.813      0.866      0.683
                 chair        417        756      0.736      0.483      0.621      0.426
                   cow        127        244      0.711      0.816      0.829      0.621
           diningtable        190        206      0.711      0.723      0.778      0.599
                   dog        418        489      0.784      0.765      0.853      0.664
                 horse        274        348      0.848      0.865      0.916      0.734
             motorbike        222        325      0.842      0.819      0.885      0.654
                person       2007       4528       0.89      0.752      0.874       0.61
           pottedplant        224        480      0.704      0.435      0.532      0.296
                 sheep         97        242      0.734      0.793       0.81       0.62
                  sofa        223        239      0.636      0.732      0.763      0.636
                 train        259        282      0.872      0.837      0.892       0.69
             tvmonitor        229        308      0.845      0.656      0.803      0.612
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to runs/yolo11_VOC/113n137
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=2, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=2, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )

        
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        residual_final = x

        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        logits = self.attn_W(x_main)
        logits_sum, logits_W = torch.split(
            logits, [1, 1], dim=1
        )
        context_scores_sum = F.softmax(logits_sum, dim=-1)
        context_scores_W = F.sigmoid(logits_W)
        x_plus_W = x_main * context_scores_W + (x_main * context_scores_sum).sum(-1, keepdim=True).expand_as(x_main)
        x_plus_W = self.conv_W(x_plus_W)

        
        logits = self.attn_H(x_plus_W)
        logits_sum, logits_H = torch.split(
            logits, [1, 1], dim=1
        )
        context_scores_sum = F.softmax(logits_sum, dim=-2)
        context_scores_H = F.sigmoid(logits_H)
        x_plus_WH = x_plus_W * context_scores_H + (x_plus_W * context_scores_sum).sum(-2, keepdim=True).expand_as(x_plus_W)
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final

class TransMoVE(nn.Module):
    def __init__(
        self,
        channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.spatial_attn = ESAAM(channels, channels)
        
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

        self.local_extractor = MoVE_GhostModule(
            channels, channels, kernel_size, num_experts
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力子层
        residual = x
        norm_x = self.norm1(x)
        x = residual + self.spatial_attn(norm_x)

        # 前馈子层
        residual = x
        out = residual + self.local_extractor(x)

        return out
    

class Gate(nn.Module):
    def __init__(
        self,
        num_experts: int = 8,
        channels: int = 512,
    ):
        super().__init__()

        self.root = int(math.isqrt(num_experts))
        self.avg_pool = nn.AdaptiveAvgPool2d((self.root, self.root))

        # 使用更大的隐藏层增强表达能力
        hidden_dim = int(num_experts * 2.0)
        self.spatial_mixer = nn.Sequential(
            nn.Linear(num_experts, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts, bias=True),
            nn.Sigmoid(),  # 绝对不能用 nn.Softmax(dim=-1), 否则性能严重下降
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        pooled = self.avg_pool(x)  # (B, C, root, root)
        # print(pooled.shape)
        weights = self.spatial_mixer(pooled.view(B, C, -1))  # (B, C, num_experts)
        return weights


class MoVE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_experts = num_experts
        padding = kernel_size // 2

        # 并行化专家计算
        self.experts = nn.ModuleList(
            [
                Conv(
                    c1=in_channels,
                    c2=in_channels,
                    k=kernel_size,
                    s=1,
                    p=padding,
                    g=in_channels,
                    act=True,
                )
                for _ in range(num_experts)
            ]
        )

        self.gate = Gate(num_experts)

        self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        A = self.num_experts

        # 获取门控权重和索引
        weights = self.gate(x)  # (B, C, A)

        # 使用分组卷积处理所有通道
        expert_outputs = torch.stack(
            [
                expert(x)
                for expert in self.experts
            ],
            dim=1,
        )
        expert_outputs = expert_outputs.view(B, C, A, H, W)

        # 权重应用与求和
        weights = weights.view(B, C, A, 1, 1)
        moe_out = (expert_outputs * weights).sum(dim=2)

        moe_out = self.out_project(moe_out)

        return moe_out


class MoVE(nn.Module):
    """
    玄学，调整了一下代码位置，效果变差了
    YOLO113n summary: 166 layers, 2,555,428 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.31it/s]
                   all       4952      12032      0.816      0.721      0.809      0.613
             aeroplane        204        285      0.914      0.781      0.886      0.671
               bicycle        239        337      0.883      0.801      0.898      0.688
                  bird        282        459      0.812      0.678      0.774      0.549
                  boat        172        263      0.739      0.666      0.734      0.488
                bottle        212        469      0.862      0.533      0.702      0.471
                   bus        174        213      0.862      0.765      0.862      0.755
                   car        721       1201      0.898       0.83      0.916      0.734
                   cat        322        358      0.795       0.83      0.877      0.708
                 chair        417        756      0.797      0.461      0.622      0.425
                   cow        127        244      0.723      0.802      0.834      0.637
           diningtable        190        206      0.752      0.663      0.753      0.612
                   dog        418        489      0.834      0.742      0.847      0.654
                 horse        274        348      0.861      0.842      0.909      0.726
             motorbike        222        325      0.867      0.825      0.894      0.673
                person       2007       4528      0.906      0.739       0.87      0.606
           pottedplant        224        480       0.72      0.403      0.529      0.299
                 sheep         97        242       0.72       0.76      0.805      0.621                                                                                                                          
                  sofa        223        239      0.664      0.762      0.772      0.636                                                                                                                          
                 train        259        282      0.874      0.848      0.898      0.691                                                                                                                          
             tvmonitor        229        308      0.844      0.687      0.794       0.61                                                                                                                          
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image                                                                                                                                 
Results saved to runs/yolo11_VOC/113n147 
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_experts = num_experts
        padding = kernel_size // 2

        # 并行化专家计算
        self.expert_conv = nn.Conv2d(
            in_channels,
            in_channels * num_experts,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.expert_norm = nn.InstanceNorm2d(in_channels * num_experts)
        self.expert_act = nn.SiLU()

        self.gate = Gate(num_experts=num_experts)

        self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        A = self.num_experts

        # 使用分组卷积处理所有通道
        expert_outputs = self.expert_act(
            self.expert_norm(self.expert_conv(x))
        )  # (B, C*A, H, W)
        expert_outputs = expert_outputs.view(B, C, A, H, W)  # (B, C, A, H, W)

        # 调整了一下expert_outputs和weights的计算顺序，验证鲁棒性
        # 可能因为神经网络有一些in-place操作，会导致权重计算存在细微差异
        # 权重应用与求和
        weights = self.gate(x) # (B, C, A) 
        weights = weights.view(B, C, A, 1, 1)
        moe_out = (expert_outputs * weights).sum(dim=2)

        moe_out = self.out_project(moe_out)

        return moe_out

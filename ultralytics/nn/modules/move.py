import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


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
    """
        num_experts: int = 16
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

    num_experts: int = 9
    YOLO113n summary: 166 layers, 2,537,480 parameters, 0 gradients, 6.5 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:10<00:00,  3.63it/s]
                       all       4952      12032      0.818      0.723      0.809      0.613
                 aeroplane        204        285      0.885      0.811       0.88      0.653
                   bicycle        239        337      0.887      0.815      0.906      0.694
                      bird        282        459       0.82      0.673      0.781      0.556
                      boat        172        263      0.688      0.658      0.733      0.465
                    bottle        212        469      0.863       0.55      0.705      0.478
                       bus        174        213      0.849      0.779      0.852      0.755
                       car        721       1201      0.901      0.817      0.912      0.736
                       cat        322        358      0.835       0.83      0.883      0.711
                     chair        417        756      0.741      0.469      0.613      0.421
                       cow        127        244      0.753      0.774      0.833      0.632
               diningtable        190        206      0.776       0.69       0.77      0.602
                       dog        418        489      0.811      0.769      0.854      0.666
                     horse        274        348      0.855      0.878      0.915      0.738
                 motorbike        222        325      0.882      0.781      0.869      0.651
                    person       2007       4528      0.892      0.761      0.877      0.612
               pottedplant        224        480      0.763      0.379       0.52      0.296
                     sheep         97        242      0.772      0.769      0.813      0.625
                      sofa        223        239      0.645      0.753      0.774      0.646
                     train        259        282      0.861      0.819      0.882      0.687
                 tvmonitor        229        308      0.882      0.682      0.816      0.628
    Speed: 0.1ms preprocess, 1.1ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n149

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
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


class GhostModule(nn.Module):
    """
        主分支kernel size为1：
        YOLO113n summary (fused): 130 layers, 2,061,428 parameters, 0 gradients, 5.2 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:08<00:00,  4.48it/s]
                       all       4952      12032      0.801      0.708       0.79      0.591
                 aeroplane        204        285      0.876      0.767      0.873      0.648
                   bicycle        239        337      0.874      0.813      0.886      0.677
                      bird        282        459      0.807      0.654      0.762      0.531
                      boat        172        263      0.731      0.624      0.707      0.474
                    bottle        212        469      0.846      0.527      0.659       0.43
                       bus        174        213      0.847      0.765      0.841      0.725
                       car        721       1201      0.887      0.828      0.904      0.724
                       cat        322        358      0.828      0.788      0.846      0.667
                     chair        417        756       0.75      0.458      0.595      0.394
                       cow        127        244      0.693      0.803      0.819      0.619
               diningtable        190        206      0.728      0.704       0.74      0.576
                       dog        418        489      0.767      0.728      0.811      0.619
                     horse        274        348      0.856      0.819      0.895      0.705
                 motorbike        222        325      0.852      0.762      0.877      0.642
                    person       2007       4528      0.893      0.736      0.859      0.594
               pottedplant        224        480      0.681      0.405      0.519      0.283
                     sheep         97        242      0.772      0.713      0.771      0.586
                      sofa        223        239      0.631      0.724      0.757      0.627
                     train        259        282      0.867      0.858        0.9      0.701
                 tvmonitor        229        308      0.836      0.678      0.783      0.601
    Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.3ms postprocess per image
    Results saved to runs/yolo11_VOC/113n139
        主分支kernel size为3：
        YOLO113n summary (fused): 130 layers, 2,409,588 parameters, 0 gradients, 6.1 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:08<00:00,  4.43it/s]
                       all       4952      12032      0.811      0.714      0.801      0.601
                 aeroplane        204        285       0.87      0.777      0.876      0.672
                   bicycle        239        337      0.867      0.813      0.898      0.689
                      bird        282        459      0.829       0.63      0.754      0.531
                      boat        172        263      0.734      0.639      0.707       0.47
                    bottle        212        469      0.864      0.515      0.671      0.442
                       bus        174        213      0.835      0.781      0.858      0.745
                       car        721       1201      0.894      0.813       0.91       0.73
                       cat        322        358      0.856       0.81      0.874      0.689
                     chair        417        756      0.774      0.475      0.623      0.421
                       cow        127        244      0.737      0.766      0.833      0.632
               diningtable        190        206      0.752      0.684      0.757      0.598
                       dog        418        489      0.817      0.749      0.829      0.634
                     horse        274        348      0.815      0.868      0.903      0.719
                 motorbike        222        325       0.85      0.783      0.878      0.649
                    person       2007       4528      0.906      0.738      0.869      0.603
               pottedplant        224        480      0.673       0.41      0.506      0.275
                     sheep         97        242      0.752      0.781      0.816      0.616
                      sofa        223        239      0.682      0.757      0.787      0.636
                     train        259        282      0.881      0.838      0.887      0.688
                 tvmonitor        229        308      0.841      0.659      0.776      0.582
    Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n140

        采用channel shuffle：
    YOLO113n summary (fused): 130 layers, 2,409,588 parameters, 0 gradients, 6.1 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:08<00:00,  4.39it/s]
                       all       4952      12032      0.806      0.715      0.801      0.602
                 aeroplane        204        285      0.868      0.784      0.867      0.659
                   bicycle        239        337      0.885      0.843        0.9      0.687
                      bird        282        459       0.78       0.66      0.759      0.522
                      boat        172        263      0.697      0.655      0.722      0.461
                    bottle        212        469      0.865      0.494      0.671      0.443
                       bus        174        213      0.838      0.737      0.855      0.752
                       car        721       1201      0.899      0.829      0.912      0.738
                       cat        322        358      0.849      0.799      0.869      0.687
                     chair        417        756      0.756      0.479      0.616      0.421
                       cow        127        244      0.705      0.811      0.833      0.631
               diningtable        190        206      0.747      0.665      0.731      0.584
                       dog        418        489      0.794      0.736      0.836       0.64
                     horse        274        348      0.857      0.845      0.912      0.721
                 motorbike        222        325      0.862      0.818      0.905      0.664
                    person       2007       4528       0.89      0.756      0.868      0.596
               pottedplant        224        480      0.697      0.403      0.517      0.277
                     sheep         97        242      0.772      0.769      0.809      0.617
                      sofa        223        239       0.64      0.757      0.761      0.639
                     train        259        282      0.874      0.833       0.89      0.687
                 tvmonitor        229        308      0.847      0.633      0.785      0.602
    Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n141

        增加输出映射层：
    YOLO113n summary (fused): 134 layers, 2,497,108 parameters, 0 gradients, 6.3 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:09<00:00,  4.33it/s]
                       all       4952      12032      0.801      0.734      0.808      0.605
                 aeroplane        204        285      0.895      0.782      0.883      0.659
                   bicycle        239        337      0.907      0.816      0.898      0.685
                      bird        282        459      0.773      0.683      0.769      0.541
                      boat        172        263      0.732      0.654      0.715      0.461
                    bottle        212        469      0.855      0.577      0.706       0.47
                       bus        174        213      0.833      0.784      0.862      0.755
                       car        721       1201      0.881      0.833      0.913       0.73
                       cat        322        358      0.858      0.813      0.878      0.689
                     chair        417        756      0.751      0.528      0.634      0.424
                       cow        127        244      0.714      0.807      0.829      0.627
               diningtable        190        206      0.757      0.709      0.769      0.614
                       dog        418        489      0.802      0.744      0.842       0.66
                     horse        274        348      0.843      0.853      0.892      0.699
                 motorbike        222        325      0.838        0.8      0.881      0.646
                    person       2007       4528      0.896       0.75      0.867        0.6
               pottedplant        224        480       0.69      0.438      0.526      0.292
                     sheep         97        242      0.719       0.81      0.833      0.625
                      sofa        223        239      0.621      0.753      0.772      0.645
                     train        259        282      0.843      0.858      0.893      0.678
                 tvmonitor        229        308      0.811      0.683      0.802      0.609
    Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n142

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 1,  # 轻量分支专家数量
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)
        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=kernel_size, act=True
        )

        self.cheap_operation = Conv(
            self.middle_channels,
            self.middle_channels,
            k=3,
            g=self.middle_channels,
            act=True,
        )  # 3x3深度分离卷积, 即num_experts=1

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out


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
            self.middle_channels,
            self.middle_channels,
            num_experts,
            3,  # 轻量分支卷积核大小一般都设为3，用于替代3x3深度分离卷积
        )

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
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

    def __init__(
        self,
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
            Conv(
                c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True
            ),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(
                c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True
            ),
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
        x_plus_W = x_main + (x_main * context_scores_W).sum(-1, keepdim=True).expand_as(
            x_main
        )
        x_plus_W = self.conv_W(x_plus_W)

        logits_H = self.attn_H(x_plus_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        x_plus_WH = x_plus_W + (x_plus_W * context_scores_H).sum(
            -2, keepdim=True
        ).expand_as(x_plus_W)
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)

        return self.final_conv(x_final) + residual_final


# class TransMoVE(nn.Module):
# """
# coco n scale
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.393
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.550
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.426
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.431
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.324
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.541
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.596
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.371
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.660
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.775
# Results saved to runs/yolo11_coco/113n
# """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         num_experts: int = 9,
#         kernel_size: int = 3,
#     ):
#         super().__init__()

#         if out_channels == in_channels:
#             self.conv = nn.Identity()
#         else:
#             self.conv = Conv(in_channels, out_channels, k=1, act=True)

#         self.spatial_attn = ESAAM(out_channels, out_channels)

#         self.norm1 = nn.GroupNorm(1, out_channels)
#         self.norm2 = nn.GroupNorm(1, out_channels)

#         self.local_extractor = MoVE_GhostModule(
#             out_channels, out_channels, kernel_size, num_experts
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         x = self.conv(x)

#         # 注意力子层
#         residual = x
#         norm_x = self.norm1(x)
#         x = residual + self.spatial_attn(norm_x)

#         # 前馈子层
#         residual = x
#         out = residual + self.local_extractor(x)

#         return out


class TransMoVE(nn.Module):
    """采用ELAN结构
    
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()

        # 注意力子层
        # --------------------------------------------------------------
        self.spatial_attn = ESAAM(out_channels, out_channels)

        # ELAN结构
        # ---------------------------------------------------------------
        num_blocks = 2
        num_in_block = 1
        middle_ratio = 0.5

        self.num_blocks = num_blocks

        middle_channels = int(out_channels * middle_ratio)
        block_channels = int(out_channels * middle_ratio)
        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if num_in_block == 1:
                internal_block = MoVE_GhostModule(
                    in_channels=middle_channels,
                    out_channels=block_channels,
                    num_experts=num_experts,
                    kernel_size=kernel_size,
                )
            else:
                internal_block = []
                for _ in range(num_in_block):
                    internal_block.append(
                        MoVE_GhostModule(
                            in_channels=middle_channels,
                            out_channels=block_channels,
                            num_experts=num_experts,
                            kernel_size=kernel_size,
                        )
                    )
                internal_block = nn.Sequential(*internal_block)

            self.blocks.append(internal_block)

        final_conv_in_channels = (
            num_blocks * block_channels + int(out_channels * middle_ratio) * 2
        )
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

        self.gamma_attn = nn.Parameter(0.01 * torch.ones(out_channels), requires_grad=True)
        self.gamma_elan = nn.Parameter(0.01 * torch.ones(out_channels), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        # 注意力子层
        residual = x
        x = residual + self.spatial_attn(x) * self.gamma_attn.view(-1, len(self.gamma_attn), 1, 1)

        # ELAN子层
        residual = x
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        return residual + self.final_conv(x_final) * self.gamma_elan.view(-1, len(self.gamma_elan), 1, 1) 


class TransMoVE_20250420(nn.Module):
    """采用ELAN结构
    COCO
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.406
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.568
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.436
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.446
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.329
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.789
Results saved to runs/yolo11_coco/113n2
   
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()

        # 注意力子层
        # --------------------------------------------------------------
        self.spatial_attn = ESAAM(out_channels, out_channels)

        # ELAN结构
        # ---------------------------------------------------------------
        num_blocks = 2
        num_in_block = 1
        middle_ratio = 0.5

        self.num_blocks = num_blocks

        middle_channels = int(out_channels * middle_ratio)
        block_channels = int(out_channels * middle_ratio)
        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if num_in_block == 1:
                internal_block = MoVE_GhostModule(
                    in_channels=middle_channels,
                    out_channels=block_channels,
                    num_experts=num_experts,
                    kernel_size=kernel_size,
                )
            else:
                internal_block = []
                for _ in range(num_in_block):
                    internal_block.append(
                        MoVE_GhostModule(
                            in_channels=middle_channels,
                            out_channels=block_channels,
                            num_experts=num_experts,
                            kernel_size=kernel_size,
                        )
                    )
                internal_block = nn.Sequential(*internal_block)

            self.blocks.append(internal_block)

        final_conv_in_channels = (
            num_blocks * block_channels + int(out_channels * middle_ratio) * 2
        )
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:


        # 注意力子层
        residual = x
        x = residual + self.spatial_attn(x)

        # ELAN子层
        residual = x
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        return self.final_conv(x_final) + residual



class GAttn(nn.Module):
    """
    Attention module with the gate function. 带门控机制的自注意力模块

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import GAttn
        >>> model = GAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)


        q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        max_attn = attn.max(dim=-1, keepdim=True).values
        exp_attn = torch.exp(attn - max_attn)
        attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
        x = (v @ attn.transpose(-2, -1))

        x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
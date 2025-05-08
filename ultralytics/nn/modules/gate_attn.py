import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
from fvcore.nn import FlopCountAnalysis

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
        self.input_gate = Conv(dim, all_head_dim, 1, act=nn.Sigmoid())
        self.output_gate = Conv(all_head_dim, dim, 1, act=nn.Sigmoid())
        self.forget_gate = Conv(num_heads, num_heads, k=(1,5), act=True)
   
    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W
        input = x

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.input_gate(x) * x
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)


        q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        attn = self.forget_gate(attn)
        x = v @ attn.transpose(-2, -1)

        x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        out = self.output_gate(x) * x + input
        return out


if __name__ == "__main__":
    
    x = torch.randn(2, 64, 80, 80)
    model = GAttn(dim=64, num_heads=2, area=4)
    # 使用FlopCountAnalysis计算FLOPs
    flops = FlopCountAnalysis(model, x)
    print(f"FLOPs: {flops.total() / 1e9} G")

    # 计算模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params / 1e6} M")

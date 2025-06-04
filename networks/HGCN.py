import torch
import torch.nn.functional as F
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_v_y = nn.Conv2d(plane*2, plane, kernel_size=1)
        self.node_q_y = nn.Conv2d(plane*2, plane, kernel_size=1)
        self.conv_wg_decode = nn.Conv1d(inter_plane*2, inter_plane*2, kernel_size=1, bias=False)
        self.bn_wg_decode = BatchNorm1d(inter_plane*2)

        self.softmax = nn.Softmax(dim=2)
        self.outdecode = nn.Sequential(nn.Conv2d(inter_plane*2, plane, kernel_size=1),
                                 BatchNorm2d(plane))
        self.xpre = nn.Sequential(nn.Conv2d(inter_plane*4, inter_plane*2, kernel_size=1),
                                 BatchNorm2d(inter_plane*2),
                                 nn.Conv2d(inter_plane*2, plane, kernel_size=1),
                                 BatchNorm2d(plane)
                                  )

    def forward(self, x, y): # y的通道数是x的一半 当有y时输入的通道数是y的通道数 当没有y时输入的是x的通道数 x来之encode y来之decode
        node_k = y
        node_v = self.node_v_y(x)
        node_q = self.node_q_y(x)
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        AV = torch.bmm(node_q, node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg_decode(AV)
        AVW = self.bn_wg_decode(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.outdecode(AVW) + self.xpre(x))
        return out

class HierarchicalSpatialGraph(nn.Module):
    """
        Feature GCN with coordinate 我GCN
    """

    def __init__(self, planes, ratio=4):
        super(HierarchicalSpatialGraph, self).__init__()
        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=1, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=1, padding=1, bias=False),
            BatchNorm2d(planes)
        )
        self.localdecode = nn.Sequential(
            nn.Conv2d(planes//2, planes//2, 3, groups=planes//2, stride=2, padding=1, bias=False),
            BatchNorm2d(planes//2),
            nn.Conv2d(planes//2, planes//2, 3, groups=planes//2, stride=1, padding=1, bias=False),
            BatchNorm2d(planes//2),
            nn.Conv2d(planes//2, planes//2, 3, groups=planes//2, stride=1, padding=1, bias=False),
            BatchNorm2d(planes//2)
        )
        self.cpeencode = RepCPE(planes, planes)
        self.cpedecode = RepCPE(planes//2, planes//2)
        self.gcn_local_attentiondecode = SpatialGCN(planes//2)

        self.finaldecode = nn.Sequential(nn.Conv2d(planes//2+planes, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes)) # 192

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat, featdecode):

        # # # # Local # # # #
        x = feat
        localtoken = self.localdecode(featdecode)
        localfeat = self.local(feat)
        # localfeat = self.cpeencode(feat)
        # localtoken = self.cpedecode(featdecode)
        local = self.gcn_local_attentiondecode(localfeat, localtoken)

        local = F.interpolate(local, size=featdecode.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = featdecode * local + featdecode

        out = spatial_local_feat
        # out = self.finaldecode(out)
        return out
# if __name__ == "__main__":
#     input0 = torch.rand((8, 384, 32, 32)).cuda(0)
#     input1 = torch.rand((8,192, 32, 32)).cuda(0)
#     dec1 = CFGCN(384).cuda(0)
#     output1 = dec1(input0,input1) #8 128 32 32
#     print("Out shape: " + str(output1.shape) ) # 8 128 32 32

class CFGCNHead(nn.Module):
    def __init__(self, inplanes, interplanes):
        super(CFGCNHead, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.dualgcn = CFGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, padding=1, dilation=1, bias=False),
        #     BatchNorm2d(interplanes),
        #     nn.ReLU(interplanes),
        #     nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # )


    def forward(self, x, y):
        if y is None:
            # gcnencoder
            output = self.conva(x)
            output = self.dualgcn(output, None)
        else:
            # gcndecoder
            output = self.conva(x)
            output = self.dualgcn(output, y)
        output = self.convb(output)
        return output
# if __name__ == "__main__":
#     input0 = torch.rand((8, 128, 32, 32)).cuda(0)
#     input1 = torch.rand((8, 64, 32, 32)).cuda(0)
#     dec1 = CFGCNHead(128,128).cuda(0)
#     output1 = dec1(input0,input1)
#     print("Out shape: " + str(output1.shape) ) # 8 128 32 32

class RepCPE(nn.Module):
    """
    This implementation of reparameterized conditional positional encoding was originally implemented
    in the following repository: https://github.com/apple/ml-fastvit

    Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """

    def __init__(
            self,
            in_channels,
            embed_dim,
            spatial_shape=(7, 7),
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepCPE, self).__init__()
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim
        self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor):
            x = self.pe(x) + x
            return x

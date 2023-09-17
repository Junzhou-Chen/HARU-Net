""" Full assembly of the parts to form the complete network """
import torch

from .HARUNet_parts import *


class HARUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        out_ch = n_classes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        cfg = {
            # height, in_ch, mid_ch, out_ch, RSU4F, side
            "encode": [[7, 3, 32, 64, False, False],  # En1
                       [6, 64, 32, 128, False, False],  # En2
                       [5, 128, 64, 256, False, False],  # En3
                       [4, 256, 128, 512, False, False],  # En4
                       [4, 512, 256, 512, True, False],  # En5
                       [4, 512, 256, 512, True, True]],  # En6
            # height, in_ch, mid_ch, out_ch, RSU4F, side
            "decode": [[4, 1024, 256, 512, True, True],  # De5
                       [4, 1024, 128, 256, False, True],  # De4
                       [5, 512, 64, 128, False, True],  # De3
                       [6, 256, 32, 64, False, True],  # De2
                       [7, 128, 16, 64, False, True]]  # De1
        }
        self.encode_num = len(cfg["encode"])

        # Encode parts
        self.En_1 = RSU(*cfg["encode"][0][:4])
        self.En_2 = RSU(*cfg["encode"][1][:4])
        self.En_3 = RSU(*cfg["encode"][2][:4])
        self.En_4 = RSU(*cfg["encode"][3][:4])
        self.En_5 = RSU4F(*cfg["encode"][4][1:4])
        self.En_6 = RSU4F(*cfg["encode"][5][1:4])

        # Decode parts
        self.De_5 = RSU4F(*cfg["decode"][0][1:4])
        self.De_4 = RSU(*cfg["decode"][1][:4])
        self.De_3 = RSU(*cfg["decode"][2][:4])
        self.De_2 = RSU(*cfg["decode"][3][:4])
        self.De_1 = RSU(*cfg["decode"][4][:4])

        # Attention parts
        self.Att1 = CBAMLayer(channel=64)
        self.Att2 = CBAMLayer(channel=128)
        self.Att3 = CBAMLayer(channel=256)
        self.Att4 = CBAMLayer(channel=512)
        self.Att5 = ChannelAttention(channel=512)

        # Side parts
        self.side_6 = nn.Conv2d(cfg["encode"][5][3], out_ch, kernel_size=3, padding=1)
        self.side_5 = nn.Conv2d(cfg["decode"][0][3], out_ch, kernel_size=3, padding=1)
        self.side_4 = nn.Conv2d(cfg["decode"][1][3], out_ch, kernel_size=3, padding=1)
        self.side_3 = nn.Conv2d(cfg["decode"][2][3], out_ch, kernel_size=3, padding=1)
        self.side_2 = nn.Conv2d(cfg["decode"][3][3], out_ch, kernel_size=3, padding=1)
        self.side_1 = nn.Conv2d(cfg["decode"][4][3], out_ch, kernel_size=3, padding=1)

        # Edge parts
        self.edge_6 = nn.Conv2d(cfg["encode"][5][3], out_ch, kernel_size=3, padding=1)
        self.edge_5 = nn.Conv2d(cfg["decode"][0][3], out_ch, kernel_size=3, padding=1)
        self.edge_4 = nn.Conv2d(cfg["decode"][1][3], out_ch, kernel_size=3, padding=1)
        self.edge_3 = nn.Conv2d(cfg["decode"][2][3], out_ch, kernel_size=3, padding=1)
        self.edge_2 = nn.Conv2d(cfg["decode"][3][3], out_ch, kernel_size=3, padding=1)
        self.edge_1 = nn.Conv2d(cfg["decode"][4][3], out_ch, kernel_size=3, padding=1)

        # Fusion parts
        self.Fusion1 = Fusion(self.encode_num * out_ch, out_ch)
        self.Fusion2 = Fusion(self.encode_num * out_ch, out_ch)

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.shape

        X_1 = x = self.En_1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        X_2 = x = self.En_2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        X_3 = x = self.En_3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        X_4 = x = self.En_4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        X_5 = x = self.En_5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        D_6 = x = self.En_6(x)

        x = F.interpolate(x, size=X_5.shape[2:], mode='bilinear', align_corners=False)
        X_5 = self.Att5(x=X_5)
        D_5 = x = self.De_5(torch.concat([x, X_5], dim=1))

        x = F.interpolate(x, size=X_4.shape[2:], mode='bilinear', align_corners=False)
        X_4 = self.Att4(x=X_4)
        D_4 = x = self.De_4(torch.concat([x, X_4], dim=1))

        x = F.interpolate(x, size=X_3.shape[2:], mode='bilinear', align_corners=False)
        X_3 = self.Att3(x=X_3)
        D_3 = x = self.De_3(torch.concat([x, X_3], dim=1))

        x = F.interpolate(x, size=X_2.shape[2:], mode='bilinear', align_corners=False)
        X_2 = self.Att2(x=X_2)
        D_2 = x = self.De_2(torch.concat([x, X_2], dim=1))

        x = F.interpolate(x, size=X_1.shape[2:], mode='bilinear', align_corners=False)
        X_1 = self.Att1(x=X_1)
        D_1 = self.De_1(torch.concat([x, X_1], dim=1))

        side_1 = F.interpolate(self.side_1(D_1), size=[h, w], mode='bilinear', align_corners=False)
        side_2 = F.interpolate(self.side_2(D_2), size=[h, w], mode='bilinear', align_corners=False)
        side_3 = F.interpolate(self.side_3(D_3), size=[h, w], mode='bilinear', align_corners=False)
        side_4 = F.interpolate(self.side_4(D_4), size=[h, w], mode='bilinear', align_corners=False)
        side_5 = F.interpolate(self.side_5(D_5), size=[h, w], mode='bilinear', align_corners=False)
        side_6 = F.interpolate(self.side_6(D_6), size=[h, w], mode='bilinear', align_corners=False)
        sides = [side_6, side_5, side_4, side_3, side_2, side_1]

        edge_1 = F.interpolate(self.edge_1(D_1), size=[h, w], mode='bilinear', align_corners=False)
        edge_2 = F.interpolate(self.edge_2(D_2), size=[h, w], mode='bilinear', align_corners=False)
        edge_3 = F.interpolate(self.edge_3(D_3), size=[h, w], mode='bilinear', align_corners=False)
        edge_4 = F.interpolate(self.edge_4(D_4), size=[h, w], mode='bilinear', align_corners=False)
        edge_5 = F.interpolate(self.edge_5(D_5), size=[h, w], mode='bilinear', align_corners=False)
        edge_6 = F.interpolate(self.edge_6(D_6), size=[h, w], mode='bilinear', align_corners=False)
        edges = [edge_6, edge_5, edge_4, edge_3, edge_2, edge_1]
        mask = self.Fusion1(torch.concat(sides, dim=1))
        edge = self.Fusion2(torch.concat(edges, dim=1))

        if self.training:
            return [mask] + sides, [edge] + edges
        else:
            return torch.sigmoid(mask), torch.sigmoid(edge)

    def use_checkpointing(self):
        self.En_1 = torch.utils.checkpoint(self.En_1)
        self.En_2 = torch.utils.checkpoint(self.En_2)
        self.En_3 = torch.utils.checkpoint(self.En_3)
        self.En_4 = torch.utils.checkpoint(self.En_4)
        self.En_5 = torch.utils.checkpoint(self.En_5)
        self.En_6 = torch.utils.checkpoint(self.En_6)
        self.De_1 = torch.utils.checkpoint(self.De_1)
        self.De_2 = torch.utils.checkpoint(self.De_2)
        self.De_3 = torch.utils.checkpoint(self.De_3)
        self.De_4 = torch.utils.checkpoint(self.De_4)
        self.De_5 = torch.utils.checkpoint(self.De_5)
        self.Att1 = torch.utils.checkpoint(self.Att1)
        self.Att2 = torch.utils.checkpoint(self.Att2)
        self.Att3 = torch.utils.checkpoint(self.Att3)
        self.Att4 = torch.utils.checkpoint(self.Att4)
        self.Att5 = torch.utils.checkpoint(self.Att5)
        self.side_1 = torch.utils.checkpoint(self.side_1)
        self.side_2 = torch.utils.checkpoint(self.side_2)
        self.side_3 = torch.utils.checkpoint(self.side_3)
        self.side_4 = torch.utils.checkpoint(self.side_4)
        self.side_5 = torch.utils.checkpoint(self.side_5)
        self.side_6 = torch.utils.checkpoint(self.side_6)
        self.edge_1 = torch.utils.checkpoint(self.edge_1)
        self.edge_2 = torch.utils.checkpoint(self.edge_1)
        self.edge_3 = torch.utils.checkpoint(self.edge_3)
        self.edge_4 = torch.utils.checkpoint(self.edge_4)
        self.edge_5 = torch.utils.checkpoint(self.edge_5)
        self.edge_6 = torch.utils.checkpoint(self.edge_6)
        self.Fusion1 = torch.utils.checkpoint(self.Fusion1)
        self.Fusion2 = torch.utils.checkpoint(self.Fusion2)




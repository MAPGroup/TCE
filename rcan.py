import torch
import torch.nn as nn


class RC_CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(RC_CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True)
    ):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(
                    n_feat,
                    n_feat,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                )
            )
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(RC_CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=act,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(
            nn.Conv2d(
                n_feat,
                n_feat,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


class rcan(nn.Module):
    def __init__(self):
        super(rcan, self).__init__()
        n_resgroups = 5
        n_resblocks = 8
        n_feats_y = 24
        n_feats_c = 16
        kernel_size = 3
        reduction = 8
        act = nn.ReLU(True)

        modules_head_y = [
            nn.Conv2d(
                1,
                n_feats_y,
                kernel_size,
                padding=kernel_size // 2,
            )
        ]
        modules_body_y = [
            ResidualGroup(
                n_feats_y,
                kernel_size,
                reduction,
                act=act,
                n_resblocks=n_resblocks,
            )
            for _ in range(n_resgroups)
        ]
        modules_body_y.append(
            nn.Conv2d(
                n_feats_y,
                n_feats_y,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        modules_tail_y = [
            nn.Conv2d(
                n_feats_y,
                1,
                kernel_size,
                padding=kernel_size // 2,
            )
        ]

        modules_head_c = [
            nn.Conv2d(
                3,
                n_feats_c,
                kernel_size,
                padding=kernel_size // 2,
            )
        ]
        modules_body_c = [
            ResidualGroup(
                n_feats_c,
                kernel_size,
                reduction,
                act=act,
                n_resblocks=n_resblocks,
            )
            for _ in range(n_resgroups)
        ]
        modules_body_c.append(
            nn.Conv2d(
                n_feats_c,
                n_feats_c,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        modules_tail_c = [
            nn.Conv2d(
                n_feats_c,
                3,
                kernel_size,
                padding=kernel_size // 2,
            )
        ]

        self.head_y = nn.Sequential(*modules_head_y)
        self.body_y = nn.Sequential(*modules_body_y)
        self.tail_y = nn.Sequential(*modules_tail_y)

        self.head_c = nn.Sequential(*modules_head_c)
        self.body_c = nn.Sequential(*modules_body_c)
        self.tail_c = nn.Sequential(*modules_tail_c)

        self.fuse_feat = nn.Conv2d(n_feats_y + n_feats_c, 32, kernel_size=1)

    def forward(self, x):
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        c = x - y.expand_as(x)

        y_feat0 = self.head_y(y)
        y_res = self.body_y(y_feat0)
        y_feat = y_res + y_feat0
        y_out = self.tail_y(y_feat)

        c_feat0 = self.head_c(c)
        c_res = self.body_c(c_feat0)
        c_feat = c_res + c_feat0
        c_out = self.tail_c(c_feat)

        out_img = y_out + c_out

        feat_cat = torch.cat([y_feat, c_feat], dim=1)
        out_feat = self.fuse_feat(feat_cat)

        return out_feat, out_img

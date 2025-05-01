import torch
import torch.nn as nn

# 引入自己的模块（路径按你实际项目）
from ultralytics.nn.modules.conv import Conv, Concat, CBAM
from ultralytics.nn.modules.block import C2f, SoftmaxBiFPNLayer, SPPF, MobileViTBlock
from ultralytics.nn.modules.head import Detect, PoseGCNHead


class DebugBiFPN_CBAM_MobileViT_PoseGCN(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        self.layers = nn.ModuleList([
            # ---------- Backbone ----------
            Conv(3, 64, 3, 2),                   # 0
            Conv(64, 128, 3, 2),                 # 1
            C2f(128, 128, 3, shortcut=True),     # 2
            Conv(128, 256, 3, 2),                # 3
            C2f(256, 256, 6, shortcut=True),     # 4 (P3)
            Conv(256, 512, 3, 2),                # 5
            C2f(512, 512, 6, shortcut=True),     # 6 (P4)
            Conv(512, 1024, 3, 2),               # 7
            C2f(1024, 1024, 3, shortcut=True),   # 8
            SPPF(1024, 1024, 5),                 # 9 (P5)

            # ---------- Neck ----------
            SoftmaxBiFPNLayer([256, 512, 1024], 256),  # 10
            CBAM(256, 7),                                # 11
            MobileViTBlock(256, 256, 3, 2, 128, 2),      # 12

            nn.Upsample(scale_factor=2, mode="nearest"), # 13
            Concat(1),                                   # 14
            C2f(768, 512, 1, shortcut=True),             # 15

            nn.Upsample(scale_factor=2, mode="nearest"), # 16
            Concat(1),                                   # 17
            C2f(768, 256, 1, shortcut=True),             # 18

            Conv(256, 256, 3, 2),                        # 19
            Concat(1),                                   # 20
            C2f(768, 512, 1, shortcut=True),             # 21

            Conv(512, 512, 3, 2),                        # 22
            Concat(1),                                   # 23
            C2f(768, 1024, 1, shortcut=True),            # 24

            # ✅ 传入 ch=(256, 512, 256) 符合 P5, P4, P3
            PoseGCNHead(nc, [17, 3], (256, 512, 256)),   # 25

            Detect(nc, [256, 512, 1024]),                # 26
        ])

    def forward(self, x):
        outputs = []
        for i, m in enumerate(self.layers):
            if i == 10:
                x = m([outputs[4], outputs[6], outputs[9]])      # P3, P4, P5
            elif i == 14:
                x = m([outputs[13], outputs[6]])                 # P5 + P4
            elif i == 17:
                x = m([outputs[16], outputs[4]])                 # P4 + P3
            elif i == 20:
                x = m([outputs[19], outputs[15]])                # (P3 down) + P4
            elif i == 23:
                x = m([outputs[22], outputs[11]])                # (P4 down) + P5
            elif i == 25:
                x = m([outputs[12], outputs[15], outputs[18]])   # PoseGCNHead 输入（P5, P4, P3）
            elif i == 26:
                x = m([outputs[18], outputs[21], outputs[24]])   # Detect 输入
            else:
                x = m(x)

            # ---------- Debug 打印 ----------
            name = m.__class__.__name__

            def _shape_repr(out):
                if isinstance(out, torch.Tensor):
                    return list(out.shape)
                elif isinstance(out, (list, tuple)):
                    return [list(t.shape) if isinstance(t, torch.Tensor) else str(type(t)) for t in out]
                else:
                    return str(type(out))

            print(f"[{i:02d}] {name:>25} → {_shape_repr(x)}")
            outputs.append(x)
        return outputs


if __name__ == "__main__":
    print("\n=======   Forward & Layer-wise debug   =======")
    model = DebugBiFPN_CBAM_MobileViT_PoseGCN(nc=4).eval()  # 例如 4 类任务
    inp = torch.randn(1, 3, 640, 640)
    _ = model(inp)

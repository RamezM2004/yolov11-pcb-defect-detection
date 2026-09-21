from pathlib import Path

# ============================================================
# Locate Ultralytics inside your venv
# ============================================================

ROOT = Path(r"C:\Users\USER\OneDrive - GJU\Desktop\4th Year\MI 2\Project")

BLOCK_PY = ROOT / r"yoloenv\Lib\site-packages\ultralytics\nn\modules\block.py"
INIT_PY  = ROOT / r"yoloenv\Lib\site-packages\ultralytics\nn\modules\__init__.py"
TASKS_PY = ROOT / r"yoloenv\Lib\site-packages\ultralytics\nn\tasks.py"

# ============================================================
# CBAM code
# ============================================================

CBAM_CODE = r'''

# ============================================================
# CBAM: Convolutional Block Attention Module
# Added for YOLO11s + CBAM experiment
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    def __init__(self, c1, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
'''

# ============================================================
# Helper functions
# ============================================================

def append_cbam_to_block():
    text = BLOCK_PY.read_text(encoding="utf-8")

    if "class CBAM(nn.Module)" in text:
        print("[OK] CBAM already exists in block.py")
        return

    text += CBAM_CODE
    BLOCK_PY.write_text(text, encoding="utf-8")
    print("[DONE] Added CBAM to block.py")


def update_init_py():
    text = INIT_PY.read_text(encoding="utf-8")

    # Add CBAM to block import
    if "from .block import" in text and "CBAM" not in text:
        lines = text.splitlines()
        new_lines = []

        for line in lines:
            if line.startswith("from .block import"):
                if line.rstrip().endswith(")"):
                    new_lines.append(line)
                else:
                    line = line.rstrip()
                    if line.endswith(","):
                        line += " CBAM,"
                    else:
                        line += ", CBAM"
                    new_lines.append(line)
            else:
                new_lines.append(line)

        text = "\n".join(new_lines) + "\n"

    # Add CBAM to __all__
    if "__all__" in text and '"CBAM"' not in text and "'CBAM'" not in text:
        text = text.replace("__all__ = (", '__all__ = (\n    "CBAM",')

    INIT_PY.write_text(text, encoding="utf-8")
    print("[DONE] Updated __init__.py")


def update_tasks_py():
    text = TASKS_PY.read_text(encoding="utf-8")

    # Add CBAM to import section if there is a modules import block
    if "CBAM" not in text:
        # Safer: add CBAM into global namespace by importing directly from block
        insert_line = "from ultralytics.nn.modules.block import CBAM\n"
        text = insert_line + text

    # Add CBAM to base_modules
    if "base_modules" in text and "CBAM" in text:
        # Find the frozenset block and insert CBAM after Conv if possible
        if "CBAM," not in text:
            text = text.replace("Conv,", "Conv,\n            CBAM,", 1)

    TASKS_PY.write_text(text, encoding="utf-8")
    print("[DONE] Updated tasks.py")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Adding CBAM to Ultralytics...")

    for p in [BLOCK_PY, INIT_PY, TASKS_PY]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    append_cbam_to_block()
    update_init_py()
    update_tasks_py()

    print("\nFinished.")
    print("Now test with:")
    print('python -c "from ultralytics.nn.modules import CBAM; print(CBAM)"')
import torch
import torch.nn as nn
from embeddings import TimestepMLP


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.res_proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        residual = self.res_proj(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + residual


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)
        attn_out, _ = self.attn(h, h, h)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return x + attn_out


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=96, time_dim=256):
        super().__init__()
        self.time_mlp = TimestepMLP(time_dim)

        self.down1 = Block(in_ch, base, time_dim)
        self.down2 = Block(base, base * 2, time_dim)
        self.down3 = Block(base * 2, base * 4, time_dim)
        self.pool = nn.MaxPool2d(2)

        self.bot1 = Block(base * 4, base * 4, time_dim)
        self.bot_attn = SelfAttention(base * 4)
        self.bot2 = Block(base * 4, base * 4, time_dim)

        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.dec3 = Block(base * 8, base * 2, time_dim)
        self.dec3_attn = SelfAttention(base * 2)

        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec2 = Block(base * 4, base, time_dim)

        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec1 = Block(base * 2, base, time_dim)

        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)

        b = self.bot1(self.pool(d3), t_emb)
        b = self.bot_attn(b)
        b = self.bot2(b, t_emb)

        u3 = self.up3(b)
        u3 = self.dec3(torch.cat([u3, d3], dim=1), t_emb)
        u3 = self.dec3_attn(u3)

        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, d2], dim=1), t_emb)

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, d1], dim=1), t_emb)

        return self.out(u1)


if __name__ == "__main__":
    model = UNet(base=96)
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        self.inc   = DoubleConv(in_ch, 64)     
        self.down1 = Down(64, 128)             
        self.down2 = Down(128, 256)            

        self.up1   = Up(256, 128)
        self.up2   = Up(128, 64)

        self.outc  = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
   
        x1 = self.inc(x)       
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)   
        x = self.up2(x, x1)   

        x = self.outc(x)

        return x

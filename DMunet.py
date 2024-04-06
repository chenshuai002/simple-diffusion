import torch, os
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from data import SteelDataset

class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)

            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):

        skip = nn.functional.interpolate(skip, size=(x.shape[2], x.shape[3]), mode='nearest')
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()

        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class Unet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7),
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        x1 = self.init_conv(x)
        down1 = self.down1(x1)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down2)
        up3 = self.up2(up2 + temb2, down1)
        up3 = nn.functional.interpolate(up3, size=(x1.shape[2], x1.shape[3]), mode='nearest')
        out = self.out(torch.cat((up3, x1), 1))
        return out

class DDPM(nn.Module):
    def __init__(self, model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.model = model.to(device)

        for k, v in self.ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def ddpm_schedules(self, beta1, beta2, T):

        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1  # 生成beta1-beta2均匀分布的数组
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,
            "oneover_sqrta": oneover_sqrta,
            "sqrt_beta_t": sqrt_beta_t,
            "alphabar_t": alphabar_t,
            "sqrtab": sqrtab,
            "sqrtmab": sqrtmab,
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,
        }

    def forward(self, x):

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise

        )

        ys = self.model(x_t, _ts / self.n_T)
        noise = nn.functional.interpolate(noise, size=(ys.shape[2], ys.shape[3]), mode='nearest')
        return self.loss_mse(noise, ys)

    def sample(self, n_sample, size, device):

        x_i = torch.randn(n_sample, *size).to(device)
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.model(x_i, t_is)
            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return x_i

class ImageGenerator(object):
    def __init__(self):
        self.epoch = 5
        self.sample_num = 1
        self.batch_size = 36
        self.lr = 0.0001
        self.n_T = 400
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = 'best_model.pth'
        self.model = Unet(in_channels=1)
        self.sampler = DDPM(model=self.model, betas=(1e-4, 0.02), n_T=self.n_T, device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.sampler.model.parameters(), lr=self.lr)
        self.best_val_loss = float('inf')  # 初始化为正无穷大
        self.init_dataloader()

    def init_dataloader(self):
        image_dir = 'images/1/sel'
        target_dir = 'images/1/target_sel'
        self.train_dataloader, self.val_dataloader = SteelDataset(image_dir, target_dir, batch_size=1)

    def train(self):
        self.sampler.train()
        print('训练开始!!')
        for epoch in range(self.epoch):
            self.sampler.model.train()
            loss_mean = 0.0
            for i, (images, labels) in enumerate(self.train_dataloader):
                if images.shape[2] < 50 or images.shape[3] < 30:
                    continue

                images, labels = images.to(self.device), labels.to(self.device)

                # 将latent和condition拼接后输入网络
                loss = self.sampler(images)
                loss_mean += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss = loss_mean / len(self.train_dataloader)
            print('epoch:{}, train loss:{:.4f}'.format(epoch+1, train_loss))
            val_loss = self.validate(epoch)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(epoch)

    def validate(self, epoch):
        self.sampler.model.eval()
        loss_mean = 0.0
        for i, (image, label) in enumerate(self.val_dataloader):
            if image.shape[2] < 50 or image.shape[3] < 30:
                continue
            image, label = image.to(self.device), label.to(self.device)
            loss = self.sampler(image)
            loss_mean += loss.item()
        val_loss = loss_mean / len(self.val_dataloader)
        print('epoch:{}, val loss:{:.4f}'.format(epoch+1, val_loss))
        return val_loss

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), self.model_path)
        print(f'Model saved at {self.model_path} with best val loss: {self.best_val_loss:.4f}')
        if epoch+1 == self.epoch:
            self.visualize_results()

    def visualize_results(self):
        self.sampler.eval()
        # 保存结果路径
        output_path = 'images/test/pred_sel'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        image_dir = 'images/test/sel'
        t = torch.zeros(1, device=self.device)
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path).convert('L')
                image_tensor = ToTensor()(image).unsqueeze(0).to(self.device)
                if image_tensor.shape[2] < 50 or image_tensor.shape[3] < 30:
                    continue
                with torch.no_grad():
                    output = self.sampler.model(image_tensor, t)

                output_img = ToPILImage(mode='L')(output.squeeze(0).cpu())
                save_path = os.path.join(output_path, f'pred_{filename}')
                output_img.save(save_path)

        print('预测结果保存完成！')

if __name__ == '__main__':
    generator = ImageGenerator()
    generator.train()

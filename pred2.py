import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage
from DMunet import Unet

class ImageProcessor:
    def __init__(self, model_path, test_dir, output_dir):
        self.model_path = model_path
        self.test_dir = test_dir
        self.output_dir = output_dir

    def load_model(self):
        self.model = Unet(in_channels=1)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def process_images(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for filename in os.listdir(self.test_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.test_dir, filename)
                img = Image.open(image_path).convert('L')
                img_tensor = ToTensor()(img).unsqueeze(0)
                if img_tensor.shape[2] < 50 or img_tensor.shape[3] < 30:
                    continue

                with torch.no_grad():
                    t = torch.zeros(1)
                    output = self.model(img_tensor, t)

                output_img = ToPILImage()(output.squeeze(0))

                save_path = os.path.join(self.output_dir, f'pred_{filename}')
                output_img.save(save_path)

        print('预测结果保存完成！')

model_path = 'best_model.pth'
test_dir = 'images/test/sel'
output_dir = 'images/test/pred_target'

processor = ImageProcessor(model_path, test_dir, output_dir)
processor.load_model()
processor.process_images()

import torchvision.transforms as T
from PIL import Image

class ResNetPreprocessor:
    """ResNet图像预处理类，支持自定义输入尺寸和标准化参数
    
    参数说明：
        target_size (int): 最终裁剪后的正方形边长（默认448）
        scale_factor (float): 初始缩放系数（默认1.2，即缩放到target_size*1.2后裁剪）
        mean (list): 标准化均值（默认ImageNet参数）
        std (list): 标准化标准差（默认ImageNet参数）
    """
    def __init__(self, 
                 target_size=448, 
                 scale_factor=1.2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.mean = mean
        self.std = std
        
        # 构建预处理流水线
        self.transform = T.Compose([
            T.Resize(int(self.target_size * self.scale_factor)),
            T.CenterCrop(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
    
    def __call__(self, img):
        """执行预处理操作
        
        参数：
            img (PIL.Image.Image): 输入图像
            
        返回：
            torch.Tensor: 处理后的张量，形状为(C, H, W)
        """
        return self.transform(img)
    
    def update_parameters(self, 
                         target_size=None, 
                         scale_factor=None,
                         mean=None,
                         std=None):
        """动态更新预处理参数"""
        if target_size is not None:
            self.target_size = target_size
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = std
        
        # 重新构建预处理流程
        self.transform = T.Compose([
            T.Resize(int(self.target_size * self.scale_factor)),
            T.CenterCrop(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def __repr__(self):
        """显示当前配置信息"""
        return f"ResNetPreprocessor(target_size={self.target_size}, " \
               f"scale_factor={self.scale_factor}, " \
               f"mean={self.mean}, std={self.std})"
"""
Perceptual Loss for SAR Super-Resolution

Implements a VGG-based perceptual loss optimized for SAR images.
Uses features from pre-trained VGG19 network (before activation) to capture
texture and structural details better than pixel-wise losses.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 based Perceptual Loss.
    
    Extracts features from intermediate layers of VGG19 and computes
    L1 distance between generated and target images.
    
    SAR-specific adaptations:
    - Handles single-channel input by repeating to 3 channels (RGB)
    - Uses pre-activation features (before ReLU) to preserve weak texture details
      often found in SAR images (following ESRGAN approach).
    """
    
    def __init__(self, layer_weights: dict = None, style_weights: dict = None, use_input_norm: bool = True, device: str = 'cpu'):
        """
        Args:
            layer_weights: Dict of layer indices to weights for Content Loss. 
                           Default uses conv4_4 ('26').
            style_weights: Dict of layer indices to weights for Style Loss.
                           Default uses relu1_2 ('3'), relu2_2 ('8'), relu3_3 ('15'), relu4_3 ('24').
            use_input_norm: Whether to normalize input with ImageNet stats.
            device: Device to load the VGG model on.
        """
        super().__init__()
        
        self.use_input_norm = use_input_norm
        
        # Content loss layers: default 'conv4_4' (index 26)
        if layer_weights is None:
            self.layer_weights = {'26': 1.0}
        else:
            self.layer_weights = layer_weights
            
        # Style loss layers: default relu1_2, relu2_2, relu3_3, relu4_3
        # Indices based on torchvision.models.vgg19 features
        if style_weights is None:
            self.style_weights = {'3': 1.0, '8': 1.0, '15': 1.0, '24': 1.0}
        else:
            self.style_weights = style_weights
            
        # Load VGG19 pre-trained on ImageNet
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # Extract features module and freeze parameters
        self.features = vgg.features
        for param in self.parameters():
            param.requires_grad = False
            
        self.features.eval()
        
        # ImageNet normalization statistics
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize_input(self, x):
        """Normalize input tensor with ImageNet mean and std"""
        return (x - self.mean) / self.std

    def _preprocess(self, x):
        """
        Preprocess SAR input:
        1. Repeat 1-channel to 3-channel (if needed)
        2. Normalize (if enabled)
        """
        # Handle grayscale (B, 1, H, W) -> (B, 3, H, W)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        if self.use_input_norm:
            x = self._normalize_input(x)
            
        return x
        
    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        # Force float32 for Gram matrix calculation to avoid overflow in fp16/AMP
        features = input.view(a, b, c * d).float()  # resise F_XL into \hat F_XL

        G = torch.bmm(features, features.transpose(1, 2))  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature map.
        return G.div(b * c * d)

    def forward(self, input_img, target_img):
        """
        Compute perceptual and style loss between input (SR) and target (HR) images.
        Returns:
            content_loss, style_loss
        """
        # Preprocess
        input_img = self._preprocess(input_img)
        target_img = self._preprocess(target_img)
        
        content_loss = 0.0
        style_loss = 0.0
        
        x = input_img
        y = target_img
        
        # Unions of all layers we need
        content_layers = set(self.layer_weights.keys())
        style_layers = set(self.style_weights.keys())
        all_layers = content_layers.union(style_layers)
        
        # Iterate through VGG layers
        sorted_indices = sorted([int(k) for k in all_layers])
        if not sorted_indices:
            return content_loss, style_loss
            
        max_idx = sorted_indices[-1]
        
        for name, layer in self.features.named_children():
            idx = int(name)
            
            # Pass through layer
            x = layer(x)
            y = layer(y)
            
            idx_str = str(idx)
            
            # Content Loss
            if idx_str in self.layer_weights:
                weight = self.layer_weights[idx_str]
                content_loss += weight * nn.functional.l1_loss(x, y)
            
            # Style Loss
            if idx_str in self.style_weights:
                weight = self.style_weights[idx_str]
                gram_x = self.gram_matrix(x)
                gram_y = self.gram_matrix(y)
                style_loss += weight * nn.functional.l1_loss(gram_x, gram_y)
                
            # Stop if we went past the last needed layer
            if idx >= max_idx:
                break
                
        return content_loss, style_loss

if __name__ == "__main__":
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = VGGPerceptualLoss().to(device)
    
    # Random SAR images (B, 1, 256, 256) ranges [-1, 1] assuming normalized input
    # Note: VGG expects roughly [0, 1] range after denormalization from model input, 
    # but here we assume input to loss is [0, 1] (standard for perceptual loss input).
    # If model output is [-1, 1], we should denorm first. 
    # Let's assume input is [0, 1] for this test.
    sr = torch.rand(2, 1, 256, 256).to(device)
    hr = torch.rand(2, 1, 256, 256).to(device)
    
    c_loss, s_loss = loss_fn(sr, hr)
    print(f"Content Loss: {c_loss.item()}")
    print(f"Style Loss: {s_loss.item()}")

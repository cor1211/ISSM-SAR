import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

# Copy of the class for standalone testing
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_weights: dict = None, style_weights: dict = None, use_input_norm: bool = True, device: str = 'cpu'):
        super().__init__()
        
        self.use_input_norm = use_input_norm
        
        if layer_weights is None:
            self.layer_weights = {'26': 1.0}
        else:
            self.layer_weights = layer_weights
            
        if style_weights is None:
            self.style_weights = {'3': 1.0, '8': 1.0, '15': 1.0, '24': 1.0}
        else:
            self.style_weights = style_weights
            
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        for param in self.parameters():
            param.requires_grad = False
        self.features.eval()
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize_input(self, x):
        return (x - self.mean) / self.std

    def _preprocess(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.use_input_norm:
            x = self._normalize_input(x)
        return x
        
    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a, b, c * d).float()
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(b * c * d)

    def forward(self, input_img, target_img):
        # Preprocess
        input_img = self._preprocess(input_img)
        target_img = self._preprocess(target_img)
        
        # Ensure inputs are valid
        if torch.isnan(input_img).any() or torch.isinf(input_img).any():
             input_img = torch.nan_to_num(input_img, nan=0.0, posinf=1.0, neginf=0.0)
        if torch.isnan(target_img).any() or torch.isinf(target_img).any():
             target_img = torch.nan_to_num(target_img, nan=0.0, posinf=1.0, neginf=0.0)
        
        content_loss = 0.0
        style_loss = 0.0
        
        x = input_img
        y = target_img
        
        content_layers = set(self.layer_weights.keys())
        style_layers = set(self.style_weights.keys())
        all_layers = content_layers.union(style_layers)
        
        sorted_indices = sorted([int(k) for k in all_layers])
        if not sorted_indices: return content_loss, style_loss
        max_idx = sorted_indices[-1]
        
        for name, layer in self.features.named_children():
            idx = int(name)
            x = layer(x)
            y = layer(y)
            idx_str = str(idx)
            
            # Content Loss
            if idx_str in self.layer_weights:
                weight = self.layer_weights[idx_str]
                if torch.isnan(x).any() or torch.isinf(x).any():
                     x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
                loss = nn.functional.l1_loss(x, y)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    content_loss += weight * loss
            
            # Style Loss
            if idx_str in self.style_weights:
                weight = self.style_weights[idx_str]
                if torch.isnan(x).any(): x = torch.nan_to_num(x)
                gram_x = self.gram_matrix(x)
                gram_y = self.gram_matrix(y)
                if torch.isnan(gram_x).any(): gram_x = torch.nan_to_num(gram_x)
                if torch.isnan(gram_y).any(): gram_y = torch.nan_to_num(gram_y)
                loss = nn.functional.l1_loss(gram_x, gram_y)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    style_loss += weight * loss
                
            if idx >= max_idx: break
        return content_loss, style_loss

def test_vgg_robustness():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")
    
    loss_fn = VGGPerceptualLoss().to(device)
    loss_fn.eval()
    
    # 1. Test clean inputs
    print("Testing clean inputs...")
    sr = torch.rand(2, 1, 256, 256).to(device)
    hr = torch.rand(2, 1, 256, 256).to(device)
    c_loss, s_loss = loss_fn(sr, hr)
    print(f"Clean - Content: {c_loss.item()}, Style: {s_loss.item()}")
    
    # 2. Test input NaN
    print("Testing input NaN...")
    sr_nan = sr.clone()
    sr_nan[0, 0, 100, 100] = float('nan')
    c_loss, s_loss = loss_fn(sr_nan, hr)
    print(f"Input NaN - Content: {c_loss.item()}, Style: {s_loss.item()}")
    assert not torch.isnan(c_loss) and not torch.isinf(c_loss)
    assert not torch.isnan(s_loss) and not torch.isinf(s_loss)
    
    # 3. Test input Inf
    print("Testing input Inf...")
    sr_inf = sr.clone()
    sr_inf[0, 0, 100, 100] = float('inf')
    c_loss, s_loss = loss_fn(sr_inf, hr)
    print(f"Input Inf - Content: {c_loss.item()}, Style: {s_loss.item()}")
    assert not torch.isnan(c_loss) and not torch.isinf(c_loss)
    assert not torch.isnan(s_loss) and not torch.isinf(s_loss)
    
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    try:
        test_vgg_robustness()
    except Exception as e:
        print(f"TEST FAILED: {e}")

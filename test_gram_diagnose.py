import torch

def test_gram_overflow():
    print("--- Testing Gram Matrix Overflow in Float16 ---")
    
    # Simulate a typical feature map from VGG (e.g. relu2_2: 128 channels, 128x128 size)
    B, C, H, W = 1, 128, 128, 128
    N = H * W # 16384 pixels
    
    # Create random features in float32 first, typical ReLU range [0, 10]
    features_32 = torch.rand(B, C, N) * 5.0
    
    # 1. Convert to float16 (Simulating AMP / Mixed Precision)
    features_16 = features_32.half().cuda()
    
    print(f"Feature Map (float16) - Max: {features_16.max().item():.4f}")
    
    # 2. Compute Gram Matrix in float16
    # G = F @ F.T
    try:
        gram_16 = torch.bmm(features_16, features_16.transpose(1, 2))
        print(f"Gram Matrix (float16 accumulation) - Max: {gram_16.max().item()}")
        
        if torch.isinf(gram_16).any():
            print(">> RESULT: OVERFLOW DETECTED in float16! (Inf values found)")
        else:
            print(">> RESULT: No overflow in float16 (unexpected for large maps)")
            
    except Exception as e:
        print(f"Error in float16 computation: {e}")

    # 3. Compute Gram Matrix with Cast to float32
    print("\n--- Testing Gram Matrix with Float32 Cast ---")
    features_cast = features_16.float()
    gram_32 = torch.bmm(features_cast, features_cast.transpose(1, 2))
    
    print(f"Gram Matrix (float32 accumulation) - Max: {gram_32.max().item()}")
    
    # Normalize
    gram_norm = gram_32 / (B * C * N)
    print(f"Normalized Gram Matrix - Max: {gram_norm.max().item()}")
    
    if not torch.isinf(gram_32).any():
        print(">> RESULT: SUCCESS with float32 cast. No overflow.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_gram_overflow()
    else:
        print("CUDA not available, cannot test float16 behavior accurately.")

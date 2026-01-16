from transformers import DynamicCache
import torch

try:
    print("Creating Cache...")
    c = DynamicCache()
    print("Cache created.")
    
    B, H, S, D = 1, 2, 8, 64
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    
    print("Updating Layer 0...")
    c.update(k, v, 0)
    print("Layer 0 Updated.")
    
    print("Updating Layer 1...")
    c.update(k, v, 1)
    print("Layer 1 Updated.")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

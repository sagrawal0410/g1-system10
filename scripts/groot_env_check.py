import torch, gr00t, flash_attn
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "ndev", torch.cuda.device_count(), "flash_attn", flash_attn.__version__, "gr00t OK")
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda:0")
    y = (x @ x).sum().item()
    print("cuda matmul OK, name:", torch.cuda.get_device_name(0))

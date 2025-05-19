import torch
print(torch.cuda.is_available())            # Should be True
print(torch.cuda.get_device_name(0))        # Should print RTX 3050

import torch
import torch.nn as nn

class MatchedModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.layers = net
        
    def load_state_dict(self, state_dict, strict=True):
        if 'layers.0.sigma' in state_dict:
            state_dict['layers.0.std'] = state_dict.pop('layers.0.sigma')
        return super().load_state_dict(state_dict, strict)

    def forward(self, x):
        return self.layers(x)

def get_ctbench_model(*args, **kwargs):
    # Load the exported PyTorch model structure from the parent certify loop
    model_path = kwargs.get("model_path", "../tmp/ctbench_abcrown_model.pt")
    torch_net = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    model = MatchedModel(torch_net)
    model.eval()
    return model

def get_ctbench_data(*args, **kwargs):
    spec = args[0] if len(args) > 0 else kwargs.get("spec", {})
    eps = spec.get("epsilon", 0.0)

    # Load the specific unverified dataset batch from the cache
    data_path = kwargs.get("data_path", "../tmp/ctbench_abcrown_data.pt")
    data = torch.load(data_path, weights_only=False)
    x, y = data[0].cpu(), data[1].cpu()
    
    # Provide explicitly raw 0-1 bounded values to satisfy ABCROWN specification checks
    data_max = torch.ones_like(x)
    data_min = torch.zeros_like(x)
    return x, y, data_max, data_min, eps

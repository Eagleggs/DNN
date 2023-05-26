import torch
from transformerLite import TransformerLite

def gpu2cpu(gpu_model_path, cpu_model_path):
    device = torch.device('cpu')
    model = torch.load(gpu_model_path, map_location=device)
    torch.save(model, cpu_model_path)

def pc2android(pc_model_path, android_model_path):
    # model = TransformerLite(1500, 2, 8)
    model = torch.load(pc_model_path)
    model.eval()
    x = torch.rand(15, 1500, 2)
    traced_script_module = torch.jit.trace(model, x)
    traced_script_module.save(android_model_path)

if __name__ == "__main__":
    gpu2cpu('../model_best.pt', 'model_cpu.pt')
    pc2android('model_cpu.pt', 'model_1.pt')
import torch
from transformerLite import TransformerLite
from torch.utils.mobile_optimizer import optimize_for_mobile



def gpu2cpu(gpu_model_path, cpu_model_path):
    device = torch.device('cpu')
    model = TransformerLite(333, 501, 16)
    model.load_state_dict(torch.load(gpu_model_path, map_location=device))
    torch.save(model, cpu_model_path)

def pc2android(pc_model_path, android_model_path):
    # model = TransformerLite(333, 501, 16)
    # model.load_state_dict(torch.load(pc_model_path))

    model = torch.load(pc_model_path)
    model.eval()
    x = torch.rand(1, 333, 501)
    traced_script_module = torch.jit.trace(model, x)
    # traced_script_module.save(android_model_path)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter("model_m.ptl")

if __name__ == "__main__":
    gpu2cpu('model_best_1.pt', 'model_cpu.pt')
    pc2android('model_cpu.pt', 'model_m.pt')

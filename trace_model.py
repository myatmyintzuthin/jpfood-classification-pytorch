import torch
import convert
import utils.utils as utils

def trace_model(torch_model_path, script_model_path, model_name, variant, width_multi, n_class):
    choose_model = convert.ConvertModel(model_name, variant, width_multi, n_class)
    model = choose_model.load_model()
    ckpt = torch.load(torch_model_path)
    torch_model = utils.load_ckpt(model, ckpt)
    torch_model.eval()
    
    dummy_input = torch.rand(1, 3, 224, 224)
    traced_script_model = torch.jit.trace(torch_model, dummy_input)
    traced_script_model.save(script_model_path)
    print(f"[INFO] Traced script model is saved as {script_model_path}")
    
    
if __name__ == "__main__":
    
    torch_model_path = 'experiments/resnet50_202303231411/resnet_food.pt'
    script_model_path = 'experiments/resnet50_202303231411/resnet_food_script.pt'
    model_name = 'resnet'
    variant = '50'
    width_multi = 1.0
    n_class = 4
    
    trace_model(torch_model_path, script_model_path, model_name, variant, width_multi, n_class)
    
    
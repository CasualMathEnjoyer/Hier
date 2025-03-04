import os

models = "/home/katka/Documents/Trained models/Transformer_encoder_decoder/"


print(f"{'Model Name':<40} {'Model Size':>12} {'RAM Estimate':>15}")
print("-" * 85)
for model_name in os.listdir(models):
    model_full_path = os.path.join(models, model_name)
    if not os.path.isfile(model_full_path): continue
    if '.keras' not in model_name: continue
    size = os.path.getsize(model_full_path) / (1024 ** 2)
    print(f"{model_name:<40} {size:>10.2f} MB {size*10:>10.2f} MB")
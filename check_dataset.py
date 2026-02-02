import numpy as np
import json
feats = np.load("example_data/toy-ppi-feats.npy")
print(f"Feature shape: {feats.shape}")
class_map = json.load(open("example_data/toy-ppi-class_map.json"))
example_val = list(class_map.values())[0]
if isinstance(example_val, list):
    print(f"Num classes (multi-label): {len(example_val)}")
else:
    print(f"Num classes: {len(set(class_map.values()))}")

import os
import json

dataset_path = './dataset2/train'
class_names = sorted(os.listdir(dataset_path))
class_indices = {class_name: i for i, class_name in enumerate(class_names)}

with open('class_labels.json', 'w') as f:
    json.dump(class_indices, f)

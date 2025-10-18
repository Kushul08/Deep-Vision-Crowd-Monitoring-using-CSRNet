import json

def merge_json_lists(json_files, output_json):
    combined_list = []
    for jf in json_files:
        with open(jf, 'r') as f:
            combined_list.extend(json.load(f))
    with open(output_json, 'w') as f:
        json.dump(combined_list, f, indent=2)
    print(f'Merged JSON saved to {output_json}')

# Merge training lists
merge_json_lists(
    ['/content/CSRNet-pytorch-master/part_A_train_h5.json', '/content/CSRNet-pytorch-master/part_B_train_h5.json'],
    '/content/CSRNet-pytorch-master/train_all_h5.json'
)

# Merge validation lists
merge_json_lists(
    ['/content/CSRNet-pytorch-master/part_A_val_h5.json', '/content/CSRNet-pytorch-master/part_B_val_h5.json'],
    '/content/CSRNet-pytorch-master/val_all_h5.json'
)

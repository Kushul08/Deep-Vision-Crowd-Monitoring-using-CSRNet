import json
import os

def update_json_list(json_path, processed_folder, suffix='.h5'):
    with open(json_path, 'r') as f:
        img_list = json.load(f)
    h5_list = [
        os.path.join(
            processed_folder,
            os.path.basename(path).replace(".jpg", suffix)
        )
        for path in img_list
    ]
    out_path = json_path.replace('.json', '_h5.json')
    with open(out_path, 'w') as f:
        json.dump(h5_list, f, indent=2)
    print(f'Updated JSON saved to: {out_path}')

# For Part A
update_json_list('/content/CSRNet-pytorch-master/part_A_train.json',
                 '/content/drive/MyDrive/Processed_Shanghai/part_A/train')
update_json_list('/content/CSRNet-pytorch-master/part_A_val.json',
                 '/content/drive/MyDrive/Processed_Shanghai/part_A/test') # or val if your split
update_json_list('/content/CSRNet-pytorch-master/part_A_test.json',
                 '/content/drive/MyDrive/Processed_Shanghai/part_A/test')
update_json_list('/content/CSRNet-pytorch-master/part_A_train_with_val.json',
                 '/content/drive/MyDrive/Processed_Shanghai/part_A/train')

# For Part B
update_json_list('/content/CSRNet-pytorch-master/part_B_train.json',
                 '/content/drive/MyDrive/Processed_Shanghai/part_B/train')
update_json_list('/content/CSRNet-pytorch-master/part_B_val.json',
                 '/content/drive/MyDrive/Processed_Shanghai/part_B/test') # or val if your split
update_json_list('/content/CSRNet-pytorch-master/part_B_test.json',
                 '/content/drive/MyDrive/Processed_Shanghai/part_B/test')
update_json_list('/content/CSRNet-pytorch-master/part_B_train_with_val.json',
                 '/content/drive/MyDrive/Processed_Shanghai/part_B/train')

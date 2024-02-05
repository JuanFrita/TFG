import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
import matplotlib.pyplot as plt
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


def merge_density_with_image(image_path, density_map, results_path, image_name):

    background_image = Image.open(image_path).convert('RGB')
    background_image_resized = background_image.resize((density_map.shape[1], density_map.shape[0]), Image.BILINEAR)

    density_map_normalized = density_map / np.max(density_map)  # Normalizar el mapa de densidad
    density_map_colored = plt.cm.jet(density_map_normalized)[:, :, :3]  # Us
    density_map_rgba = np.zeros((density_map.shape[0], density_map.shape[1], 4), dtype=np.uint8)
    density_map_rgba[..., :3] = density_map_colored * 255
    density_map_rgba[..., 3] = 75 #ajustar la transparencia del mapa de densidad

    density_map_image = Image.fromarray(density_map_rgba)
    background_image_resized_pil = Image.fromarray(np.array(background_image_resized))

    combined_image = Image.alpha_composite(background_image_resized_pil.convert('RGBA'), density_map_image)

    os.makedirs(results_path, exist_ok=True)

    save_path = os.path.join(results_path, image_name)
    combined_image.save(save_path)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    test_path = os.path.join(args.data_dir, 'test')
    datasets = Crowd(test_path, 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda') #cambiar a cuda en pc con nvidia
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(f"torch.sum {torch.sum(outputs)}")
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)
            density_map = outputs.squeeze().cpu().numpy()
            
            merge_density_with_image(os.path.join(f"{args.data_dir}/test",f"{name[0]}.jpg"), density_map, f"{args.save_dir}/results", f"{name[0]}.png")

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)


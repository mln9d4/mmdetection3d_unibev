import torch
import os
import matplotlib.pyplot as plt

def main():
    file_name = "vis_data.pt"
    path_name = "/home/mingdayang/mmdetection3d/outputs/inference/unibev_nus_LC_cnw_256_modality_dropout_mininuscene/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590"

    path = os.path.join(path_name, file_name)

    model = torch.load(path)
    print(type(model))
    print(model.keys())
    

    for key, value in model.items():
        print(f"{key}: {len(value)}")

    key = 'img_bev_embed'
    activation = model[key].squeeze(0)
    activation = activation.view(200, 200, -1)
    num_feature_maps = activation.shape[-1]

    plt.imshow(activation[:, :, 0].cpu().numpy(), cmap='viridis')
    plt.title("Feature map 0 from img_bev_embed")
    plt.colorbar()
    plt.savefig("dummy_fig.png")

    # # Create subplots to display all feature maps
    # fig, axarr = plt.subplots(num_feature_maps // 16, 16, figsize=(15, 15))  # Adjust grid size if needed

    # # Visualize each feature map
    # for idx in range(num_feature_maps):
    #     row = idx // 16
    #     col = idx % 16
    #     axarr[row, col].imshow(activation[:, :, idx].cpu().numpy(), cmap='viridis')
    #     axarr[row, col].axis('off')

    # fig.savefig("dummy_fig.png")




if __name__ == "__main__":
    main()
import os
from PIL import Image
import torch
from torchvision import transforms
from model_video import build_model
import numpy as np
import argparse

def debug_test(gpu_id, model_path, datapath, save_root_path, group_size, img_size, img_dir_name):
    print(f"DEBUG: Starting UFO test with:")
    print(f"  model_path: {model_path}")
    print(f"  datapath: {datapath}")
    print(f"  save_root_path: {save_root_path}")
    print(f"  group_size: {group_size}")
    print(f"  img_size: {img_size}")
    print(f"  img_dir_name: {img_dir_name}")
    
    # Build model
    print("DEBUG: Building model...")
    device = torch.device(gpu_id)
    net = build_model(device).to(device)
    net = torch.nn.DataParallel(net)
    
    print("DEBUG: Loading model weights...")
    net.load_state_dict(torch.load(model_path, map_location=gpu_id))
    net.eval()
    net = net.module.to(device)
    print("DEBUG: Model loaded successfully")

    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
    
    with torch.no_grad():
        for p in range(len(datapath)):
            print(f"DEBUG: Processing datapath {p}: {datapath[p]}")
            img_path = os.path.join(datapath[p], img_dir_name)
            print(f"DEBUG: Image path: {img_path}")
            
            if not os.path.exists(img_path):
                print(f"ERROR: Image path does not exist: {img_path}")
                continue
                
            all_class = os.listdir(img_path)
            print(f"DEBUG: Found classes: {all_class}")
            
            image_list, save_list = list(), list()
            for s in range(len(all_class)):
                print(f"DEBUG: Processing class {s}: {all_class[s]}")
                class_path = os.path.join(datapath[p], img_dir_name, all_class[s])
                image_path = sorted(os.listdir(class_path))
                print(f"DEBUG: Found {len(image_path)} images in class {all_class[s]}")
                print(f"DEBUG: First few images: {image_path[:5]}")
                
                idx=[]
                block_size=(len(image_path)+group_size-1)//group_size
                print(f"DEBUG: Block size: {block_size}")
                for ii in range(block_size):
                  cur=ii
                  while cur<len(image_path):
                    idx.append(cur)
                    cur+=block_size
                
                new_image_path=[]
                for ii in range(len(image_path)):
                  new_image_path.append(image_path[idx[ii]])
                image_path=new_image_path
                
                image_list.append(list(map(lambda x: os.path.join(datapath[p], img_dir_name, all_class[s], x), image_path)))
                save_list.append(list(map(lambda x: os.path.join(save_root_path[p], all_class[s], x[:-4]+'.png'), image_path)))
                
            print(f"DEBUG: Created {len(image_list)} image lists")
            
            for i in range(len(image_list)):
                print(f"DEBUG: Processing image list {i}")
                cur_class_all_image = image_list[i]
                print(f"DEBUG: Images in this class: {len(cur_class_all_image)}")
                
                cur_class_rgb = torch.zeros(len(cur_class_all_image), 3, img_size, img_size)
                for m in range(len(cur_class_all_image)):
                    if m < 3:  # Only print for first few images
                        print(f"DEBUG: Loading image {m}: {cur_class_all_image[m]}")
                    rgb_ = Image.open(cur_class_all_image[m])
                    if rgb_.mode == 'RGB':
                        rgb_ = img_transform(rgb_)
                    else:
                        rgb_ = img_transform_gray(rgb_)
                    cur_class_rgb[m, :, :, :] = rgb_

                cur_class_mask = torch.zeros(len(cur_class_all_image), img_size, img_size)
                divided = len(cur_class_all_image) // group_size
                rested = len(cur_class_all_image) % group_size
                print(f"DEBUG: divided: {divided}, rested: {rested}")
                
                if divided != 0:
                    print(f"DEBUG: Processing {divided} divided groups")
                    for k in range(divided):
                        if k < 2:  # Only print for first couple groups
                            print(f"DEBUG: Processing group {k}")
                        group_rgb = cur_class_rgb[(k * group_size): ((k + 1) * group_size)]
                        group_rgb = group_rgb.to(device)
                        _, pred_mask = net(group_rgb)
                        cur_class_mask[(k * group_size): ((k + 1) * group_size)] = pred_mask
                        
                if rested != 0:
                    print(f"DEBUG: Processing remaining {rested} images")
                    group_rgb_tmp_l = cur_class_rgb[-rested:]
                    group_rgb_tmp_r = cur_class_rgb[:group_size - rested]
                    group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                    group_rgb = group_rgb.to(device)
                    _, pred_mask = net(group_rgb)
                    cur_class_mask[(divided * group_size):] = pred_mask[:rested]

                print(f"DEBUG: Creating save directory: {save_root_path[p]}")
                class_save_path = os.path.join(save_root_path[p], all_class[i])
                print(f"DEBUG: Class save path: {class_save_path}")
                if not os.path.exists(class_save_path):
                    os.makedirs(class_save_path)
                    print(f"DEBUG: Created directory: {class_save_path}")

                print(f"DEBUG: Saving {len(cur_class_all_image)} masks")
                for j in range(len(cur_class_all_image)):
                    exact_save_path = save_list[i][j]
                    if j < 3:  # Only print for first few
                        print(f"DEBUG: Saving mask {j} to: {exact_save_path}")
                    result = cur_class_mask[j, :, :].cpu().numpy()  # Add .cpu() for tensor conversion
                    result = Image.fromarray((result * 255).astype(np.uint8))
                    w, h = Image.open(image_list[i][j]).size
                    result = result.resize((w, h), Image.BILINEAR)
                    result.convert('L').save(exact_save_path)

            print('DEBUG: UFO processing complete!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./weights/video_best.pth',help="restore checkpoint")
    parser.add_argument('--data_path',default='./datasets/elvis/', help="dataset for evaluation")
    parser.add_argument('--output_dir', default='./VSOD_results/wo_optical_flow/elvis/', help='directory for result')
    parser.add_argument('--task', default='VSOD', help='task')
    parser.add_argument('--gpu_id', default='cuda:0', help='id of gpu')
    args = parser.parse_args()
    
    gpu_id = args.gpu_id
    device = torch.device(gpu_id)
    model_path = args.model
    val_datapath = [args.data_path]
    save_root_path = [args.output_dir]
    
    debug_test(gpu_id, model_path, val_datapath, save_root_path, 5, 224, 'image')
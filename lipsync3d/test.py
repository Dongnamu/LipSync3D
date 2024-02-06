import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Lipsync3DMeshDataset
from model import Lipsync3DMesh, Lipsync3DModel
from loss import L2Loss
import time
from utils import mesh_tensor_to_landmarkdict, draw_mesh_images
import os
from tqdm import tqdm
import cv2
import shutil

# -------------------------------------------- Added by Jonghoon Shin
import torch.nn as nn
from IQA_pytorch import SSIM
from torchvision.utils import save_image
# -------------------------------------------- Added by Jonghoon Shin

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device
    calculate_test_loss = (opt.src_dir == opt.tgt_dir)
    dataset = Lipsync3DMeshDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    model = Lipsync3DMesh().to(device)
    # model = nn.DataParallel(model)
    criterionGeo = L2Loss()
    criterionTex = SSIM(channels=3)

    if opt.load_only_mesh:
        state_dict = torch.load(os.path.join(opt.tgt_dir, opt.model_name))
        audioEncoder_state = {}
        geometryDecoder_state = {}

        for key, value in state_dict.items():
            if 'AudioEncoder' in key:
                audioEncoder_state[key.replace('AudioEncoder.', '')] = value
            if 'GeometryDecoder' in key:
                geometryDecoder_state[key.replace('GeometryDecoder.', '')] = value
        model.AudioEncoder.load_state_dict(audioEncoder_state)
        model.GeometryDecoder.load_state_dict(geometryDecoder_state)
        
    # state_dict = torch.load(os.path.join(opt.tgt_dir, 'Lipsync3dnet.pth'))
    # audioEncoder_state = {}
    # geometryDecoder_state = {}

    # for key, value in state_dict.items():
    #     if 'AudioEncoder' in key:
    #         audioEncoder_state[key.replace('AudioEncoder.', '')] = value
    #     if 'GeometryDecoder' in key:
    #         geometryDecoder_state[key.replace('GeometryDecoder.', '')] = value
    # model.AudioEncoder.load_state_dict(audioEncoder_state)
    # model.GeometryDecoder.load_state_dict(geometryDecoder_state)

    # state_dict = torch.load(os.path.join(opt.tgt_dir, 'Lipsync3dnet_ADAM.pth'))
    # textureEncoder_state = {}
    # textureDecoder_state = {}

    # for key, value in state_dict.items():
    #     if 'TextureEncoder' in key:
    #         textureEncoder_state[key.replace('module.TextureEncoder.', '')] = value
    #     if 'TextureDecoder' in key:
    #         textureDecoder_state[key.replace('module.TextureDecoder.', '')] = value
    # model.TextureEncoder.load_state_dict(textureEncoder_state)
    # model.TextureDecoder.load_state_dict(textureDecoder_state)

    def emptyFolder(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    ckpt = torch.load(os.path.join(opt.tgt_dir, opt.model_name), map_location=device)
    model.load_state_dict(ckpt)
    
    emptyFolder(os.path.join(opt.src_dir, 'reenact_mesh'))
    emptyFolder(os.path.join(opt.src_dir, 'reenact_mesh_image'))
    emptyFolder(os.path.join(opt.src_dir, 'reenact_texture'))
    emptyFolder(os.path.join(opt.src_dir, 'predicted_normalised_mesh'))
    
    avg_loss = 0

    # previous_texture = torch.zeros((1, 3, 140, 280)).to(device)
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(tqdm(test_dataloader)):
            audio_feature = data['audio_feature'].to(device)
            reference_mesh = data['reference_mesh'].to(device)
            normalized_mesh = data['normalized_mesh'].to(device)
            filename = data['filename'][0]
            R = data['R'][0].to(device)
            RT = R.transpose(0, 1)
            t = data['t'][0].to(device)
            c = data['c'][0].to(device)
            # reference_mouth = data['reference_mouth'].to(device)

            # if i == 0:
            #     previous_texture = reference_mouth

            # geometry_diff, texture_diff = model(audio_feature, previous_texture)
            geometry_diff = model(audio_feature)
            geometry_diff = geometry_diff.reshape(-1, 478, 3)
            geometry = reference_mesh + geometry_diff
            # predicted_mouth = reference_mouth + texture_diff

            if calculate_test_loss and (i > int(len(test_dataloader) * opt.train_rate)):
                geoLoss = criterionGeo(geometry, normalized_mesh)
                avg_loss += geoLoss.detach() / int(len(test_dataloader) * (1 - opt.train_rate))

            geometry = geometry[0].transpose(0, 1)
            normlaised_geometry = geometry.clone().detach()
            normalised_landmark_dict = mesh_tensor_to_landmarkdict(normlaised_geometry)
            
            geometry = (torch.matmul(RT, (geometry - t)) / c).transpose(0, 1).cpu().detach()
            landmark_dict = mesh_tensor_to_landmarkdict(geometry)

            # previous_texture = predicted_mouth
            
            # save_image(predicted_mouth[0], os.path.join(opt.src_dir, 'reenact_texture',filename.split('.')[0]+'.jpg'))
            torch.save(normalised_landmark_dict, os.path.join(opt.src_dir,'predicted_normalised_mesh',filename))
            torch.save(landmark_dict, os.path.join(opt.src_dir, 'reenact_mesh', filename))
    
    if calculate_test_loss:
        print('Average Test loss : ', avg_loss)

    print('Start drawing reenact mesh')
    image = cv2.imread(os.path.join(opt.tgt_dir, 'reference_frame.png'))
    image_rows, image_cols, _ = image.shape
    draw_mesh_images(os.path.join(opt.src_dir, 'reenact_mesh'), os.path.join(opt.src_dir, 'reenact_mesh_image'), image_rows, image_cols)
            


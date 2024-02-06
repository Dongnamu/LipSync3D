from torch import optim
from torch.optim import optimizer

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Lipsync3DDataset
from model import Lipsync3DModel
from loss import L2Loss, L1Loss
from audiodvp_utils.visualizer import Visualizer
import time
import os
from adabound.adabound import AdaBound

# -------------------------------------------- Added by Jonghoon Shin
import torch.nn as nn
from IQA_pytorch import SSIM
# -------------------------------------------- Added by Jonghoon Shin


if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device

    dataset = Lipsync3DDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    visualizer = Visualizer(opt)
    model = Lipsync3DModel(use_auto_regressive=opt.autoregressive).to(device)

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

    if opt.load_model:
        if os.path.exists(os.path.join(opt.tgt_dir, opt.model_name)):
            model.load_state_dict(torch.load(os.path.join(opt.tgt_dir, opt.model_name), map_location=device))

    if opt.changeAggressive or opt.changeSize:
        if os.path.exists(os.path.join(opt.tgt_dir, opt.model_name)):
            state_dict = torch.load(os.path.join(opt.tgt_dir, opt.model_name))
            audioEncoder_state = {}
            geometryDecoder_state = {}
            textureEncoder_state = {}
            textureDecoder_state = {}
            
            for key, value in state_dict.items():
                if 'AudioEncoder' in key:
                    audioEncoder_state[key.replace('AudioEncoder.', '')] = value
                if 'GeometryDecoder' in key:
                    geometryDecoder_state[key.replace('GeometryDecoder.', '')] = value

                if opt.changeSize:
                    if 'TextureEncoder' in key:
                        id = int(key.split('.')[1])
                        if id != 11:
                            textureEncoder_state[key.replace('TextureEncoder.', '')] = value
                        else:
                            if 'weight' in key:
                                textureEncoder_state[key.replace['TextureEncoder.', '']] = torch.randn((value.shape[0], 2))
                            else:
                                textureEncoder_state[key.replace['TextureEncoder.', '']] = torch.randn(value.shape[0])
                if 'TextureDecoder' in key:
                    id = int(key.split('.')[1])
                    if id != 0:
                        textureDecoder_state[key.replace('TextureDecoder.', '')] = value
                    else:
                        if 'weight' in key:
                            textureDecoder_state[key.replace('TextureDecoder.', '')] = torch.randn((value.shape[0], 32))
                        else:
                            textureDecoder_state[key.replace('TextureDecoder.', '')] = torch.randn(value.shape[0])


            model.AudioEncoder.load_state_dict(audioEncoder_state)
            model.GeometryDecoder.load_state_dict(geometryDecoder_state)
            if opt.changeSize:
                model.TextureEncoder.load_state_dict(textureEncoder_state)
            model.TextureDecoder.load_state_dict(textureDecoder_state)

            for name, param in model.named_parameters():
                if 'AudioEncoder' in name or 'GeometryDecoder' in name:
                    param.requires_grad = False


    if opt.freeze_mesh:
        for name, param in model.named_parameters():
            if 'AudioEncoder' in name or 'GeometryDecoder' in name:
                param.requires_grad = False


    if opt.freeze_mesh or opt.changeAggressive:
        # optimizer = AdaBound(list(model.TextureDecoder.parameters())+list(model.TextureEncoder.parameters()), lr=opt.lr, final_lr=opt.lr * 100, weight_decay=5e-4, amsbound=True)
        optimizer = optim.Adam(list(model.TextureDecoder.parameters())+list(model.TextureEncoder.parameters()), lr=opt.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    os.makedirs(os.path.join(opt.tgt_dir, 'checkpoint'), exist_ok=True)

    model = nn.DataParallel(model)


    total_iters = 0
    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()
        epoch_iter = 0

        # if epoch == 151 and opt.freeze_mesh:
        #     model = Lipsync3DModel(use_auto_regressive=opt.autoregressive).to(device)
        #     model.load_state_dict(torch.load(os.path.join(opt.tgt_dir, 'checkpoint', 'checkpoint_150.pth')))
            
        #     for name, param in model.named_parameters():
        #         if 'AudioEncoder' in name or 'GeometryDecoder' in name:
        #             param.requires_grad = False
        
        #     optimizer = optim.Adam(list(model.TextureDecoder.parameters())+list(model.TextureEncoder.parameters()), lr=opt.lr)

        #     model = nn.DataParallel(model)

        for i, data in enumerate(train_dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            optimizer.zero_grad()

            audio_feature = data['audio_feature'].to(device)
            reference_mesh = data['reference_mesh'].to(device)
            normalized_mesh = data['normalized_mesh'].to(device)
            previous_mouth = data['previous_mouth'].to(device)
            gt_mouth = data['texture_mouth'].to(device)
            reference_mouth = data['reference_mouth'].to(device)

            geometry_diff, texture_diff = model(audio_feature, texture_pred=previous_mouth)
            # geometry_diff = model(audio_feature)
            geometry_diff = geometry_diff.reshape(-1, 478, 3)
            geometry = reference_mesh + geometry_diff
            geoLoss = criterionGeo(geometry, normalized_mesh)

            # print(torch.max(gt_mouth), torch.min(gt_mouth), torch.max(texture_diff), torch.min(texture_diff))
            # -------------------------------------------- Added by Jonghoon Shin
            predicted_mouth = reference_mouth + texture_diff
            texLoss = criterionTex(predicted_mouth * 255., gt_mouth * 255., as_loss=True)
            # -------------------------------------------- Added by Jonghoon Shin


            # -------------------------------------------- Modified by Jonghoon Shin
            loss = texLoss + opt.lambda_geo * geoLoss
            # -------------------------------------------- Modified by Jonghoon Shin
            
            loss.backward()
            optimizer.step()

            if total_iters % opt.print_freq == 0:    # print training losses
                # -------------------------------------------- Modified by Jonghoon Shin
                losses = {'geoLoss': geoLoss, 'texLoss' : texLoss}
                # -------------------------------------------- Modified by Jonghoon Shin
                visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)
                visualizer.plot_current_losses(total_iters, losses)
                # -------------------------------------------- Added by Jonghoon Shin
                visualizer.plot_current_texture(total_iters, predicted_mouth, gt_mouth)
                # -------------------------------------------- Added by Jonghoon Shin


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

        if epoch % opt.checkpoint_interval == 0 and epoch != 0:
            torch.save(model.module.state_dict(), os.path.join(opt.tgt_dir, 'checkpoint', 'checkpoint_{}.pth'.format(epoch)))
            print("Checkpoint saved")

    torch.save(model.module.state_dict(), os.path.join(opt.tgt_dir, 'Lipsync3dnet.pth'))
            


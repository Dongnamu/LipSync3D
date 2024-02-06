from mmap import ACCESS_DEFAULT
from torch import optim

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Lipsync3DTextureDataset
from model import Lipsync3DMesh, Lipsync3DTexture, SyncNet_color
from audiodvp_utils.visualizer import Visualizer
import time
import os
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from IQA_pytorch import SSIM

opt = Options().parse_args()
device = opt.device

logloss = nn.BCELoss()

syncnet = SyncNet_color().to(device)

for p in syncnet.parameters():
    p.requires_grad = False

syncnet_checkpoint = torch.load(opt.syncnet_checkpoint_path)

s = syncnet_checkpoint['state_dict']
new_s = {}

for k, v in s.items():
    new_s[k.replace('module.', '')] = v

syncnet.load_state_dict(new_s)
# syncnet = nn.DataParallel(syncnet)

def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def get_sync_loss(mel, g):
    g = torch.reshape(g, (g.size(0), g.size(1)*g.size(2),g.size(3),g.size(4)))
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

if __name__ == '__main__':
    device = opt.device
    y_split = 48

    dataset = Lipsync3DTextureDataset(opt)
    train_dataloder = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=opt.num_workers,
        drop_last=True
    )

    visualizer = Visualizer(opt)
    mesh_model = Lipsync3DMesh().to(device)
    state_dict = torch.load(os.path.join(opt.mesh_model_path))
    audioEncoder_state = {}
    geometryDecoder_state = {}

    criterionTex = SSIM(channels=3)

    for key, value in state_dict.items():
        if 'AudioEncoder' in key:
            audioEncoder_state[key.replace('AudioEncoder.', '')] = value
        if 'GeometryDecoder' in key:
            geometryDecoder_state[key.replace('GeometryDecoder.', '')] = value
    
    mesh_model.AudioEncoder.load_state_dict(audioEncoder_state)
    mesh_model.GeometryDecoder.load_state_dict(geometryDecoder_state)
    
    for p in mesh_model.parameters():
        p.requires_grad = False
    
    texture_model = Lipsync3DTexture().to(device)

    if opt.load_model:
        if os.path.exists(opt.model_name):
            texture_model.load_state_dict(torch.load(opt.model_name))
    
    optimizer = optim.Adam(texture_model.parameters(), lr=opt.lr)

    # mesh_model = nn.DataParallel(mesh_model)
    # texture_model = nn.DataParallel(texture_model)

    total_iters = 0
    resize = transforms.Resize((48, 96))
    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()
        epoch_iters = 0

        for i, data in enumerate(train_dataloder):
            total_iters += opt.batch_size
            epoch_iters += opt.batch_size

            optimizer.zero_grad()
            audio_features = data['audio_feature']
            reference_mouths = data['reference_mouth']
            previous_mouths = data['previous_mouth']
            texture_mouths = data['texture_mouth']
            mel = data['mel'].to(device)

            predicted_bathces = []
            ori_size_batches = []
            texLoss = []
            for i in range(audio_features.size(0)):
                reference_mouth = reference_mouths[i].to(device)
                predicted = []
                ori_size = []
                for j in range(5):
                    audio_feature = audio_features[i][j].to(device)
                    previous_mouth = previous_mouths[i][j].to(device)
                    texture_mouth = texture_mouths[i][j].to(device)

                    prevoius_mouth_input = torch.reshape(previous_mouth, (1, *previous_mouth.size()))
                    audio_feature_input = torch.reshape(audio_feature, (1, *audio_feature.size()))

                    audio_latent = mesh_model(audio_feature_input, latentMode = True)
                    texture_diff = texture_model(audio_latent, prevoius_mouth_input)
                    texture_diff = torch.squeeze(texture_diff, 0)
                    predicted_mouth = reference_mouth + texture_diff
                    ori_size.append((predicted_mouth.detach().cpu().numpy().transpose(1,2,0) * 255.).astype(np.uint8))
                    texLoss.append(criterionTex(torch.reshape(predicted_mouth, (1, *predicted_mouth.size())) * 255., torch.reshape(texture_mouth, (1, *texture_mouth.size())) * 255., as_loss=True))
                    predicted_mouth = resize(predicted_mouth)
                    predicted.append(predicted_mouth)
                    
                ori_size_batches.append(ori_size)

                predicted = torch.stack(predicted, 0)
                predicted_bathces.append(predicted)

            ori_size_batches = np.array(ori_size_batches)

            predicted_bathces = torch.stack(predicted_bathces)
            sync_loss = get_sync_loss(mel, predicted_bathces)
            texture_loss = sum(texLoss) / len(texLoss)
            loss = texture_loss + 0.2 * sync_loss
            loss.backward()
            optimizer.step()

            if total_iters % opt.print_freq == 0:
                losses = {'texture loss': texture_loss, 'sync loss':sync_loss}
                visualizer.print_current_losses(epoch, epoch_iters, losses, 0, 0)
                
                folder = os.path.join(opt.tgt_dir, 'texture_checkpoint', 'samples_{}'.format(total_iters))
                os.makedirs(folder, exist_ok=True)

                for batch_idx, c in enumerate(ori_size_batches):
                    for t in range(len(c)):
                        cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), cv2.cvtColor(c[t], cv2.COLOR_BGR2RGB))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

        if epoch % opt.checkpoint_interval == 0 and epoch != 0:
            torch.save(texture_model.module.state_dict(), os.path.join(opt.tgt_dir, 'texture_checkpoint', 'texture_checkpoint_{}.pth'.format(epoch)))
            print("Checkpoint saved")

    torch.save(texture_model.module.state_dict(), os.path.join(opt.tgt_dir,'texture_model.pth'))

    
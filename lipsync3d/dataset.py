import sys

from hparams import hparams

sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import os
import torch
import numpy as np
import librosa
from utils import landmarkdict_to_normalized_mesh_tensor, landmarkdict_to_mesh_tensor
from audiodvp_utils import util
from torch.utils.data import Dataset
from natsort import natsorted
import torchvision.transforms as transforms
import cv2
from PIL import Image
import audio as audioLibrary
import random

class Lipsync3DMeshDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = opt.tgt_dir

        self.stabilized_mesh = [os.path.join(self.tgt_dir, 'stabilized_norm_mesh', x) for x in natsorted(os.listdir(os.path.join(self.tgt_dir, 'stabilized_norm_mesh')))]


        stft_path = os.path.join(self.src_dir, 'audio/audio_stft.pt')
        if not os.path.exists(stft_path):
            audio = librosa.load(os.path.join(self.src_dir, 'audio/audio.wav'),16000)[0]
            audio_stft = librosa.stft(audio, n_fft=510, hop_length=160, win_length=480)
            self.audio_stft = torch.from_numpy(np.stack((audio_stft.real, audio_stft.imag)))
            torch.save(self.audio_stft, os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        else:
            self.audio_stft = torch.load(os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        
        self.mesh_dict_list = util.load_coef(os.path.join(self.tgt_dir, 'mesh_dict'))
        self.filenames = util.get_file_list(os.path.join(self.tgt_dir, 'mesh_dict'))
        reference_mesh_dict = torch.load(os.path.join(self.tgt_dir, 'reference_mesh.pt'))

        self.reference_mesh = landmarkdict_to_normalized_mesh_tensor(reference_mesh_dict)
        
        if opt.isTrain:
            minlen = min(len(self.mesh_dict_list), self.audio_stft.shape[2] // 4)
            train_idx = int(minlen * self.opt.train_rate)
            self.mesh_dict_list = self.mesh_dict_list[:train_idx]
            self.filenames = self.filenames[:train_idx]

        print('Training set size: ', len(self.filenames))

    def __len__(self):
        return min(self.audio_stft.shape[2] // 4, len(self.filenames))

    def __getitem__(self, index):

        audio_idx = index * 4

        audio_feature_list = []
        for i in range(audio_idx - 12, audio_idx + 12):
            if i < 0:
                audio_feature_list.append(self.audio_stft[:, :, 0])
            elif i >= self.audio_stft.shape[2]:
                audio_feature_list.append(self.audio_stft[:, :, -1])
            else:
                audio_feature_list.append(self.audio_stft[:, :, i])

        audio_feature = torch.stack(audio_feature_list, 2)

        filename = os.path.basename(self.filenames[index])

        if not self.opt.isTrain:
            landmark_dict = self.mesh_dict_list[index]
            normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            # stabilized_mesh = torch.tensor(torch.load(self.stabilized_mesh[index]))

            R = torch.from_numpy(landmark_dict['R']).float()
            t = torch.from_numpy(landmark_dict['t']).float()
            c = float(landmark_dict['c'])

            return {'audio_feature': audio_feature, 'filename': filename, 
                    'reference_mesh': self.reference_mesh, 'normalized_mesh': normalized_mesh,
                    'R': R, 't': t, 'c': c}

        else:
            landmark_dict = self.mesh_dict_list[index]
            normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            # stabilized_mesh = torch.tensor(torch.load(self.stabilized_mesh[index]))
            return {
                'audio_feature': audio_feature, 'filename': filename,
                'reference_mesh' : self.reference_mesh, 'normalized_mesh': normalized_mesh
            }

class Lipsync3DTextureDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = opt.tgt_dir
        self.y_size = 140

        stft_path = os.path.join(self.src_dir, 'audio/audio_stft.pt')
        if not os.path.exists(stft_path):
            audio = librosa.load(os.path.join(self.src_dir, 'audio/audio.wav'),16000)[0]
            audio_stft = librosa.stft(audio, n_fft=510, hop_length=160, win_length=480)
            self.audio_stft = torch.from_numpy(np.stack((audio_stft.real, audio_stft.imag)))
            torch.save(self.audio_stft, os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        else:
            self.audio_stft = torch.load(os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        

        self.filenames = util.get_file_list(os.path.join(self.tgt_dir, 'mesh_dict'))

        wavPath = os.path.join(self.src_dir, 'audio/audio.wav')
        wav = audioLibrary.load_wav(wavPath, hparams.sample_rate)
        self.orig_mel = audioLibrary.melspectrogram(wav).T

        # -------------------------------------------- Added by Jonghoon Shin
        self.transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.texture_image = [os.path.join(self.tgt_dir, 'texture_images', x) for x in natsorted(os.listdir(os.path.join(self.tgt_dir, 'texture_images')))]
        self.mouthRegion = [62, 78, 191, 95, 80, 88, 81, 178, 82, 87, 13, 14, 312, 317, 311, 402, 310, 318, 415, 324, 308, 293]
        reference_texture_image = np.array(Image.open(os.path.join(self.tgt_dir, 'reference_texture.jpg')).convert('RGB'))
        self.reference_mouth_image = self.transform(reference_texture_image[self.y_size:, :, :])
        # self.reference_mouth_image = self.transform(reference_texture_image[y-64:y+64, x-64:x+64, :])

        # -------------------------------------------- Added by Jonghoon Shin
        
        if opt.isTrain:
            minlen = min(len(self.filenames), self.audio_stft.shape[2] // 4)
            train_idx = int(minlen * self.opt.train_rate)
            self.filenames = self.filenames[:train_idx]
            # -------------------------------------------- Added by Jonghoon Shin
            self.texture_image = self.texture_image[:train_idx]
            # -------------------------------------------- Added by Jonghoon Shin

        print('training set size: ', len(self.filenames))


    def __len__(self):
        return min(self.audio_stft.shape[2] // 4, len(self.filenames))

    def crop_audio_window(self, spec, start_frame):
        start_frame_num = start_frame + 1
        start_idx = int(80. * (start_frame_num / float(25)))
        end_idx = start_idx + 16
        return spec[start_idx : end_idx, :]

    def __getitem__(self, index):
        
        filename = os.path.basename(self.filenames[index])

        if not self.opt.isTrain:
            audio_idx = index * 4

            audio_feature_list = []
            for i in range(audio_idx - 12, audio_idx + 12):
                if i < 0:
                    audio_feature_list.append(self.audio_stft[:, :, 0])
                elif i >= self.audio_stft.shape[2]:
                    audio_feature_list.append(self.audio_stft[:, :, -1])
                else:
                    audio_feature_list.append(self.audio_stft[:, :, i])

            audio_feature = torch.stack(audio_feature_list, 2)

            filename = os.path.basename(self.filenames[index])

            # -------------------------------------------- Modified by Jonghoon Shin
            # normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            # -------------------------------------------- Modified by Jonghoon Shin

            return {'audio_feature': audio_feature, 'filename': filename, 
                    # -------------------------------------------- Added by Jonghoon Shin
                    'reference_mouth':self.reference_mouth_image,
                    # -------------------------------------------- Added by Jonghoon Shin
                    }
        else:

                if index + 5 >= len(self.filenames):
                    index = random.randint(0, len(self.filenames) - 6)

                audio_idx = []

                for i in range(index, index+5):
                    audio_idx.append(index * 4)

                audio_feature_list = []

                for j in range(len(audio_idx)):
                    audio_feature = []
                    for i in range(audio_idx[j] - 12, audio_idx[j] + 12):
                        if i < 0:
                            audio_feature.append(self.audio_stft[:, :, 0])
                        elif i >= self.audio_stft.shape[2]:
                            audio_feature.append(self.audio_stft[:, :, -1])
                        else:
                            audio_feature.append(self.audio_stft[:, :, i])
                    audio_feature = torch.stack(audio_feature, 2)
                    audio_feature_list.append(audio_feature)

                audio_feature_list = torch.stack(audio_feature_list, 0)

                mel = self.crop_audio_window(self.orig_mel.copy(), index)
                mel = torch.FloatTensor(mel.T).unsqueeze(0)
                previous_mouth = []
                texture_mouth = []

                for i in range(index, index+5):
                # -------------------------------------------- Added by Jonghoon Shin
                    if i == 0:
                        previous_mouth.append(torch.zeros(3, 140, 280))
                    else:
                        single_previous_texture = cv2.imread(self.texture_image[i - 1])
                        previous_mouth.append(self.transform(single_previous_texture[self.y_size:, :, :]))

                    texture = np.array(Image.open(self.texture_image[index]).convert('RGB'))
                    texture_mouth.append(self.transform(texture[self.y_size:, :, :]))
                # -------------------------------------------- Added by Jonghoon Shin

                previous_mouth = torch.stack(previous_mouth, 0)
                texture_mouth = torch.stack(texture_mouth, 0)
                return {
                    'audio_feature': audio_feature_list,
                    # -------------------------------------------- Added by Jonghoon Shin
                    'reference_mouth':self.reference_mouth_image,
                    'previous_mouth':previous_mouth,
                    'texture_mouth':texture_mouth,
                    'mel':mel
                    # -------------------------------------------- Added by Jonghoon Shin
                }



class Lipsync3DDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = opt.tgt_dir
        self.y_size = 140

        stft_path = os.path.join(self.src_dir, 'audio/audio_stft.pt')
        if not os.path.exists(stft_path):
            audio = librosa.load(os.path.join(self.src_dir, 'audio/audio.wav'),16000)[0]
            audio_stft = librosa.stft(audio, n_fft=510, hop_length=160, win_length=480)
            self.audio_stft = torch.from_numpy(np.stack((audio_stft.real, audio_stft.imag)))
            torch.save(self.audio_stft, os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        else:
            self.audio_stft = torch.load(os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        
        self.mesh_dict_list = util.load_coef(os.path.join(self.tgt_dir, 'mesh_dict'))
        self.filenames = util.get_file_list(os.path.join(self.tgt_dir, 'mesh_dict'))
        reference_mesh_dict = torch.load(os.path.join(self.tgt_dir, 'reference_mesh.pt'))

        self.reference_mesh = landmarkdict_to_normalized_mesh_tensor(reference_mesh_dict)
        
        # -------------------------------------------- Added by Jonghoon Shin
        self.transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.texture_image = [os.path.join(self.tgt_dir, 'texture_images', x) for x in natsorted(os.listdir(os.path.join(self.tgt_dir, 'texture_images')))]
        self.texture_mesh = [os.path.join(self.tgt_dir, 'texture_mesh', x) for x in natsorted(os.listdir(os.path.join(self.tgt_dir, 'texture_mesh')))]
        self.stabilized_mesh = [os.path.join(self.tgt_dir, 'stabilized_norm_mesh', x) for x in natsorted(os.listdir(os.path.join(self.tgt_dir, 'stabilized_norm_mesh')))]
        self.mouthRegion = [62, 78, 191, 95, 80, 88, 81, 178, 82, 87, 13, 14, 312, 317, 311, 402, 310, 318, 415, 324, 308, 293]
        reference_texture_mesh = torch.load(os.path.join(self.tgt_dir, 'texture_reference_mesh.pt'))
        reference_texture_image = np.array(Image.open(os.path.join(self.tgt_dir, 'reference_texture.jpg')).convert('RGB'))
        x = np.average(reference_texture_mesh[self.mouthRegion, 0]).astype(int)
        y = np.average(reference_texture_mesh[self.mouthRegion, 1]).astype(int)
        self.reference_mouth_image = self.transform(reference_texture_image[self.y_size:, :, :])
        # self.reference_mouth_image = self.transform(reference_texture_image[y-64:y+64, x-64:x+64, :])

        # -------------------------------------------- Added by Jonghoon Shin
        
        if opt.isTrain:
            minlen = min(len(self.mesh_dict_list), self.audio_stft.shape[2] // 4)
            train_idx = int(minlen * self.opt.train_rate)
            self.mesh_dict_list = self.mesh_dict_list[:train_idx]
            self.filenames = self.filenames[:train_idx]
            # -------------------------------------------- Added by Jonghoon Shin
            self.texture_image = self.texture_image[:train_idx]
            self.texture_mesh = self.texture_mesh[:train_idx]
            self.stabilized_mesh = self.stabilized_mesh[:train_idx]
            # -------------------------------------------- Added by Jonghoon Shin

        print('training set size: ', len(self.filenames))


    def __len__(self):
        return min(self.audio_stft.shape[2] // 4, len(self.filenames))

    def __getitem__(self, index):

        audio_idx = index * 4

        audio_feature_list = []
        for i in range(audio_idx - 12, audio_idx + 12):
            if i < 0:
                audio_feature_list.append(self.audio_stft[:, :, 0])
            elif i >= self.audio_stft.shape[2]:
                audio_feature_list.append(self.audio_stft[:, :, -1])
            else:
                audio_feature_list.append(self.audio_stft[:, :, i])

        audio_feature = torch.stack(audio_feature_list, 2)

        filename = os.path.basename(self.filenames[index])

        if not self.opt.isTrain:
            landmark_dict = self.mesh_dict_list[index]

            # -------------------------------------------- Modified by Jonghoon Shin
            # normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            stabilized_mesh = torch.tensor(torch.load(self.stabilized_mesh[index]))
            # -------------------------------------------- Modified by Jonghoon Shin
            
            R = torch.from_numpy(landmark_dict['R']).float()
            t = torch.from_numpy(landmark_dict['t']).float()
            c = float(landmark_dict['c'])

            return {'audio_feature': audio_feature, 'filename': filename, 
                    'reference_mesh': self.reference_mesh, 'normalized_mesh': stabilized_mesh,
                    'R': R, 't': t, 'c': c,
                    # -------------------------------------------- Added by Jonghoon Shin
                    'reference_mouth':self.reference_mouth_image,
                    # -------------------------------------------- Added by Jonghoon Shin
                    }
        else:
            # landmark_dict = self.mesh_dict_list[index]
            # -------------------------------------------- Modified by Jonghoon Shin
            # normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            stabilized_mesh = torch.tensor(torch.load(self.stabilized_mesh[index]))
            # -------------------------------------------- Modified by Jonghoon Shin

            # -------------------------------------------- Added by Jonghoon Shin
            texture = np.array(Image.open(self.texture_image[index]).convert('RGB'))
            mesh = torch.load(self.texture_mesh[index])
            x = np.average(mesh[self.mouthRegion, 0]).astype(int)
            y = np.average(mesh[self.mouthRegion, 1]).astype(int)
            texture_mouth = self.transform(texture[self.y_size:, :, :])
            # texture_mouth = self.transform(texture[y-64:y+64, x-64:x+64, :])

            if index == 0:
                previous_texture = torch.zeros(3, 140, 280)
            else:
                previous_texture = cv2.imread(self.texture_image[index - 1])
                mesh = torch.load(self.texture_mesh[index - 1])
                x = np.average(mesh[self.mouthRegion, 0]).astype(int)
                y = np.average(mesh[self.mouthRegion, 1]).astype(int)
                previous_texture = self.transform(previous_texture[self.y_size:, :, :])
                # previous_texture = self.transform(previous_texture[y-64:y+64, x-64:x+64, :])

            # -------------------------------------------- Added by Jonghoon Shin

            return {
                'audio_feature': audio_feature, 'filename': filename,
                'reference_mesh' : self.reference_mesh, 'normalized_mesh': stabilized_mesh,
                # -------------------------------------------- Added by Jonghoon Shin
                'reference_mouth':self.reference_mouth_image,
                'texture_mouth':texture_mouth,
                'previous_mouth':previous_texture
                # -------------------------------------------- Added by Jonghoon Shin
            }
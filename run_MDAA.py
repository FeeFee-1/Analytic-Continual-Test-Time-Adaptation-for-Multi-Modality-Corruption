import argparse
import math
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import sys
import torch
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
from tqdm import tqdm
from utilities import accuracy, seed_everything
from TTA import READ, MDAA

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='vggsound', choices=['vggsound', 'ks50'], help='dataset name')
parser.add_argument("--model", type=str, default='cav-mae-ft', help="the model used")
parser.add_argument("--dataset_mean", type=float, default=-5.081, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, default=4.4849, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, default=1024, help="the input length in frames")
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--weight_list", type=str, help="weight path")
parser.add_argument("--gpu", type=str, default='0', help="gpu device number")
parser.add_argument("--testmode", type=str, default='multimodal', help="how to test the model")
parser.add_argument('--tta-method', type=str, required=True, choices=['READ', 'Tent', 'EATA', 'SAR', 'MDAA', 'CoTTA', 'MMTTA', 'None'], help='which TTA method to be used')
parser.add_argument('--corruption-modality', type=str, required=True, choices=['video', 'audio', 'cross', 'none'], help='which modality to be corrupted')
parser.add_argument('--severity-start', type=int, default=5, help='the start severity of the corruption')
parser.add_argument('--severity-end', type=int, default=5, help='the end severity of the corruption')
parser.add_argument('--MDAApretrained', type=bool, required=True)
parser.add_argument('--theta', type=float, default=0.001) 
parser.add_argument('--reverse', type=bool, default=True)
args = parser.parse_args()


if args.dataset == 'vggsound':
    args.n_class = 309
    args.alpha = 7
    args.json_root = 'json_csv_files/vgg'
    args.label_csv = 'json_csv_files/class_labels_indices_vgg.csv'
    args.pretrain_path = 'checkpoints/vgg_65.5.pth'
    args.weight_list = 'json_csv_files/weights_vgg.json'
    args.buffer_size = 5000
elif args.dataset == 'ks50':
    args.n_class = 50
    args.alpha = 2
    args.json_root = 'json_csv_files/ks50'
    args.label_csv = 'json_csv_files/class_labels_indices_ks50.csv'
    args.pretrain_path = 'checkpoints/cav_mae_ks50.pth'
    args.weight_list = 'json_csv_files/weights_ks50.json'
    args.buffer_size = 8000


if args.corruption_modality == 'video':
    corruption_list = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression'
    ]
elif args.corruption_modality == 'audio':
    corruption_list = [
    'gaussian_noise',
    'traffic',
    'crowd',
    'rain',
    'thunder',
    'wind'
    ]
elif args.corruption_modality == 'both':
    corruption_list = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6'
    ]
elif args.corruption_modality == 'cross':
    corruption_list = [
    '1','2','3','4','5','6','7','8','9','10',
    '11','12','13','14','15','16','17','18','19','20','21']
elif args.corruption_modality == 'none':
    corruption_list = ['clean']
    args.severity_start = args.severity_end = 0


if args.reverse:
    filename = args.dataset + '-'+ str(args.theta) + '-' + args.corruption_modality + '-' + args.tta_method + '-reverse' + '.csv'
    corruption_list = corruption_list[::-1]
else:
    filename = args.dataset + '-'+ str(args.theta) + '-' + args.corruption_modality + '-' + args.tta_method + '.csv'
import csv
os.makedirs('result' + '/' + args.tta_method, exist_ok=True)
with open('result' + '/' + args.tta_method + '/' + filename, mode='a+', encoding="ISO-8859-1", newline='') as file:
    wr = csv.writer(file)
    wr.writerow(corruption_list)
   
for itr in range(1, 6):
    final_result = []
    seed = int(str(itr)*3)
    seed_everything(seed=seed)
    print("### Seed= {}, Round {} ###".format(seed, itr))

    if args.tta_method == 'MDAA':
        va_model = MDAA.MDAA(args)
        if not args.MDAApretrained:
            base_data_val = os.path.join(args.json_root, 'clean', 'severity_0.json')
            im_res = 224
            val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                              'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}
            base_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(base_data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
            a_R, v_R, x_R = MDAA.cls_align(base_loader, va_model, args)
            checkpoint = {
                'va_model_state_dict': va_model.state_dict(),
                'a_R': a_R,
                'v_R': v_R,
                'x_R': x_R
            }
            torch.save(checkpoint, 'checkpoints/'+args.dataset+'_model.pth')
        else:
            checkpoint = torch.load('checkpoints/'+args.dataset+'_model.pth')
            va_model.load_state_dict(checkpoint['va_model_state_dict'], strict=False)
            a_R = checkpoint['a_R']
            v_R = checkpoint['v_R']
            x_R = checkpoint['x_R']

        for corruption in corruption_list:
            for severity in range(args.severity_start, args.severity_end+1):
                if corruption == 'clean':
                    data_val = os.path.join(args.json_root, corruption, 'severity_0.json')
                else:
                    data_val = os.path.join(args.json_root, args.corruption_modality, corruption, 'severity_{}.json'.format(severity))
                print('===> Now handling: ', data_val)

                im_res = 224
                val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                                  'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}
                
                tta_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
                                                         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
                data_bar = tqdm(tta_loader)
                batch_accs = []
                for i, (a_input, v_input, labels) in enumerate(data_bar):
                    outputs, a_R, v_R, x_R = MDAA.IL_align(a_input, v_input, va_model, a_R, v_R, x_R, args)
                    batch_acc = accuracy(outputs, labels, topk=(1,))[0].item()
                    batch_accs.append(batch_acc)
                    data_bar.set_description(f'Batch#{i}: ACC#{batch_acc:.2f}')
                
                epoch_acc = sum(batch_accs) / len(batch_accs)
                print('Epoch1: acc is {:.2f}'.format(epoch_acc))
                final_result.append(epoch_acc)

    elif args.tta_method in ['READ', 'None']:
        if args.model == 'cav-mae-ft':
            va_model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11)
        else:
            raise ValueError('model not supported')

        if args.pretrain_path != 'None':
            mdl_weight = torch.load(args.pretrain_path)
            va_model = torch.nn.DataParallel(va_model)
            va_model.load_state_dict(mdl_weight, strict=False)
            print('Loaded pre-trained weights from ', args.pretrain_path)

        va_model = torch.nn.DataParallel(va_model) if not isinstance(va_model, torch.nn.DataParallel) else va_model
        va_model.cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        adapt_flag = args.tta_method != 'None'
        
        va_model = READ.configure_model(va_model)
        trainables = [p for p in va_model.parameters() if p.requires_grad]
        print('Total params: {:.3f}M, Trainable: {:.3f}M'.format(
            sum(p.numel() for p in va_model.parameters()) / 1e6,
            sum(p.numel() for p in trainables) / 1e6))

        params, param_names = READ.collect_params(va_model)
        optimizer = torch.optim.Adam([{'params': params, 'lr': 1e-4}], weight_decay=0)
        read_model = READ.READ(va_model, optimizer, device, args)

        for corruption in corruption_list:
            for severity in range(args.severity_start, args.severity_end+1):
                if corruption == 'clean':
                    data_val = os.path.join(args.json_root, corruption, 'severity_0.json')
                else:
                    data_val = os.path.join(args.json_root, args.corruption_modality, corruption, 'severity_{}.json'.format(severity))
                print('===> Now handling: ', data_val)

                im_res = 224
                val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                                  'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

                tta_loader = torch.utils.data.DataLoader(
                    dataloader.AudiosetDataset(data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

                read_model.eval()
                with torch.no_grad():
                    data_bar = tqdm(tta_loader)
                    batch_accs = []
                    for i, (a_input, v_input, labels) in enumerate(data_bar):
                        a_input = a_input.to(device)
                        v_input = v_input.to(device)
                        outputs, loss = read_model((a_input, v_input), adapt_flag=adapt_flag)
                        batch_acc = accuracy(outputs[1], labels, topk=(1,))[0].item()
                        batch_accs.append(batch_acc)
                        data_bar.set_description(f'Batch#{i}: L0#{loss[0]:.4f}, L1#{loss[1]:.6f}, ACC#{batch_acc:.2f}')

                    epoch_acc = sum(batch_accs) / len(batch_accs)
                    print('Epoch1: acc is {:.2f}'.format(epoch_acc))
                    final_result.append(epoch_acc)
    
    if args.reverse:
        final_result = final_result[::-1]
    with open('result' + '/' + args.tta_method + '/' + filename, mode='a+', encoding="ISO-8859-1", newline='') as file:
        wr = csv.writer(file)
        wr.writerow(final_result)
    
print('written to {}.'.format('result/'+filename ))

import math
import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from .floating_region import FloatingRegionScore
from .spatial_purity import SpatialPurity
from networks import VAE
from torch.utils.data import DataLoader
from data_config import C0_TrainSet,LGE_TrainSet


def PixelSelection(net, tgt_epoch_loader,in_channels = 4,active_pixels=1,epoch = -1):
    net.eval()

    active_pixels = active_pixels
    calculate_purity = SpatialPurity(in_channels=in_channels, size=3).cuda()
    mask_radius = 3

    with torch.no_grad():
        for tgt_raw in tqdm(tgt_epoch_loader):

            tgt_data = tgt_raw
            tgt_input = tgt_data['img']
            path2mask = tgt_data['path_to_mask']
            origin_mask, origin_label = \
                tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda()

            tgt_size = tgt_input.shape[-2:]
            tgt_out1,_,_, _, _, _ ,_,_,_,_,_,_,_,_,_,_,_= net(tgt_input, gate=1)
            tgt_out,_,_, _, _, _ ,_,_,_,_,_,_,_,_,_,_,_= net(tgt_input, gate=0)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda()
                ground_truth = origin_label[i].cuda()
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                output,output1 = tgt_out[i:i + 1, :, :, :],tgt_out1[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                output = output.squeeze(dim=0)
                output1 = F.interpolate(output1, size=size, mode='bilinear', align_corners=True)
                output1 = output1.squeeze(dim=0)
                risk = torch.sum((output - output1)**2,dim=0)
                p = torch.softmax(output, dim=0)
                entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
                pseudo_label = torch.argmax(p, dim=0)
                one_hot = F.one_hot(pseudo_label, num_classes=in_channels).float()
                one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
                purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
                score = entropy * purity * risk

                score[active] = -float('inf')

                for pixel in range(active_pixels):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    # mask out
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    net.train()



def RegionSelection( net, tgt_epoch_loader,in_channels = 4):

    net.eval()

    floating_region_score = FloatingRegionScore(in_channels=in_channels, size=3).cuda()
    per_region_pixels = 3** 2
    active_radius = 2
    mask_radius = 2
    active_ratio = 0.001 /6

    with torch.no_grad():
        for i,tgt_raw in enumerate(tqdm(tgt_epoch_loader)):

            # tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            # tgt_input,_,_,_,origin_label,tgt_data = tgt_raw
            tgt_data = tgt_raw
            tgt_input = tgt_data['img']
            path2mask = tgt_data['path_to_mask']
            origin_mask, origin_label = \
                tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda()
            tgt_size = tgt_input.shape[-2:]
            tgt_out,_,_, _, _, _ ,_,_,_,_,_,_,_,_,_,_,_= net(tgt_input, gate=0)

            active_mask = origin_mask[0,:,:].cuda(non_blocking=True)
            ground_truth = origin_label[0,:,:].cuda(non_blocking=True)
            # size = (192, 192)
            size = (220,240)
            num_pixel_cur = size[0] * size[1]
            active = active_indicator[0]
            selected = selected_indicator[0]
            output = tgt_out[0:1, :, :, :]
            output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)

            # truemax, truearg0 = torch.max(output, 1, keepdim=False)
            score, purity, entropy = floating_region_score(output)
            # score[active] = -float('inf')

            active_regions = math.ceil(num_pixel_cur * active_ratio / per_region_pixels)
            # active_regions = 1
            for pixel in range(active_regions):
                values, indices_h = torch.max(score, dim=0)
                _, indices_w = torch.max(values, dim=0)
                w = indices_w.item()
                h = indices_h[w].item()

                active_start_w = w - active_radius if w - active_radius >= 0 else 0
                active_start_h = h - active_radius if h - active_radius >= 0 else 0
                active_end_w = w + active_radius + 1
                active_end_h = h + active_radius + 1

                mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
                mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
                mask_end_w = w + mask_radius + 1
                mask_end_h = h + mask_radius + 1

                score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                selected[active_start_h:active_end_h, active_start_w:active_end_w] = True

                active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                    ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]
                
            active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
            active_mask.save(path2mask[0])
            indicator = {
                    'active': active,
                    'selected': selected
                }
            torch.save(indicator, path2indicator[0])
        
    net.train()



            




import time
import torch
import torchvision
import datetime
import os
from model import render_single, Renderer
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, test_set, train_len, val_len, cfg, device):
        self.writer = SummaryWriter(os.path.join(cfg.output.dir, 'logs'))

        self.img = test_set['gt_imgs'].permute(2,0,1)
        self.gs = torch.cat(test_set['gs'], dim=-1).to(device).unsqueeze(0)
        self.pe = test_set['gs'][0].to(device).unsqueeze(0)
        self.Ks = test_set['Ks'].unsqueeze(0)
        self.w2cs = test_set['w2cs'].unsqueeze(0)

        self.renderer = Renderer(tuple(cfg.dataset.resolution), requires_grad=False)

        self.total_loss = 0
        self.p_loss = 0
        self.c_loss = 0

        self.ari = 0

        self.best = float('inf')

        self.train_len = train_len
        self.val_len = val_len

    def record_loss(self, total_loss:torch.Tensor, p_loss:torch.Tensor, c_loss:torch.Tensor):
        self.total_loss += total_loss.item()
        self.p_loss += p_loss.item()
        self.c_loss += c_loss.item()

    def record_ari(self, gt:torch.Tensor, pred:torch.Tensor):
        pass

    def plt_loss(self, epoch:int, start_time, mode='Train') -> bool:
        assert mode in ['Train', 'Val']
        if mode == 'Train':
            len = self.train_len
        else:
            len = self.val_len            

        self.total_loss /= len
        self.p_loss /= len
        self.c_loss /= len

        isBest = False
        if mode == 'Val' and self.best > self.total_loss:
            self.best = self.total_loss
            isBest = True

        self.writer.add_scalars(f'{mode} Loss', {
            'total': self.total_loss,
            'position': self.p_loss,
            'color': self.c_loss,
        }, epoch)

        print ("{} Loss: {}, Time: {}".format(mode, self.total_loss,
                    datetime.timedelta(seconds=time.time() - start_time)))
        
        self.total_loss = 0
        self.p_loss = 0
        self.c_loss = 0

        return isBest

    def plt_render(self, model, epoch:int):
        gs_out, gs_slot, gs_mask, _ = model(self.gs, self.pe)
        recon_combined, recon_slots = render_single(self.renderer, gs_out, gs_slot, gs_mask, self.Ks, self.w2cs)
        code_recon, code_slot = render_single(self.renderer, gs_out, gs_slot, gs_mask, self.Ks, self.w2cs, color_code=True)
        
        recon_combined = recon_combined.permute(2,0,1)
        code_recon = code_recon.permute(2,0,1)
        recon_slots = recon_slots.permute(0,3,1,2)
        code_slot = code_slot.permute(0,3,1,2)
        result = torchvision.utils.make_grid(torch.stack([self.img,recon_combined,code_recon],dim=0))
        self.writer.add_image('result', result, epoch)

        recon_slots = torchvision.utils.make_grid(torch.cat([recon_slots,code_slot],dim=0), nrow=5)
        self.writer.add_image('slots_recon', recon_slots, epoch)

        del recon_combined, recon_slots, gs_out, gs_slot, gs_mask
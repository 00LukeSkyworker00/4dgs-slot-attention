import time
import torch
import torchvision
import datetime
import os
from model import Renderer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import adjusted_rand_score

class Logger():
    def __init__(self, test_set, train_len, val_len, cfg, device):
        self.writer = SummaryWriter(os.path.join(cfg.output.dir, 'logs'))

        self.img = test_set['gt_render'][0]
        self.gs = test_set['gs'][:1].to(device)
        self.pe = test_set['pe'][:1].to(device)
        self.Ks = test_set['Ks'][0]
        self.w2cs = test_set['w2cs'][0]
        self.vid_ano = test_set['ano'][0]

        self.renderer = Renderer(tuple(cfg.dataset.resolution), self.img.shape[0], requires_grad=False)

        self.total_loss = 0
        self.p_loss = 0
        self.c_loss = 0

        self.best_loss = float('inf')
        self.best_ari = 0
        self.best_arifg = 0

        self.train_len = train_len
        self.val_len = val_len

        self.ari = 0
        self.ari_fg = 0
        self.ari_sample_len = 0

    def record_loss(self, total_loss:torch.Tensor, p_loss:torch.Tensor, c_loss:torch.Tensor):
        self.total_loss += total_loss.item()
        self.p_loss += p_loss.item()
        self.c_loss += c_loss.item()

    def record_ari(self, batch_gs, batch_mask, batch_Ks, batch_w2cs, batch_ano):
        for gs,mask,Ks,w2cs,vid_ano in zip(batch_gs,batch_mask,batch_Ks,batch_w2cs, batch_ano):
            vid_mask = self.renderer.make_vid_slot(gs, mask, Ks, w2cs, mask_as_color=True)
            vid_mask = vid_mask[...,0:1].argmax(dim=1).cpu().numpy()
            
            for ano,pred in zip(vid_ano,vid_mask):
                pred_fg = pred[ano != 0]
                anos_fg = ano[ano != 0]
                pred = pred.flatten()
                anos = ano.flatten()
                ari =adjusted_rand_score(anos,pred)
                arifg = adjusted_rand_score(anos_fg,pred_fg)
                self.ari += ari
                self.ari_fg += arifg
            self.ari_sample_len += len(vid_mask)

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
        if mode == 'Val' and self.best_loss > self.total_loss:
            self.best_loss = self.total_loss
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
    
    def plt_ari(self, epoch) -> tuple[bool,bool]:

        ari = self.ari / self.ari_sample_len
        arifg = self.ari_fg / self.ari_sample_len
        self.writer.add_scalars('Metrics', {
            'ARI': ari,
            'ARI-FG': arifg,
        }, epoch)

        isBestAri = False
        isBestArifg = False
        if self.best_ari < ari:
            self.best_ari = ari
            isBestAri = True
        if self.best_arifg < arifg:
            self.best_arifg = arifg
            isBestArifg = True
        
        self.ari = 0
        self.ari_fg = 0
        self.ari_sample_len = 0
        
        return isBestAri, isBestArifg

    def plt_render(self, model, epoch:int):
        with torch.no_grad():
            model.eval()
            batch_gs, batch_slot, batch_mask, _ = model(self.gs, self.pe,isInference=True)

            vid_mask = self.renderer.make_vid_slot(self.gs[0], batch_mask[0], self.Ks, self.w2cs, mask_as_color=True)  # [F, N_S, 128, 128, 3]
            cmap = self.renderer.create_cmap(vid_mask.shape[1], vid_mask.device)  # [N_S, 3]
            vid_mask = vid_mask.argmax(dim=1).squeeze(-1) # [F, 128, 128]
            ari_vis = cmap[vid_mask] # [F, 128, 128, 3]
            arifg_vis = ari_vis.cpu() * (self.vid_ano != 0) # [F, 128, 128, 3]

            recon_combined, recon_slots = self.renderer.make_vid(batch_gs[0], batch_slot[0], batch_mask[0], self.Ks, self.w2cs)
            code_combined, code_slot = self.renderer.make_vid(self.gs[0], self.gs[0], batch_mask[0], self.Ks, self.w2cs, color_code=True)

            result = torch.stack([self.img,recon_combined,code_combined,ari_vis,arifg_vis.to(ari_vis.device),torch.zeros_like(ari_vis)]).permute(0,1,4,2,3)
            result_slots = torch.cat([recon_slots,code_slot], dim=1).permute(1,0,4,2,3)
            
            self.writer.add_video('result', result, epoch,fps=10)
            self.writer.add_video('slots_recon', result_slots, epoch,fps=10)

            del recon_combined, recon_slots, code_combined, code_slot, batch_gs, batch_slot, batch_mask, result, result_slots
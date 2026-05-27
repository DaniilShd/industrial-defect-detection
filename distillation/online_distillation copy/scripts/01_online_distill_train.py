#!/usr/bin/env python3
"""
Online Detection Distillation — PREDICTION-LEVEL + LTDETR FIX
Использует правильный API для LTDETR из документации.
"""

import sys, logging, time, copy, json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_iou
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(Path(__file__).parent.parent/'logs'/'online_distill.log', mode='w')])
logger = logging.getLogger(__name__)

# ============================================================
# ДАТАСЕТ
# ============================================================
class DefectDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, lbl_dir, num_classes=4, size=(640,640)):
        self.img_dir = Path(img_dir); self.lbl_dir = Path(lbl_dir)
        self.num_classes = num_classes; self.size = size
        self.files = sorted([f for f in self.img_dir.glob("*") if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
        logger.info(f"  {img_dir}: {len(self.files)} images")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        ow, oh = img.size
        img = img.resize(self.size, Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img,dtype=np.float32)/255.0).permute(2,0,1)
        boxes, labels = [], []
        lbl_path = self.lbl_dir / f"{self.files[idx].stem}.txt"
        if lbl_path.exists():
            sx, sy = self.size[1]/ow, self.size[0]/oh
            for line in open(lbl_path):
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls = int(float(parts[0]))
                if cls >= self.num_classes: continue
                xc, yc, w, h = map(float, parts[1:5])
                x1 = max(0, (xc-w/2)*ow*sx); y1 = max(0, (yc-h/2)*oh*sy)
                x2 = min(self.size[1], (xc+w/2)*ow*sx); y2 = min(self.size[0], (yc+h/2)*oh*sy)
                if x2>x1 and y2>y1: boxes.append([x1,y1,x2,y2]); labels.append(cls+1)
        target = {'boxes': torch.tensor(boxes,dtype=torch.float32) if boxes else torch.zeros((0,4)),
                  'labels': torch.tensor(labels,dtype=torch.int64) if labels else torch.zeros(0,dtype=torch.int64)}
        return img_tensor, target

def collate_fn(batch): return tuple(zip(*batch))

# ============================================================
# ONLINE DISTILLATION TRAINER
# ============================================================
class OnlineDistillationTrainer:
    def __init__(self, config, student_name, student_cfg, device):
        self.config = config; self.student_name = student_name
        self.student_cfg = student_cfg; self.device = device
        self.init_type = student_cfg['type']
        self.dcfg = config.get('online_distillation', {})
        
        self.teacher = None
        if self.init_type == 'online_distilled':
            import lightly_train
            logger.info("Loading teacher (LTDETR)...")
            # Загружаем модель правильно, как указано в документации
            self.teacher = lightly_train.load_model(config['teacher']['weights'])
            self.teacher.to(device)
            self.teacher.eval()
            for p in self.teacher.parameters(): 
                p.requires_grad = False
            logger.info(f"  Teacher loaded, classes: {self.teacher.classes if hasattr(self.teacher, 'classes') else 'unknown'}")
        
        self.model = self._create_model(); self.model.to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=student_cfg.get('lr', 0.001), 
            weight_decay=student_cfg.get('weight_decay', 0.0005)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=student_cfg['epochs'], eta_min=1e-6
        )
        
        dp = Path(config['detection']['data_path'])
        train_ds = DefectDataset(dp/'train'/'images', dp/'train'/'labels', config['detection']['num_classes'])
        val_ds = DefectDataset(dp/'val'/'images', dp/'val'/'labels', config['detection']['num_classes'])
        self.train_loader = DataLoader(train_ds, batch_size=student_cfg['batch'], shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=student_cfg['batch'], shuffle=False, num_workers=2, collate_fn=collate_fn)
        
        self.best_map = 0.0; self.best_epoch = 0
        self.patience = student_cfg.get('patience',20); self.patience_counter = 0
        self.output_dir = Path(config['paths']['detection_output']) / student_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_model(self):
        pretrained = self.init_type in ['imagenet_pretrained', 'online_distilled']
        backbone = resnet_fpn_backbone('resnet18', pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=self.config['detection']['num_classes']+1)
        logger.info(f"  Model: Faster R-CNN ResNet-18 FPN, pretrained={pretrained}, lr={self.student_cfg.get('lr',0.001)}")
        return model
    
    def _get_teacher_predictions_batch(self, images_tensors):
        """
        Получить предсказания teacher для батча.
        Согласно документации, model.predict() принимает путь к изображению или PIL Image.
        """
        teacher_preds = []
        for img_tensor in images_tensors:
            # Конвертируем тензор в PIL Image (как требует документация)
            img_np = (img_tensor.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # Используем метод predict() как указано в документации
            # Результат содержит 'bboxes', 'scores', 'labels'
            pred = self.teacher.predict(img_pil)
            
            teacher_preds.append({
                'boxes': torch.tensor(pred['bboxes'], device=self.device).float(),
                'scores': torch.tensor(pred['scores'], device=self.device).float(),
                'labels': torch.tensor(pred['labels'], device=self.device).long(),
            })
        return teacher_preds
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0; total_det = 0.0; total_dist = 0.0
        use_distill = self.teacher and epoch > 3 and epoch <= self.dcfg.get('teacher_epochs', 40)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Supervised detection loss
            loss_dict = self.model(images, targets)
            det_loss = sum(loss_dict.values())
            loss = det_loss
            distill_loss = torch.tensor(0.0, device=self.device)
            
            # Prediction-level distillation (только scores!)
            if use_distill:
                teacher_preds = []
                with torch.no_grad():
                    for img_tensor in images:
                        img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_np)
                        pred = self.teacher.predict(img_pil)
                        teacher_preds.append({
                            'scores': pred['scores'].detach().clone().to(self.device).float()
                        })
                
                stacked = torch.stack(images)
                student_preds = self.model(stacked)
                
                d_loss = 0.0; n_pairs = 0
                for t, s in zip(teacher_preds, student_preds):
                    if len(t['scores']) > 0 and len(s['scores']) > 0:
                        n = min(len(t['scores']), len(s['scores']))
                        d_loss += F.mse_loss(s['scores'][:n], t['scores'][:n])
                        n_pairs += 1
                
                distill_loss = d_loss / max(n_pairs, 1)
                
                if batch_idx == 0 and epoch == 4:
                    logger.info(f"    [DISTILL] pairs={n_pairs}, distill_loss={distill_loss.item():.4f}")
                
                loss = 0.7 * det_loss + 0.3 * distill_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_det += det_loss.item()
            total_dist += distill_loss.item()
            
            if batch_idx % 100 == 0 or batch_idx == len(self.train_loader) - 1:
                logger.info(f"    Batch {batch_idx:4d}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
        
        n = len(self.train_loader)
        logger.info(f"  Epoch {epoch} avg | Total: {total_loss/n:.4f} | Det: {total_det/n:.4f} | Dist: {total_dist/n:.4f}")
        return total_loss / n
    
    def validate(self):
        self.model.eval()
        preds, gts = [], []; total_pred, total_gt = 0, 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                for out, tgt in zip(outputs, targets):
                    keep = out['scores'] > 0.01
                    total_pred += keep.sum().item(); total_gt += len(tgt['boxes'])
                    preds.append({'boxes':out['boxes'][keep].cpu(),'scores':out['scores'][keep].cpu(),'labels':(out['labels'][keep]-1).cpu()})
                    gts.append({'boxes':tgt['boxes'].cpu(),'labels':(tgt['labels']-1).cpu()})
        aps = [self._ap(preds,gts,c,0.5) for c in range(self.config['detection']['num_classes'])]
        mAP = float(np.mean(aps))
        logger.info(f"  Pred: {total_pred}, GT: {total_gt} | mAP@50: {mAP:.4f}")
        return {'mAP_50': mAP}
    
    def _ap(self, preds, gts, cls, iou_thr):
        dets=[]; n_gt=0
        for i,(p,g) in enumerate(zip(preds,gts)):
            gm=g['labels']==cls; n_gt+=gm.sum().item()
            pm=p['labels']==cls
            for b,s in zip(p['boxes'][pm],p['scores'][pm]): dets.append({'img':i,'score':float(s),'box':b})
        if n_gt==0 or len(dets)==0: return 0.0
        dets.sort(key=lambda x:x['score'], reverse=True)
        matched={i:[False]*(g['labels']==cls).sum().item() for i,g in enumerate(gts)}
        tp=np.zeros(len(dets)); fp=np.zeros(len(dets))
        for i,d in enumerate(dets):
            g=gts[d['img']]; gm=g['labels']==cls; gb=g['boxes'][gm]
            if len(gb)==0: fp[i]=1; continue
            ious=box_iou(d['box'].unsqueeze(0),gb)[0]; bi,bv=ious.max(0)
            if bv>=iou_thr and not matched[d['img']][int(bi.item())]: tp[i]=1; matched[d['img']][int(bi.item())]=True
            else: fp[i]=1
        tp_cum=np.cumsum(tp); fp_cum=np.cumsum(fp)
        rec=tp_cum/n_gt; prec=tp_cum/np.maximum(tp_cum+fp_cum,1e-16)
        ap=0.0
        for t in np.linspace(0,1,11):
            if np.any(rec>=t): ap+=float(np.max(prec[rec>=t]))
        return ap/11.0
    
    def train(self):
        epochs=self.student_cfg['epochs']; history=[]; t0=time.time(); best_state=None
        for epoch in range(1,epochs+1):
            train_loss=self.train_epoch(epoch); metrics=self.validate()
            history.append({'epoch':epoch,'train_loss':train_loss,'val_map50':metrics['mAP_50']})
            star="⭐" if metrics['mAP_50']>self.best_map else ""
            logger.info(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | mAP@50: {metrics['mAP_50']:.4f} {star}")
            if metrics['mAP_50']>self.best_map:
                self.best_map=metrics['mAP_50']; self.best_epoch=epoch; self.patience_counter=0
                best_state=copy.deepcopy(self.model.state_dict())
                torch.save({'model':best_state,'mAP':self.best_map}, self.output_dir/'best_model.pth')
            else: self.patience_counter+=1
            if self.patience_counter>=self.patience: logger.info(f"Early stop epoch {epoch}"); break
            self.scheduler.step()
        if best_state: self.model.load_state_dict(best_state)
        torch.save({'model':self.model.state_dict(),'mAP':self.best_map}, self.output_dir/'model_final.pth')
        json.dump({'history':history}, open(self.output_dir/'training_results.json','w'), indent=2)
        elapsed=(time.time()-t0)/3600
        logger.info(f"Done {elapsed:.1f}h | Best mAP@50: {self.best_map:.4f}")
        return {'model_name':self.student_name,'init_type':self.init_type,'best_val_map50':self.best_map,'best_epoch':self.best_epoch,'epochs_trained':len(history),'training_time_hours':round(elapsed,3),'history':history}

def main():
    config = yaml.safe_load(open(Path(__file__).parent.parent/"config_online_distillation.yaml"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []
    summary_path = Path(config['paths']['detection_output']) / "all_results.json"
    
    for name, cfg in config['students'].items():
        output_path = Path(config["paths"]["detection_output"]) / name / "model_final.pth"
        if output_path.exists():
            logger.info(f"\n{'#'*60}\nSkipping {name} - already trained\n{'#'*60}")
            continue
        
        logger.info(f"\n{'#'*60}\nTraining: {name}\n{'#'*60}")
        t = OnlineDistillationTrainer(config, name, cfg, device)
        all_results.append(t.train())
        json.dump(all_results, open(summary_path, 'w'), indent=2)
    
    logger.info(f"\nResults: {summary_path}")
    for r in all_results: 
        logger.info(f"  {r['model_name']:<35} mAP@50: {r['best_val_map50']:.4f}")

if __name__ == '__main__': 
    main()
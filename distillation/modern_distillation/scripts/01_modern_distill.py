#!/usr/bin/env python3
"""Современная дистилляция 2025-2026"""
import sys, logging
from pathlib import Path
from datetime import datetime
import torch, yaml
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.modern_distiller import ModernDistiller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(Path(__file__).parent.parent/'logs'/'modern_distill.log', mode='w')])
logger = logging.getLogger(__name__)

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, d, sz=(224,224)):
        self.imgs = sorted([f for f in Path(d).rglob("*") if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
        self.t = transforms.Compose([transforms.Resize(sz), transforms.RandomHorizontalFlip(0.5), transforms.ColorJitter(0.2,0.2), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        logger.info(f"Found {len(self.imgs)} images")
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        try: return self.t(Image.open(self.imgs[i]).convert('RGB')), 0
        except: return torch.zeros(3,224,224), 0

def main():
    config = yaml.safe_load(open(Path(__file__).parent.parent/'config_modern_distillation.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcfg = config['modern_distillation']
    
    import lightly_train
    teacher = lightly_train.load_model(config['teacher']['weights'])
    if hasattr(teacher, 'backbone'): teacher = teacher.backbone
    
    student = resnet18(pretrained=False)
    dl = DataLoader(UnlabeledDataset(dcfg['unlabeled_data'], tuple(dcfg['image_size'])), batch_size=dcfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    out = Path(config['paths']['pretrain_output'])/'modern_distilled'
    distiller = ModernDistiller(teacher, student, config, device)
    distiller.distill(dl, dcfg['epochs'], out)
    
    (Path(config['paths']['pretrain_output'])/'modern_model_path.txt').write_text(str(out/'backbone_weights.pt'))
    logger.info(f"Done! {out/'backbone_weights.pt'}")

if __name__=='__main__': main()
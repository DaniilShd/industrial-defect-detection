#!/usr/bin/env python3
"""Гибридная дистилляция знаний"""
import sys, logging
from pathlib import Path
from datetime import datetime
import torch, yaml
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.hybrid_distiller import HybridDistiller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(Path(__file__).parent.parent/'logs'/'hybrid_distill.log', mode='w')])
logger = logging.getLogger(__name__)

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: Path, img_size=(224, 224)):
        self.images = sorted([f for f in Path(image_dir).rglob("*") if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
        self.transform = transforms.Compose([transforms.Resize(img_size), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        try: return self.transform(Image.open(self.images[idx]).convert('RGB')), 0
        except: return torch.zeros(3, 224, 224), 0

def main():
    config = yaml.safe_load(open(Path(__file__).parent.parent/'config_hybrid_distillation.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    distill_cfg = config['hybrid_distillation']
    
    import lightly_train
    teacher = lightly_train.load_model(config['teacher']['weights'])
    if hasattr(teacher, 'backbone'): teacher = teacher.backbone
    
    student = resnet18(pretrained=False)
    dataloader = DataLoader(UnlabeledDataset(Path(distill_cfg['unlabeled_data']), tuple(distill_cfg['image_size'])), batch_size=distill_cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    output_dir = Path(config['paths']['pretrain_output'])/'hybrid_distilled'
    distiller = HybridDistiller(teacher, student, config, device)
    distiller.distill(dataloader, distill_cfg['epochs'], output_dir)
    
    cache_file = Path(config['paths']['pretrain_output'])/'hybrid_model_path.txt'
    cache_file.write_text(str(output_dir/'backbone_weights.pt'))
    logger.info(f"Done! Backbone: {output_dir/'backbone_weights.pt'}")

if __name__ == '__main__': main()
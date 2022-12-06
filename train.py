from math import gamma
import os
import yaml
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import SegmentationDataset
from model import Unet
from loss import FocalLoss, dice_loss

class Args():
    def __init__(self):
        self.config = 'train_config.yaml'
args = Args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(model, loss_fn, optimizer, data_loader, epoch, data_type):
    d_loss = dice_loss
    f_loss = FocalLoss(gamma=5)

    running_loss = 0

    data_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)    
    for i, (imgs, masks) in data_loop:
        imgs = imgs.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.long)
        
        mask_preds = model(imgs)

        # loss = loss_fn(mask_preds, masks)
        loss = d_loss(mask_preds, masks) + f_loss(mask_preds, masks)
        
        running_loss += loss.item()

        if data_type=='train':
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step() 
    
        loop_description = "{} epoch {}".format(data_type, epoch)
        data_loop.set_description(loop_description)
        data_loop.set_postfix(loss=loss.item())    

    epoch_loss = running_loss / (i+1)

    return epoch_loss

def train_model(model, cfg):
    if not os.path.exists(cfg["model_save_root"]):
        os.makedirs(cfg["model_save_root"])

    TRAIN_SAVE_PATH = os.path.join(cfg["model_save_root"], cfg["exp_name"]+"_train.pt")
    VAL_SAVE_PATH = os.path.join(cfg["model_save_root"], cfg["exp_name"]+".pt")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, cfg["lr"])

    train_dataset = SegmentationDataset(cfg["train_csv_path"])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True
    )
    
    val_dataset = SegmentationDataset(cfg["val_csv_path"], eval=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False
    )    

    loss_fn = dice_loss#FocalLoss(gamma=5) # torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    writer = SummaryWriter(os.path.join('logs', cfg['exp_name'])) 

    best_train_loss = 1000
    best_val_loss = 1000
    best_val_acc = 0
    patience_lvl = 0

    if cfg["cont_epoch"] is not None:
        start_epoch = cfg["cont_epoch"]
    else:
        start_epoch = 0

    for epoch in range(start_epoch, cfg["epochs"]+start_epoch):
        model.train()
        train_loss = run_epoch(model, loss_fn, optimizer, train_dataloader, epoch, "train")
        writer.add_scalar("loss/train_loss", train_loss, epoch)

        model.eval()
        val_loss = run_epoch(model, loss_fn, None, val_dataloader, epoch, "val")
        writer.add_scalar("loss/val_loss", val_loss, epoch)

        print("Epoch {} - Train Loss: {} - Val Loss: {}".format(epoch, train_loss, val_loss))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), TRAIN_SAVE_PATH)

        if val_loss < best_val_loss:
            patience_lvl = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), VAL_SAVE_PATH)
        else:
            patience_lvl += 1
            if patience_lvl >= cfg['patience']:
                print("Early stopping at Epoch {} - Best Val Final Acc: {}".\
                     format(epoch, best_val_acc))
                break

if __name__=='__main__':
    with open(args.config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    model = Unet(classes=10)

    if cfg["model_load_path"] is not None:
        print("Loading model from {}".format(cfg["model_path"]))
        model.load_state_dict(torch.load(cfg["model_path"]))

    model.to(device)

    train_model(model, cfg)
import os
import csv
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_png
from dataset import create_dataloaders
from wavelet_ae import WaveletAutoEncoder
from loss import CompressionLoss
from metrics import psnr, msssim


class Trainer:
    def __init__(self, data_dir, log_dir=None) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = WaveletAutoEncoder(self.device)
        self.model = self.model.to(device=self.device)
        
        self.loss_func = CompressionLoss()
        self.optimizer = optim.Adam(self.model.parameters(), 1e-4)
        self.epochs = 100
        self.batch_size = {"train": 8, "valid": 1}

        self.data_loaders = create_dataloaders(data_dir, self.batch_size)
        self.csv_epoch_data = {"train": {"full loss": [],
                                    "recon loss": [],
                                    "entropy loss": []},
                            "valid": {"psnr": [],
                                    "msssim": [],
                                    "full loss": [],
                                    "recon loss": [],
                                    "entropy loss": []}}


        self.writer = SummaryWriter(log_dir)

    def write_csv(self, file_name, data_dict):

        log_dir = self.writer.log_dir
        file_path = os.path.join(log_dir, file_name + ".csv")

        with open(file_path, 'w') as csvfile:    
            csv_writer = csv.writer(csvfile)
            keys = sorted(data_dict.keys())
            csv_writer.writerow(keys)
            csv_writer.writerows(zip(*[data_dict[key] for key in keys]))

        print('wrote csv at %s' % file_path)

    def save_model(self, write_dir=None):
        log_dir = self.writer.log_dir if write_dir is None else write_dir
        model_path = os.path.join(log_dir, "final_model.pt")
        torch.save(self.model.state_dict(), model_path)
        print('saved model at %s' % model_path)

    def load_model(self, model_dict_path):
        self.model.load_state_dict(torch.load(model_dict_path))
        self.model = self.model.to(device=self.device)
        print('loaded model from %s' % model_dict_path)

    def eval(self, save_images=True, data_set_name='valid'):
        with torch.no_grad():
            self.model.eval()
            data_loader = self.data_loaders[data_set_name]
            num_batches = len(data_loader)
            
            images_dir = os.path.join(self.writer.log_dir, "eval_" + data_set_name)
            if data_loader.batch_size != 1:
                save_images = False
                print("evaluation cannot save images for data loader with batch size larger than 1!")

            if save_images:
                if not os.path.exists(images_dir):
                    os.mkdir(images_dir)
            
            msssim_score = 0
            psnr_score = 0
            for batch_idx, images in enumerate(data_loader):
                print("evaluating batch %d/%d" % (batch_idx + 1, num_batches))
                images = images.to(self.device)
                recon_images, _ = self.model(images)

                msssim_score += msssim(recon_images, images)
                psnr_score += psnr(recon_images, images)
                
                if save_images:
                    recon_images = (((recon_images + 1.) / 2.) * 255).to(torch.uint8)
                    write_png(recon_images.squeeze(0).cpu(), os.path.join(images_dir, "%d.png" % (batch_idx + 1)))

        return psnr_score.item() / num_batches, msssim_score.item() / num_batches

    def validate(self, epoch):
        with torch.no_grad():
            self.model.eval()
            num_batches = len(self.data_loaders['valid'])
            
            averages = {
                "psnr": 0,
                "msssim": 0,
                "full loss": 0,
                "recon loss": 0,
                "entropy loss": 0
            }

            for batch_idx, images in enumerate(self.data_loaders['valid']):
                print("Validating on batch %d/%d" % (batch_idx + 1, num_batches))

                images = images.to(self.device)
                recon_images, symbols = self.model(images)

                full_loss, recon_loss_val, entropy_loss_val = self.loss_func(recon_images, images, symbols)
                
                images = (images + 1.) * 0.5
                recon_images = (recon_images + 1.) * 0.5

                averages['psnr'] += psnr(images, recon_images).item()
                averages["msssim"] += msssim(images, recon_images).item()
                averages["full loss"] += full_loss.item()
                averages["recon loss"] += recon_loss_val
                averages["entropy loss"] += entropy_loss_val
            
            for key in averages:
                averages[key] /= num_batches
                
            for key, val in averages.items():
                self.csv_epoch_data['valid'][key].append(val)

            print("Validation scores:", averages)
            self.writer.add_scalars("Loss/valid", {"full loss": averages["full loss"], "recon loss": averages["recon loss"],
                                                   "entropy loss": averages["entropy loss"]}, epoch)
            self.writer.add_scalar("PSNR/valid", averages['psnr'], epoch)
            self.writer.add_scalar("MSSSIM/valid", averages['msssim'], epoch)
            self.writer.add_images("Images/GT", images, epoch)
            self.writer.add_images("Images/reconstructed", recon_images, epoch)

    def train(self):
        
        for epoch in range(self.epochs):
            print("Training epoch %d/%d" % (epoch + 1, self.epochs))
            self.model.train()
            num_batches = len(self.data_loaders['train'])

            averages = {"full loss": 0, "recon loss": 0, "entropy loss": 0}

            for batch_idx, patches in enumerate(self.data_loaders["train"]):
                self.optimizer.zero_grad()

                patches = patches.to(self.device)
                recon_patch, symbols = self.model(patches)

                full_loss, recon_loss_val, entropy_loss_val = self.loss_func(recon_patch, patches, symbols)
                # full_loss += wavelet_loss
                full_loss.backward()
                self.optimizer.step()

                full_loss_val = full_loss.item()
                if batch_idx % 20 == 0:
                    print(f"Batch {batch_idx + 1}/{num_batches}: Full loss: {full_loss_val}, Recon loss: {recon_loss_val}, Entropy loss: {entropy_loss_val}")#, Wavelet loss: {wavelet_loss.item()}")
                
                averages["full loss"] += full_loss_val
                averages["recon loss"] += recon_loss_val
                averages["entropy loss"] += entropy_loss_val

                self.writer.add_scalars("Loss/train", {"full loss": full_loss_val, "recon loss": recon_loss_val,
                                                       "entropy loss": entropy_loss_val}, batch_idx + epoch * num_batches)
            for key in averages:
                averages[key] /= num_batches
            
            for key, val in averages.items():
                self.csv_epoch_data['train'][key].append(val)

            self.validate(epoch)
        
        self.writer.flush()
        self.write_csv('train_data', self.csv_epoch_data['train'])
        self.write_csv('valid_data', self.csv_epoch_data['valid'])
        return self.model


if __name__ == "__main__":
    data_dir = "/home/orweiser/university/Digital-Processing-of-Single-and-Multi-Dimensional-Signals/data"
    trainer = Trainer(data_dir)
    trainer.train()
    trainer.save_model()
    print(trainer.eval())

    # trainer.load_model('/home/orweiser/university/Digital-Processing-of-Single-and-Multi-Dimensional-Signals/runs/Sep28_16-32-46_ub-orweiser/final_model.pt')
    # print(trainer.eval())


import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import monai
import matplotlib.pyplot as plt
import copy
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os
class model():
    def train(model, train_loader, val_loader, lr,model_tag, epochs=1):
        min_loss = torch.inf
        train_loss = list()
        val_loss = list()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer = opt,step_size= 70, gamma= 0.9)

        criterion = nn.MSELoss()
        tensorboard_path = '/staging/nnigam/exp_results/runs'

        writer = SummaryWriter(tensorboard_path)

        best_model = model
        for e in range(epochs):
            running_loss = 0.0
            model.to(device)
            model.train()
            for inputs, targets in iter(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                opt.zero_grad()
                outputs = model(inputs)
                loss = criterion(targets, outputs)
                running_loss += loss.item()
                loss.backward()
                opt.step()
                

            if e % 1 == 0: 
                print ('epoch = %d training loss = %.4f' %(e, running_loss))
            train_loss.append(running_loss)
            writer.add_scalar('Training loss',running_loss,  global_step=e)

    
            running_loss = 0.0 
            model.eval()
            with torch.no_grad():
                for inputs, targets in iter(val_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)          
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    running_loss += loss.item()

                if  e % 1 == 0:
                    print ('epoch = %d validation loss = %.4f' %(e, running_loss))
                val_loss.append(running_loss)
                writer.add_scalar('Validation loss',running_loss,  global_step=e)

            if val_loss[-1] < min_loss:
                min_loss = val_loss[-1]
              # saving models for lowest loss
                print ('saving model at epoch e = %d' %(e))
                model.to("cpu")

                path = '/staging/nnigam/exp_results/'
                torch.save(model,path+model_tag)
                best_model = copy.deepcopy(model)

                for inputs, targets in val_loader:
                    axes = plt.subplot(1,3,1)
                    plt.imshow(torch.rot90(inputs[0][0][:,:,16]), cmap='gray')
                    plt.xlabel("Noisy Input")
                    axes = plt.subplot(1,3,2)
                    plt.imshow(torch.rot90(targets[0][0][:,:,16]), cmap='gray')
                    plt.xlabel("Denoised Output")
                    device = "cpu"
                    model= model.to(device)
                    outputs = model(inputs.to(device))
                    axes = plt.subplot(1,3,3)
                    outputs = outputs.detach().numpy()
                    outputs[outputs<0] = 0
                    outputs[outputs>1] = 1#np.clip(outputs,0,1)
                    plt.imshow(torch.rot90(torch.tensor(outputs[0][0][:,:,16])), cmap='gray')
                    plt.xlabel("Reconstructed Output")
                    writer.add_figure('Overall Reconstruction', plt.gcf(), global_step=e)
                    break
                
            scheduler.step()

        print ('completed training')
        #self.net = nn
        return best_model, train_loss, val_loss

    def loss_curves(model_tag,train_loss, val_loss):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize = (6, 6))
        f.suptitle('emotic')
        ax1.plot(range(0,len(train_loss)),train_loss, color='Blue')
        ax2.plot(range(0,len(val_loss)),val_loss, color='Red')
        ax1.legend(['train'])
        ax2.legend(['val'])
        path = '/staging/nnigam/exp_results/'
        plt.savefig(path+'Loss_Curves_'+model_tag+'.png')

    def test(model, test_loader):
        min_loss = torch.inf

        test_loss = list()
        test_psnr = list()
        results = {}
        #model_context, model_body, emotic_model = models

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #opt = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.MSELoss()
        model.to(device)
    
        #model.eval()
        with torch.no_grad():
          running_loss = 0
          psnr = 0
          for inputs, targets in iter(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)          
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            #psnr = piq.psnr(torch.clamp(outputs, 0,1), targets, data_range=1.).item()
            psnr = -10*torch.log10(loss)
            #running_loss += loss.item()
            test_loss.append(loss.item())
            test_psnr.append(psnr)
            
        
        print("Avg PSNR:", sum(test_psnr)/len(test_psnr))
        print("Avg loss:", sum(test_loss)/len(test_loss))
        results['psnr'] = sum(test_psnr)/len(test_psnr)
        results['loss'] = sum(test_loss)/len(test_loss)

        return results

    def infer(model, test_loader,model_tag):
        model.eval()
        for inputs, targets in test_loader:
          axes = plt.subplot(1,3,1)
          plt.imshow(torch.rot90(inputs[0][0][:,:,16]), cmap='gray')
          plt.xlabel("Noisy Input")
          axes = plt.subplot(1,3,2)
          plt.imshow(torch.rot90(targets[0][0][:,:,16]), cmap='gray')
          plt.xlabel("Denoised Output")
          device = "cpu"
          model= model.to(device)
          outputs = model(inputs.to(device))
          axes = plt.subplot(1,3,3)
          outputs = outputs.detach().numpy()
          outputs[outputs<0] = 0
          outputs[outputs>1] = 1#np.clip(outputs,0,1)
          plt.imshow(torch.rot90(torch.tensor(outputs[0][0][:,:,16])), cmap='gray')
          plt.xlabel("Reconstructed Output")
          path = '/staging/nnigam/exp_results/'
          plt.savefig(path+'Inference_Image_'+model_tag+'.png')
          break


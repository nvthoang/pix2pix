from genericpath import isdir
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as T
import numpy as np
import time
import os

import cv2
def chm2binary(chm_imgs):
    threshs=[]
    for i in range(len(chm_imgs)):
        _, thresh = cv2.threshold(chm_imgs[i], 5,255,cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        threshs.append(thresh)
    threshs = np.array(threshs).reshape(len(chm_imgs), 1, 1024, 1024)  
    return torch.Tensor(threshs)

#loss function
def generator_loss(generated_image, target_img, G, real_target, lf='mse', l1:bool=False):
    assert lf in ['bce', 'mse']
    adversarial_loss = nn.BCELoss() if lf=='bce' else nn.MSELoss()
    l1_loss = nn.L1Loss()
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img) if l1 else 0
    gen_total_loss = gen_loss + (100 * l1_l)
    return gen_total_loss

def discriminator_loss(output, label, lf='mse'):
    assert lf in ['bce', 'mse']
    adversarial_loss = nn.BCELoss() if lf=='bce' else nn.MSELoss()
    disc_loss = adversarial_loss(output, label)
    return disc_loss

if os.path.isdir("./checkpoints")==False:
    os.makedirs("./checkpoints")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#training function
def training(model,
             train_dl,
             eval_dl=None,
             num_epochs:int=50,
             lr:float=0.0001,
             patch_dim:tuple=(30, 30),
             resize_to:tuple=(1024,1024),
             save_model_per_n_epoch:int=None,
             start_epoch:int=1,
             start_val_loss:float=100.0,
             current_best_epoch:int=1,
             loss_function='bce'):
    '''
    model: list [generator, discriminator]
    train_dl: data_loader for training set
    test_ld: data_loader for val/test set
    num_epochs: num of training epochs
    patch_dim: dim of patch returned by discriminator
    '''
    assert loss_function in ['bce', 'mse']
    resize_img=T.Resize(resize_to)
    generator, discriminator=model[0], model[1]
    G_optimizer=torch.optim.Adam(generator.parameters(), lr=lr)
    D_optimizer=torch.optim.Adam(discriminator.parameters(), lr=lr)
    G_train_losses, D_train_losses = [], []
    G_test_losses, D_test_losses = [], []
    val_total_loss=start_val_loss
    current_best_epoch=current_best_epoch
    #===========================
    #Training
    for epoch in range(start_epoch, start_epoch+num_epochs): 
        generator.train()
        discriminator.train()
        G_train_loss=D_train_loss= 0.0
        start=time.time()
        num_batch=len(train_dl.dataset)
        for input_img, target_img in train_dl:
            D_optimizer.zero_grad()
            input_img=resize_img(input_img)
            target_img=resize_img(target_img)
            input_img=input_img.to(device)
            target_img=target_img.to(device)
            #ground truth labels real and fake
            real_target=Variable(torch.ones(input_img.size(0), 1, patch_dim[0], patch_dim[1]).to(device)) #8*1*126*126
            fake_target=Variable(torch.zeros(input_img.size(0), 1, patch_dim[0], patch_dim[1]).to(device)) #8*1*126*126
            #generator forward pass
            generated_image=generator(input_img) #8*1*1024*1024
            #train discriminator with fake/generated images
            #NOTICE: BINARIZE the generated image for Descriminator
            generated_image_binarized=chm2binary(generated_image.detach().cpu().numpy().squeeze(1)).to(device)
            disc_inp_fake=torch.cat((input_img, generated_image_binarized), 1) #8*5*1024*1024 NOTICE
            # disc_inp_fake=torch.cat((input_img, generated_image), 1) #8*5*1024*1024
            D_fake=discriminator(disc_inp_fake.detach()) #8*1*126*126
            D_fake_loss=discriminator_loss(D_fake, fake_target)
            #train discriminator with real images
            #NOTICE: BINARIZE the target image for Descriminator
            target_img_binarized=chm2binary(target_img.detach().cpu().numpy().squeeze(1)).to(device)
            # disc_inp_real=torch.cat((input_img, target_img), 1) #8*5*1024*1024
            disc_inp_real=torch.cat((input_img, target_img_binarized), 1) #8*5*1024*1024 #NOTICE
            D_real=discriminator(disc_inp_real) #8*1*126*126
            D_real_loss=discriminator_loss(D_real, real_target)
            #average discriminator loss
            D_total_loss=(D_real_loss + D_fake_loss)/2
            D_train_loss+=D_total_loss
            #compute gradients and run optimizer step
            D_total_loss.backward()
            D_optimizer.step()
            #train generator with real labels
            G_optimizer.zero_grad()
            # fake_gen=torch.cat((input_img, generated_image), 1)
            fake_gen=torch.cat((input_img, generated_image_binarized), 1)
            G=discriminator(fake_gen)
            G_loss=generator_loss(generated_image, target_img, G, real_target)                                 
            G_train_loss+=G_loss
            #compute gradients and run optimizer step
            G_loss.backward()
            G_optimizer.step()
        G_train_losses.append(G_train_loss.detach().cpu().numpy()/num_batch)
        D_train_losses.append(D_train_loss.detach().cpu().numpy()/num_batch)
        end=time.time()
        #==========================
        if save_model_per_n_epoch!=None:
            if epoch%save_model_per_n_epoch==0:
                torch.save(generator.state_dict(), f"./checkpoints/generator_{epoch}.pth")
                torch.save(discriminator.state_dict(), f"./checkpoints/discriminator_{epoch}.pth")
        #==========================
        #Save current epoch for retraining if runtime error
        torch.save(generator.state_dict(), f"./checkpoints/current_generator.pth")
        torch.save(discriminator.state_dict(), f"./checkpoints/current_discriminator.pth")
        #==========================
        #Log
        log=f"Training: Epoch {epoch} - G loss: {G_loss/num_batch}, D loss: {D_total_loss/num_batch}, Time: {end-start}"
        print(log)
        if epoch!=1:
            with open("./log.txt", "a") as f:
                f.writelines(log + "\n")
        else:
            with open("./log.txt", "w") as f:
                f.writelines(log + "\n")
        f.close()
        #===========================
        #Evaluation
        if eval_dl!=None:
            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                num_batch=len(eval_dl.dataset)
                G_test_loss=D_test_loss=0.0
                for input_img, target_img in eval_dl:
                    input_img=resize_img(input_img)
                    target_img=resize_img(target_img)
                    input_img=input_img.to(device)
                    target_img=target_img.to(device)
                    #ground truth labels real and fake
                    real_target=Variable(torch.ones(input_img.size(0), 1, patch_dim[0], patch_dim[1]).to(device)) 
                    fake_target=Variable(torch.zeros(input_img.size(0), 1, patch_dim[0], patch_dim[1]).to(device)) 
                    #generator forward pass
                    generated_image=generator(input_img)
                    #train discriminator with fake/generated images
                    generated_image_binarized=chm2binary(generated_image.detach().cpu().numpy().squeeze(1)).to(device)
                    disc_inp_fake=torch.cat((input_img, generated_image_binarized), 1) #8*5*1024*1024 NOTICE
                    # disc_inp_fake=torch.cat((input_img, generated_image), 1)
                    D_fake=discriminator(disc_inp_fake.detach())
                    D_fake_loss=discriminator_loss(D_fake, fake_target)
                    #train discriminator with real images
                    target_img_binarized=chm2binary(target_img.detach().cpu().numpy().squeeze(1)).to(device)
                    disc_inp_real=torch.cat((input_img, target_img_binarized), 1) #8*5*1024*1024 #NOTICE
                    # disc_inp_real=torch.cat((input_img, target_img), 1)
                    D_real=discriminator(disc_inp_real)
                    D_real_loss=discriminator_loss(D_real, real_target)
                    #average discriminator loss
                    D_total_loss=(D_real_loss + D_fake_loss)/2
                    D_test_loss+=D_total_loss
                    #train generator with real labels
                    # fake_gen=torch.cat((input_img, generated_image), 1)
                    fake_gen=torch.cat((input_img, target_img_binarized), 1)
                    G=discriminator(fake_gen)
                    G_loss=generator_loss(generated_image, target_img, G, real_target)                                 
                    G_test_loss+=G_loss
                G_test_loss=G_test_loss.detach().cpu().numpy()/num_batch    
                D_test_loss=D_test_loss.detach().cpu().numpy()/num_batch
                G_test_losses.append(G_test_loss)
                D_test_losses.append(D_test_loss)
                #===========================
                #Update current best epoch and save model at the epoch
                if G_test_loss + D_test_loss < val_total_loss:
                    val_total_loss = G_test_loss + D_test_loss
                    current_best_epoch = epoch
                    torch.save(generator.state_dict(), f"./checkpoints/best_generator.pth")
                    torch.save(discriminator.state_dict(), f"./checkpoints/best_discriminator.pth")
                #===========================
                #Log
                log=f"Evaluation: Epoch {epoch} - G loss: {G_test_loss/num_batch}, D loss: {D_test_loss/num_batch}"
                print(log)
                print(f"Current best epoch: {current_best_epoch}")
                with open("./log.txt", "a") as f:
                    f.writelines(log + "\n")
                f.close()
    #===========================
    if eval_dl!=None:
        return (G_train_losses, D_train_losses), (G_test_losses, D_test_losses)  
    else:
        return (G_train_losses, D_train_losses)

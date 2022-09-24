import torch 
import numpy as np

#custom weights initialization called on generator and discriminator
def init_weights(net, init_type='normal', scaling=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

# read log file as loss numerical values
def log2loss(log_path:str, num_line:int=None):
    outcome = open(log_path).readlines()
    num_line=len(outcome)-1
    train_G_losses, val_G_losses, train_D_losses, val_D_losses = [], [], [], []
    for i in np.arange(0,num_line,2):
        G_loss=outcome[i].split("-")[1][1:].split(",")[0].split(":")[1][1:]
        D_loss=outcome[i].split("-")[1][1:].split(",")[1].split(":")[1][1:]
        if G_loss.find("e")==-1 and len(G_loss[G_loss.find("e"):])!=1:
            G_loss=float(G_loss)
        else:
            G_loss=float(G_loss[:G_loss.find("e")] + "e-1")
        if D_loss.find("e")==-1 and len(D_loss[D_loss.find("e"):])!=1:
            D_loss=float(D_loss)
        else:
            D_loss=float(D_loss[:D_loss.find("e")] + "e-1")
        train_G_losses.append(G_loss)
        train_D_losses.append(D_loss)
        #-----------------------------------
        G_loss=outcome[i+1].split("-")[1][1:].split(",")[0].split(":")[1][1:]
        D_loss=outcome[i+1].split("-")[1][1:].split(",")[1].split(":")[1][1:]
        if G_loss.find("e")==-1 and len(G_loss[G_loss.find("e"):])!=1:
            G_loss=float(G_loss)
        else:
            G_loss=float(G_loss[:G_loss.find("e")] + "e-1")
        if D_loss.find("e")==-1 and len(D_loss[D_loss.find("e"):])!=1:
            D_loss=float(D_loss)
        else:
            D_loss=float(D_loss[:D_loss.find("e")] + "e-1")
        val_G_losses.append(G_loss)
        val_D_losses.append(D_loss)  
    #-----------------------------------
    train_losses = (train_G_losses, train_D_losses)
    val_losses = (val_G_losses, val_D_losses)
    #-----------------------------------
    return train_losses, val_losses
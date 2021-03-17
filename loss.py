import torch
import torch.nn as nn
criteration =nn.MultiLabelSoftMarginLoss()
def focal(output,label):
    lxy=F.softmax(output)
    Loss1=0
    Loss2=0
    zz=output.size(0)
    for i in range(0,zz):
        a=output[i].view([1,32])#32是特征长度
        b=label[i].view([1,32])
        #print(output.shape)
        #print(label.shape)
        Loss=criteration(a,b)
        qr=lxy[i].mul(label[i])
        Loss1=Loss1+Loss
        qr1=qr.detach().cpu().numpy()
        Loss2=Loss2+((1-qr1.mean())**2)*Loss
        #print(qr)
    Loss1=Loss1/zz
    Loss2=Loss2/zz
    #print(Loss1)
    return Loss2

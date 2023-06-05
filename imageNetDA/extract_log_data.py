import numpy as np

file1 = open('/home/utkarsh/output/cifar10/search/only_rkd_term/log_plain.txt', 'r')
file1 = open('/home/utkarsh/output/cifar10/search-new/gamma_s_1e+7/log_plain.txt', 'r')
file1 = open('/home/utkarsh/output/cifar10/train-new/gamma_s_1e5_100000/log_plain.txt', 'r')
Lines = file1.readlines()

kl_loss = np.zeros(200)
rkd_loss = np.zeros(200)
loss = np.zeros(200)
ce_loss = np.zeros(200)
latency = np.zeros(200)

search=False

if search:
   kl_loss_index = 20
   loss_index = 11
   ce_loss_index = 17 
   latency_index = 14
else:
   kl_loss_index = 17
   loss_index = 11
   ce_loss_index = 14
   latency_index = 11
epoch = 1
count = 0
for  line in Lines:
    #print(len(line))
    if line.find("Epoch")!=-1 and line.find("kl")!=-1 and len(line)>120:
       arr = line.split(' ')
       #print(len(line),arr)
       if int(arr[5])==epoch:
          count+=1
          kl_loss[epoch-1]+=float(arr[kl_loss_index])
          kl_loss[epoch-1]+=float(arr[kl_loss_index])
          loss[epoch-1]+=float(arr[loss_index])
          latency[epoch-1]+=float(arr[latency_index])
          ce_loss[epoch-1]+=float(arr[ce_loss_index])
       else:
          kl_loss[epoch-1]/=count
          loss[epoch-1]/=count
          ce_loss[epoch-1]/=count
          latency[epoch-1]/=count
          count=1
          epoch+=1
          kl_loss[epoch-1]+=float(arr[kl_loss_index])
          loss[epoch-1]+=float(arr[loss_index])
          latency[epoch-1]+=float(arr[latency_index])
          ce_loss[epoch-1]+=float(arr[ce_loss_index])

kl_loss[epoch-1]/=count
loss[epoch-1]/=count
ce_loss[epoch-1]/=count
latency[epoch-1]/=count

#print(kl_loss)

print("kl_loss")
for i in range(kl_loss.shape[0]):
    print(kl_loss[i],end=",")

print("loss")
for i in range(kl_loss.shape[0]):
    print(loss[i],end=",")

print("ce_loss")
for i in range(kl_loss.shape[0]):
    print(ce_loss[i],end=",")

if search:
   print("latency")
   for i in range(kl_loss.shape[0]):
       print(latency[i],end=",")

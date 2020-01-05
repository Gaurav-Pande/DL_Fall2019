Name: Sashank Gondala (Sashank on EvalAI)
Email id: sgondala@gatech.edu
Best accuracy: 83%

I used the following final architecture

- Conv, Conv, Pool, Conv, Linear, Linear

All kernels are of size 3, and padding 1. (Same convolutions)
Each convolution filter has 80 kernels.
Pooling filter is a MaxPool of kernel size 2 and stride 2.
Trained using Adam with LR of 0.001 and Weight decay of 5e-4. 

For data augmentations, I used RandomHorizontalFlip and RandomCrops during training part.

Other ideas: I tried several different strategies 
- Training a Densenet/VGG16 end to end - I got a good enough accuracy but felt those bigger models are an overkill for this.

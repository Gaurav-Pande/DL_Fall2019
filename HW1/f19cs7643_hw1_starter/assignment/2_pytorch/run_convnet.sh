#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
kernel_size=3
hidden_dim=20
epochs=5
python -u train.py \
    --model convnet \
    --kernel-size $kernel_size \
    --hidden-dim $hidden_dim \
    --epochs $epochs \
    --weight-decay 0.95 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.001 | tee convnet_${kernel_size}_${hidden_dim}_${epochs}.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

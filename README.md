# Applied Deep Learning Course Project: The affect of color spaces on learning
Image classification is typically done in RGB space. 
Some papers found thatdifferent color spaces achieve better performance, in some specific tasks. 
In thiswork we tried to learn the best color space for CIFAR-10 classification task, bydirectly learning a per-color embedding. 
This will settle the question of the bestcolor space once and for all.  
During the work we try to understand how colorspaces affect learning, by examining the classification task on CIFAR-10 dataset.

### Usage
python3 ./main.py --help
This will give you all the information you need in order to run the different models.

For example:
python main.py --net_type SimpleConvNetWithEmbedding --embedding_size 32 --init_embedding_as random_poly --embedding_continuity_loss 128 --epochs 100 --lr 0.001 0.005 0.007 0.01 --bs 128 96 64 32 --momentum 0.9 --weight_decay 0.001 0.01 0.1 --device_num 2 --save_model 

This will train the SimpleConvNetWithEmbedding model for 100 epochs, with any combination of the given hyper-parameters (such as learning-rate, batch-size, momentum and weight-decay). It will use as embedding of size 32^3, initialized as random 3D polynomial. It will also use the continuity loss with 128 random colors at each step. 

# SRCNN
keras implementation of Image Super-Resolution Using Deep Convolutional Networks.

*GT* | *Bicubic* | *SRCNN* 
:---: | :---: | :---: |
<img src = 'results/original.png'> | <img src = 'results/mybicubic.jpg'> | <img src = 'results/mysrcnn-result-withMyBicubic.jpg'> 


## Implementation Details

Our implementation used TensorFlow and keras to train SRCNN. We used almost same methods as described in the paper. We trained the network with 91-image dataset (without augmentation) and validated with Set5 dataset while training. At test time, to get the same result size for up-scaling factor 3. Also, we padded border of the ground truth and bicubic interpolated test image by 6 to make same size with SRCNN result. The main difference between own implementation and original paper in using the SSIM error function instead mean square error.

According to the paper, the best performance on Set5 with upscaling factor 3 is the average PSNR value of 32.75dB with filter size 9-5-5 and ImageNet training dataset, but we were **aim to 32.39dB** which is the demonstrated average PSNR value when the model is trained with 91-image dataset, 9-1-5 filter size and Y only. After training 5000 epoch, we got the above result for single image , **25.77dB/0.918**.

Pretrained-model with 91-image training dataset and up-scaling factor 3 is given.

Note that we trained and tested with Y-channel. If you want to train and test with 3-channels (YCbCr or RGB), you may add or implement some type-casting code.


## Installation

```bash
git clone https://github.com/rezaprince/SRCNN.git
```

## Documentation

To pre-process the train and test dataset, you have 2 option :
first: you can execute the Matlab code in prepare-data-with-matlab directory

second: you can execute the python prepare.py code,

Click [here][data] to download the pre-processed training data with 91 dataset. Put the file data directory.

The pre-processed test data with Set5 is provided.

### Training SRCNN
Use `main.py` to train the network. Run `python main.py` to view the training process. Example usage:
```bash
# Quick training
python main.py
```

you can change the options parameters in main.py
```bash
EPOCHS  		#network epochs  	
SCALE_FACTOR		#scale factor for training network
LOSS			#loss function ==> 'mse' or dssimloss
DATA_TRAIN		#prepared train h5 file
DATA_TEST		#prepared test h5 file
MODEL_NAME		#model name to use for load pretrained model or training model save
BATCH_SIZE		
TRAIN			#if false you can use pretrained model and dont train model
```

### Testing SRCNN
Also use `main.py` to test the network. Pretrained-model with 91-image training dataset and up-scaling factor 3 is given. Example usage:
```bash
# Quick testing
change the TRAIN constant in main.py to False and then run

python main.py

remember that you should change MODEL_NAME constant to use your own trained model

```



Please note that if you want to train or test with your own dataset, you need to execute the Matlab code with your own dataset first :)


## Results

### The single result of PSNR (dB)/SSIM trained with up-scale factor 3

*Code* | *Image* | *Scale* | *Bicubic* | *SRCNN*
:---: | :---: | :---: | :---: | :---: |
**SRCNN** | Butterfly | 3x | 24.61dB | 27.57dB/0.902
**SRCNN-SSIM**| Butterfly | 3x | 24.61dB | 25.77dB/0.918

### Some of the result images

 


## References

- [Official Website][1]
    - We referred to the original Matlab and Caffe code.

- [jinsuyoo/SRCNN-Tensorflow][2]
    - We highly followed the structure of this repository.

[data]: https://drive.google.com/file/d/1a52tikv7Za_wtmAEiZFEE1qMowtljb3_/view?usp=sharing
[1]: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
[2]: https://github.com/jinsuyoo/SRCNN-Tensorflow
# Unseen Object Reasoning with Shared Appearance Cues
[ArXiv Paper](https://arxiv.org/pdf/2406.15565)

![Positional Embedding clusters] (image.png)
~~~
conda install -c conda-forge pytorch-lightning
pip install -r requirements.txt
~~~

## run code
To train defalut backbone
~~~
python main_prediction_imagenet_res50.py --dataset imagenet_mini --network default
~~~
To train resnet backbone
~~~
python main_prediction_imagenet_res50.py --dataset imagenet_mini --network res50
~~~

inference:
~~~
python inference.py --eval_exp_id <exp_id> --eval_ckpt_file epoch=<X-step=XXX>.ckpt
~~~

see opts.py for more parameters

# AWS Setup


~~~
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install seaborn matplotlib opencv-python
conda install -c conda-forge pytorch-lightning
~~~


~~~
cd ~
git clone git@github.com:Ridecell/Experimental.git
conda activate pytorch17
cd Experimental
export PYTHONPATH=$PYTHONPATH:$(pwd)
~~~

sandbox cifar 10 training
~~~
python novel_objects/src/scripts/main_prediction_cifar_10.py
~~~

Imagenet training
~~~
ln -s ~/data ~/Experimental/novel_objects/
python novel_objects/src/main_prediction_imagenet_default.py
~~~

TO-DO

- [ ] Add MLFlow to CIFAR-10
- [ ] MLFlow to imagenet default
- [ ] Changing to patch level operations in Imagenet default
- [ ] Write evaluation scripts
- 

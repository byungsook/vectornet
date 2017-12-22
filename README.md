Tensorflow implementation of [Semantic Segmentation for Line Drawing Vectorization Using Neural Networks](http://www.byungsoo.me).

## Requirements

- [anaconda3 / python3.6](https://www.anaconda.com/download/#linux)
- [TensorFlow 1.4](https://github.com/tensorflow/tensorflow)
- [CairoSVG 2.1.2](http://cairosvg.org/)
- [Matplotlib 2.1.0](https://matplotlib.org/)
- [imageio 2.2.0](https://pypi.python.org/pypi/imageio)
- [tqdm](https://github.com/tqdm/tqdm)
- Run 'pip install tensorflow-gpu cairosvg matplotlib imageio tqdm'

## Usage

To train PathNet on Random Line Data Set:
    
    python main.py --archi=path --dataset=line --data_dir='data'

## Results

### Generator output (64x64) with `gamma=0.5` after 300k steps

## Reference

[carpedm20] (https://github.com/carpedm20/BEGAN-tensorflow)

## Author

Byungsoo Kim / [@kimby](http://www.byungsoo.me)

<!-- 
## Useful Settings

anaconda: (Windows) [ImportError: No module named 'pip._vendor.requests.adapters' for any pip command](https://github.com/ContinuumIO/anaconda-issues/issues/6719)

    conda install pip -f

anaconda: (Windows) [dlopen() failed to load a library: cairo / cairo-2](https://github.com/Kozea/CairoSVG/issues/84)

    Install [GTK+](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases/download/2017-11-15/gtk3-runtime-3.22.26-2017-11-15-ts-win64.exe)

git: replace existing folder

    git clone https://myrepo.com/git.git temp
    mv temp/.git code/.git
    rm -rf temp

git: line ending

    git config --global core.autocrlf true # for windows (checkout crlf, commit unix)
    git config --global core.autocrlf input # for linux (checkout as-is, commit unix)

git: save credentials

    git config --global credential.helper 'store --file ~/.git-credentials'

visual studio code: old tasks.json

    "version": "0.1.0",
	"command": "python",
	"isShellCommand": true,
	"args": ["${file}"],
	"showOutput": "always"

visual studio code: default setup of keybindings.json

    { "key": "f7",               "command": "workbench.action.tasks.runTask" },
    { "key": "shift+f7",         "command": "workbench.action.tasks.terminate" },
    { "key": "f6",               "command": "python.execInTerminal" }

visual studio code: specify python version in user/workspace settings

    "python.pythonPath": "~/Anaconda3/envs/py27/python"

visual studio code: stop at the beginning of debugging

    "stopOnEntry": false, (launch.json)
 -->



<!-- 
## Usage

First download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets with:

    $ apt-get install p7zip-full # ubuntu
    $ brew install p7zip # Mac
    $ python download.py

or you can use your own dataset by placing images like:

    data
    └── YOUR_DATASET_NAME
        ├── xxx.jpg (name doesn't matter)
        ├── yyy.jpg
        └── ...

To train a model:

    $ python main.py --dataset=CelebA --use_gpu=True
    $ python main.py --dataset=YOUR_DATASET_NAME --use_gpu=True

To test a model (use your `load_path`):

    $ python main.py --dataset=CelebA --load_path=CelebA_0405_124806 --use_gpu=True --is_train=False --split valid


## Results

### Generator output (64x64) with `gamma=0.5` after 300k steps

![all_G_z0_64x64](./assets/all_G_z0_64x64.png)


### Generator output (128x128) with `gamma=0.5` after 200k steps

![all_G_z0_64x64](./assets/all_G_z0_128x128.png)


### Interpolation of Generator output (64x64) with `gamma=0.5` after 300k steps

![interp_G0_64x64](./assets/interp_G0_64x64.png)


### Interpolation of Generator output (128x128) with `gamma=0.5` after 200k steps

![interp_G0_128x128](./assets/interp_G0_128x128.png)

    
### Interpolation of Discriminator output of real images
    
![alt tag](./assets/AE_batch.png)   
![alt tag](./assets/interp_1.png)   
![alt tag](./assets/interp_2.png)   
![alt tag](./assets/interp_3.png)   
![alt tag](./assets/interp_4.png)   
![alt tag](./assets/interp_5.png)   
![alt tag](./assets/interp_6.png)   
![alt tag](./assets/interp_7.png)   
![alt tag](./assets/interp_8.png)   
![alt tag](./assets/interp_9.png)   
![alt tag](./assets/interp_10.png)


## Related works

- [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [DiscoGAN-pytorch](https://github.com/carpedm20/DiscoGAN-pytorch)
- [simulated-unsupervised-tensorflow](https://github.com/carpedm20/simulated-unsupervised-tensorflow)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io) -->

# vectornet

## How to use

```
Install Anaconda 4.2.0, Python 3.5 version 64bit Installter
Install CUDA 8.0
Install cuDNN 5.1, add path to PATH (W: e.g. cudnn=C:/cuda, PATH=%PATH%;%cudnn%;%cudnn%\bin)
Run 'conda --update conda'
Install Tensorflow r0.12 (https://www.tensorflow.org/get_started/os_setup#anaconda_installation, W: should use '--ignore-installed' flag to avoid 'easy-install.pth' remove error)
Install CairoSVG using pip (pip install cairosvg, W: Install GTK+ (gtk3-runtime-3.20.2-2016-04-09-ts-win64.exe) to avoid [dlopen() failed to load a library: cairo / cairo-2](https://github.com/Kozea/CairoSVG/issues/84) error.)
```
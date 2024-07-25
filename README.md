# PlumeCNN

Identifying the boundry of heat plumes, using the data from direct numerical simulation of Rayleigh–Bénard convection(RBC) in CYLINDRICAL container.

Currently dense neural network is supported for both 2D (sliced) and 3D data, convolutional neural network is only supported for 2D data. Data visualization is also only supported for 2D data.

Now the user have to train the model everytime they use the model, because the structure of a convolutional neural network will depend on the shape of input, it is hard to keep the consistency of data shape throughout each use of this program. Feel free to add the function to save the model after training. Besides, training the model each time only takes about 25 seconds. 

Occationally the program will fail to work, usually just let it rerun once and it will be fine.

## Required Softwares to Install

1. [Nek5000](https://nekx5000.mcs.anl.gov)

    The software used to solve partial differential equations to get the simulation result of RBC.

2. [VisIt](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.0/gui_manual/Intro/Installing_VisIt.html)

    The software used to visualize and rearrange the simulation data from `Nek5000`.

3. [Python](https://www.python.org/downloads/)

    The programming language being used. PlumeCNN support python version higher or equal to 3.6.

## The Output

There are 3 types of outputs you can choose. 

The first one is a `CSV` file, with headers of coordinates, parameters (temperature, velocity magnitude, z-velocity), gradient of  these parameters, and the result of classification with header `is_boundary`. **This is currently the ONLY available output for 3D (unsliced) data.**

 - The coordinates is not the same as the original one. It is been interpolated to an evenly-spaced, user-specified resolution. The min and max value of coordinates are the same. For sliced data, the coordinates will always be x and y, regardless of original slicing direction. 
 - The temperature, z-velocity have been normalized to the range [-1,1], the velocity magnitude has been normalized to the range [0,1].
 - The range of `is_boundary` is [0,1] or [-1,1] based on user's choice. Its magnitude indicates how likely it is to be a boundary of plume. If the range of the data is [-1,1], when it is positive, it indicates the boundry of a hot plume; when it is negative, it indicates the boundry of a cold plume.

The second one is a `PNG` image, making a scatterplot of the value of `is_boundary`. The colormap will vary based on whether the range of `is_boundary` is [0,1] or [-1,1].

The last one is a movie (Sadly, currently not in use). If you want to output a time series of result, this will zip the image of each time frames together, and will consider uneven time differences between each frames.

## Manual

### 1. Get Data

- First, run your simulation in a **CYLINDRICAL CONTAINER** on `Nek5000`. Make sure you normalized data of temperature so that it is supposd to be in the range from 0-1. 

    Then, you need to prepare two sets of data **FROM THIS SIMULATION**: the data to train the model and the data you want to classify. 

    These two sets of data should have the same slicing direction, if they are sliced.

    For the data for training, usually 2 or 3 sliced data is enough.

    For the data to classify, you can output as much as you want.

- Open the simulation result in `VisIt`. You can choose to slice the data perpendicular to x/y/z axis. Currently PlumeCNN only supports 3D database and database sliced perpendicular to x, y, or z axis.

- In `VisIt`, Click `File - Export Database`, in `Xmdv` format. You can choose to use either comma or space to separate in the databse. 

    Below are variables you **must** export, regardless of sequence:

  - coordinates of your grids (An option after you click the button "Export")
  - scalars/temperature
  - scalars/time_derivative/conn_based/mesh_time (If you want to output a movie)
  - scalars/velocity_magnitude
  - scalars/z_velocity
  
  Other variables will not be used. So it is not recommended to export them.

- If you want to output the movie, you have to opt the `Export all time states`, and feel free to change its format, as long as the sequence between files can easily be detected. The program can help you to select the range of files you want to read. **Don't change their filename after you export the data!** Otherwise, the file with modifies name cannot be in the right place within that list of all files to read.

### 2. Install

- Download this whole repository to the computer from [Github](https://github.com/Rorororopu/PlumeCNN).

  Or use the command `git clone <https://github.com/Rorororopu/PlumeCNN.git>' at the directory where you want to store this directory.

  If your HPC is using [slurm](https://slurm.schedmd.com/documentation.html) to manage workload, please:

- use `sinfo` to check available nodes

- `ssh` to an available node like `n001`

- `cd` to its `PlumeCNN` repository

- Create and activate a vitrual environment via **SSH**. Don't input them on your computer using [`sshfs`](https://osxfuse.github.io), because your computer will be confused and take the version on **YOUR PC** of python as the version on the **HPC** you're using. These are commands you need to input:

    `python3 -m venv <name of your virtual environment>`

    `source <name of your virtual environment>/bin/activate`

    You could use the TAB key to let the terminal automatically complete your command.

  If you want to view and edit the code on HPC with an IDE/editor on your personal device(which is usually done by installing and using the command [`sshfs`](https://osxfuse.github.io)), make sure it could recognize the virtual environment you created on the directory at the **HPC**.

  For instance, if you are using [Visual Studio Code(VSC)](https://code.visualstudio.com)(Strongly recommended!), instead of simply using the [`sshfs`](https://osxfuse.github.io), you could install the extension`Remote - SSH`, open the terminal and input the command `Remote-SSH: Connect to Host...`(Not the command running on the terminal, but the `Show and Run Command`, found from the **SEARCH BAR** of VSC!). 

  Then, install the `Python` extension. From the search bar, use the command `Python: Create Environment...`, select `venv`. 

- Don't move/upload/download the virtual environment directory, because it is very easy to create, but there are countless files in the virtual environment. Moving/uploading/downloading it will be very slow!

- Install required package. Please input:

    `pip install --upgrade pip`

    `pip install matplotlib numpy pandas pathlib scipy tensorflow`

- `.gitignore` is the file to tell git that which file/directory they don't have to track their version. If you want to use `git` to track your version, please add `<name of your virtual environment>/` to a text file titled `.gitignore`, so that `git` won't sync this folder everytime you commit. Otherwise, this big file will make every commit process insanely slow!

    To know more about using `git` and [`Github`](https://github.com), feel free to ask [ChatGPT](https://chatgpt.com), [Claude](https://claude.ai) or other big language model! They are very capable of doing this.

### 3. Run

- **Everytime** you want to run the program on the **TERMINAL**, please input:

    `source <name of your virtual environment>/bin/activate` (You don't have to input this again if you have just input that previously)

    `python main.py`, and then follow its instruction. 
    
    When you finish running, please input:
    `deactivate` to close the virtual environment.

  If you are using IDE like [VSC](https://code.visualstudio.com), they will automatically complete this if you use its `Remote - SSH` extension.

- If you don't want to input the path of files to read everytime you run the program, you could also open the `main.py` and edit the code by yourself.

- The code being commented at `main.py` is proven to be working, for both NN model and CNN model. The UI version hasn't been tested yet, but you can know the logic of the code by viewing them.

- If you like to run the code step by step, you could also use [Jupyter Notebook](https://jupyter.org) to run the program. If you are using VSC, just install the extension for it and you can simple use it! The filename extension for Jupyter Notebook is `.ipynb`.

## How it works

Below explains what are roles of each `*.py` file and how they interact. To see more detail, feel free to look through the codes file and play around with them. There are comments in the code and I'll make my best to make them easy to understand.

1. `main.py`

    Only this file will interact with you. 

    The rest of the files are only storing functions to call at this file.

    - `main.ipynb` is an example of directly using the code to classify. This is proven to be working.

2. `get_directory_file`

    Getting paths of data files you want `PlumeCNN` to read.

3. `preliminary_processing`

    Get some basic informations for all the datas to read, including:
    - The direction of the slicing (if it exists).
    - The aspect ratio of the container (diameter/depth).
    - The range of variables
    - and more...

    They can simplify the further data processing.

4. `mapper.py`

    To simplify the analysis process in [VisIt](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.0/gui_manual/Intro/Installing_VisIt.html), where the grid may not be evenly spaced (with finer grids typically at the container's boundary), this file maps the original data to an evenly spaced table at a user-specified resolution.

5. `analyzer.py`

    Normalize data, calculate gradients of variable. It also stored function to calculate vorticity and the result of multiplication of two variables, although they are just used in experiments, not used in any current files. 

6. `neural_network.py` and  `convolutional_neural_network.py`

    Use [Neural Network (NN)](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) or [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) to predict whether a point is the boundry of heat plume. It could also do visualization on how the model improves on itself.

    Check [3Blue1Brown's series of videos](https://www.youtube.com/watch?v=aircAruvnKk) if you want to know how NN and CNN works. It's not very hard to understand!

    It is strongly recommended to use [Jupyter Notebook](https://jupyter.org) to test this part of code, so that you can run it part by part, to save a lot of time! On [VSC](https://code.visualstudio.com), you just need to install an extension to use it.

7. `visualizer.py`

    Read the result of analyzation and output image/movie to the directory user indicated.

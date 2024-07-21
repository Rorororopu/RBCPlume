# PlumeCNN

Identifying the boundry of heat plumes, using the data from direct numerical simulation of Rayleigh–Bénard convection(RBC) in CYLINDRICAL container.

## Required Softwares to Install

1. [Nek5000](https://nekx5000.mcs.anl.gov)

    The software used to solve partial differential equations to get the simulation result of RBC.

2. [VisIt](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.0/gui_manual/Intro/Installing_VisIt.html)

    The software used to visualize and rearrange the simulation data from `Nek5000`.

3. [Python](https://www.python.org/downloads/)

    The programming language being used. PlumeCNN support python version higher or equal to 3.6.

## Manual

- Run your simulation in CYLINDRICAL CONTAINER on `Nek5000`. Make sure you normalized data of temperature so that it is supposd to be in the range from 0-1. 

- Open the simulation result in `VisIt`

- In `VisIt`, Click `File - Export Database`, in `Xmdv` format. You can choose to use either comma or space to separate in the databse. **Remember to export the coordinate, and don't change their filename after you export the data!**

    Below are variables you **must** export, regardless of sequence:

  - Coordinates of your grids (An option after you click the button "Export")
  - scalars/temperature
  - scalars/time_derivative/conn_based/mesh_time (This is supposed to be used when outputing movie, but this function hasn't been developed yet, so it's okay to not export this varble now.)
  - scalars/velocity_magnitude
  - scalars/z_velocity
  
  Other variables will not be used. So it is not recommended to export other variables.
  Currently PlumeCNN only supports 3D database and database sliced perpendicular to x, y, or z axis.

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

    `pip install matplotlib numpy pandas pathlib scipy sklearn tensorflow`

- `.gitignore` is the file to tell git that which file/directory they don't have to track their version. If you want to use `git` to track your version, please add `<name of your virtual environment>/` to a text file titled `.gitignore`, so that `git` won't sync this folder everytime you commit. Otherwise, this big file will make every commit process insanely slow!

    To know more about using `git` and [`Github`](https://github.com), feel free to ask [ChatGPT](https://chatgpt.com), [Claude](https://claude.ai) or other big language model! They are very capable of doing this.

- **Everytime** you want to run the program on the **TERMINAL**, please input:

    `source <name of your virtual environment>/bin/activate` (You don't have to input this again if you have just input that previously)

    `python main.py`, if you want to classify your data using pre-trained model.
    `python main_train.py`, if you want to train your own model, or

    and follow its instruction. When you finish running, please input:

    `deactivate`

    to close the virtual environment.

  If you are using IDE like [VSC](https://code.visualstudio.com), they will automatically complete this if you use its `Remote - SSH` extension.

## Your Result

## How it works

Below explains what are roles of each `*.py` file and how they interact. To see more detail, feel free to look through the codes file and play around with them. There are comments in the code and I'll make my best to make them easy to understand.

1. `main.py`

    Only this file will interact with you. 

    The rest of the files are only storing functions to call at this file.

2. `get_directory_file`

    Getting a list of paths you want `PlumeCNN` to read.

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

    Read the result of analyzation and output image to the directory user indicated.

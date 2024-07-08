# PlumeCNN

Identifying the boundry of heat plumes, using the data from direct numerical simulation of Rayleigh–Bénard convection(RBC) in cylindrical container.

## Required Softwares to Install

1. [Nek5000](https://nek5000.mcs.anl.gov)

    The software used to solve partial differential equations to get the simulation result of RBC.

2. [VisIt](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.0/gui_manual/Intro/Installing_VisIt.html)

    The software used to visualize and rearrange the simulation data from `Nek5000`.

3. [Python](https://www.python.org/downloads/)

    The programming language being used. PlumeCNN support python version higher or equal to 3.6.

## Manual

- Run your simulation on `Nek5000`. Make sure you normalized data of temperature so that it is supposd to be in the range from 0-1. Currently only simulation in cylindrical container is tested.

- Open the simulation result in `VisIt`

- In `VisIt`, Click `File - Export Database`, in `Xmdv` format. You can choose to use either comma or space to separate in the databse. **Don't change their filename, and remember to export the coordinate!**

    Below are variables you **must** export:

  - Coordinates of your grids (An option after you click the button "Export")
  - scalars/temperature
  - scalars/time_derivative/conn_based/mesh_time
  - scalars/velocity_magnitude
  - scalars/z_velocity
  
  Other variables will not be used. So it is not recommended to export other variables.
  Currently PlumeCNN only supports 3D database and database sliced perpendicular to x, y, or z axis.

- Download this whole repository to the computer from [Github](https://github.com/Rorororopu/PlumeCNN).

  Or use the command `git clone <https://github.com/Rorororopu/PlumeCNN.git>' at the directory where you want to store this directory.

- use `sinfo` to check available nodes

- `ssh` to a node like `n001`

- `cd` to its `PlumeCNN` repository

- Create and activate a vitrual environment on the **TERMINAL** via **SSH**. If Don't input them on your computer using [`sshfs`](https://osxfuse.github.io), because your computer may be confused about your versions of python, your permission of files and some other issues. Please input:

    `python3 -m venv <name of your virtual environment>`

    `source <name of your virtual environment>/bin/activate`

    You could use the TAB key to let the terminal automatically complete your command.

    If you still want to view and edit the code on an IDE/editor on your personal device(which is usually done by installing and using the command [`sshfs`](https://osxfuse.github.io)), make sure it could read the virtual environment you created on the directory at the HPC.

    For instance, if you are using [Visual Studio Code(VSC)](https://code.visualstudio.com)(Strongly recommended!), instead of simply using the [`sshfs`](https://osxfuse.github.io), you could install the extension`Remote - SSH`, open the terminal and input the command `Remote-SSH: Connect to Host...`(Not the command running on the terminal, but the `Show and Run Command`, found from the **SEARCH BAR** of VSC!). 

    Then, install the `Python` extension. From the search bar, use the command `Python: Create Environment...`, select `venv`. Remember to add `.venv` to `.gitignore` file!

- Install required package. Please input:

    `pip install --upgrade pip`

    `pip install matplotlib numpy pandas pathlib scipy sklearn tensorflow`

- If you want to use `git` to track your version, please add `<name of your virtual environment>/` to a text file titled `.gitignore`, so that `git` won't sync this folder everytime you commit. Otherwise, this big file will make every commit process insanely slow!

    To know more about using `git` and [`Github`](https://github.com), feel free to ask [ChatGPT](https://chatgpt.com), Claude or other big language model! They are very capable of doing this.

- **Everytime** you want to run the program on the terminal, please input:

    `source <name of your virtual environment>/bin/activate`

    `python main.py`

    and follow its instruction. When you finish running, please input:

    `deactivate`

    to close the virtual environment.

    However, [VSC](https://code.visualstudio.com) will automatically complete this if you use its `Remote - SSH` extension.

## How it works

Below explains what are roles of each `*.py` file and how they interact.To see more detail, feel free to look through the codes file and play around with them. There are comments in the code and I'll make my best to make them easy to understand.

1. `main.py`

    Only this file will interact with you.

    The rest of the files are only storing functions to call at `main.py`.

2. `get_directory_file`

    Getting a list of paths you want `PlumeCNN` to read.

3. `preliminary_processing`

    Get some basic informations for all the datas to read, including:
    - The direction of the slicing (if it exists).
    - The aspect ratio of the container (diameter/depth).
    - The range of x, y, and z coordinates
    - and more...

    They can simplify the further data processing.

4. `mapper.py`

    To simplify the analysis process in [VisIt](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.0/gui_manual/Intro/Installing_VisIt.html), where the grid may not be evenly spaced (with finer grids typically at the container's boundary), this file maps the original data to an evenly spaced table at a user-specified resolution.

5. `analyzer.py`

    Normalize data, calculate necessary parameters: gradients and vorticity.

6. `model.py`

    Use [NN(Neural Network)](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) and [CNN(Convolutional Neural Network)](https://en.wikipedia.org/wiki/Convolutional_neural_network) to predict whether a point is the boundry of heat plume. It could also do visualization on how the model improves on itself.

    Check [3Blue1Brown's series of videos](https://www.youtube.com/watch?v=aircAruvnKk) if you want to know how NN and CNN works. It's not very hard to understand!

7. `visualizer.py`

    Read the result of analyzation and output picture to the directory we indicated.

# Green StableYolo: Optimizing Inference Time and Image Quality of Text2Image Generation

This repository contains the **source codes** and **raw experiment results** for the paper.

>**Abstract:** Tuning the parameters and prompts for improving AI-based image generation systems has been a substantial yet unaddressed challenge. Hence we introduce Green StableYolo, which improves the parameters and prompts for stability.ai’s deep learning Stable Diffusion using NSGA-II and Yolo to both reduce GPU inference time and increase quality of the generated images. Our experiments show that despite a relatively slight trade-off (18%) in image quality compared to the baseline StableYolo (which only considers image quality), Green StableYolo achieves a substantial reduction in inference time (266% less) and a 526% higher hypervolume, thereby advancing the state-of-the-art in multi-objective optimization for text-to-image.

# Documents
- **results**:
contains the raw experiment results for 15 runs with StableYolo and Green StableYolo.

- **GreenStableYolo.py**: 
the *main program* for using Green StableYolo.

- **requirements.txt**:
the necessary packages required to run the program.

# Prerequisites and Installation
1. Download all the files into the same folder/clone the repository.

2. Install the specified version of Python:
the codes have been tested with **Python 3.6 - 3.9**, other versions might cause errors.

3. Install all missing packages in **requirements.txt** or according to runtime messages.

# Run *Green StableYolo*

- **Command line**: cd to the folder with the codes, input the command below, and the rest of the processes will be fully automated.

        python GreenStableYolo.py
        
- **Python IDE (e.g. Pycharm)**: Open the *GreenStableYolo.py* file on the IDE, and simply click 'Run'.

# Change experiment setup

    Edit line 388 prompt = "Two people and a bus" to test with different prompts.

    Edit lines 390-400 to change the hyperparameters for NSGA-II.

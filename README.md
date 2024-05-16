# GreenStableYolo: Optimizing Inference Time and Image Quality of Text-to-Image Generation

This repository contains the **source codes** and **raw experiment results** for the paper.

>**Abstract:** Tuning the parameters and prompts for improving AI-based text-to-image generation has remained a substantial yet unaddressed challenge. Hence we introduce GreenStableYolo, which improves the parameters and prompts for stability.ai’s Stable Diffusion to both reduce GPU inference time and increase image generation quality using NSGA-II and Yolo. Our experiments show that despite a relatively slight trade-off (18%) in image quality compared to StableYolo (which only considers image quality), GreenStableYolo achieves a substantial reduction in inference time (266% less) and a 526% higher hypervolume, thereby advancing the state-of-the-art for text-to-image generation.

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

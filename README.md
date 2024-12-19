# Investigation of the Decomposition of Quantum Mechanical Energies of Methane into Body-Ordered Contributions 
This project investigate the contribution of the different body ordered components in making a machine-learning interatomic potential (MLIP). 
The implemented method are linear regression, ridge regression, kernel ridge regression and neural network. 

### Installation
1. Clone the repository 
```bash
git clone https://github.com/zoepevans/CS433-project2-mvz
```
2. Verify you have the required packages :

    - numpy
    - pandas
    - matplotlib
    - os
    - sklearn 
    - skmatter
    - torch

3. The data should be put on a folder named "data" parallel to the repository and the data named energy.np and features_xb.npy where x is the body ordered (2,3 or 4)

### Usage

This project includes several run_XXX.py scripts (run_Kernel.py, run_LinReg.py, ...) that perform the differents method implemented.
All the parameters of interest can be modified directly in thoses files. 
They are designed to be run independently using the following command :
```bash
python run_XXX.py
```
### Overview of the scripts

1. run_LinReg.py

This script performs linear regression on the dataset. It reads the data, preprocesses it, and fits a linear regression model. The script outputs the root mean squared error of the predictions divided by the std of the training target. You can choose one of the input data path and modify other parameters directly in the script.

2. run_Sampling.py

This script performs sampling of the dataset using CUR decomposition. It reads the data, applies the sampling method, and outputs the data with sampled features.  You can choose one of the input data path and modify other parameters directly in the script. If you uncomment the last part you can also sample on the data instead of just the features.

3. run_Kernel.py

This script performs kernel ridge regression on the dataset. It reads the data, preprocesses it, and fits a kernel ridge regression model. The script outputs the root mean squared error of the predictions divided by the std of the training target.  You can choose one of the input data path and modify other parameters directly in the script.

4. run_ridge.py

This script performs ridge regression on the dataset. It reads the data, preprocesses it, and fits a ridge regression model. The script outputs the root mean squared error of the predictions divided by the std of the training target.  You can choose one of the input data path and modify other parameters directly in the script.

5. run_NN.py

This script performs neural network regression on the dataset. It reads the data, preprocesses it, and fits a neural network model. The script outputs the root mean squared error of the predictions divided by the std of the training target.  You can choose one of the input data path and modify other parameters directly in the script.


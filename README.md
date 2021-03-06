# Bit-Error-Rate (BER) Predict using Neural Network
<img src="https://filebox.ece.vt.edu/~jbhuang/images/vt-logo.png" width="240" align="right">  


Aug, 2019    
Jet Yu, jianyuan@vt.edu  
Hussein Metwaly Saad, husseinm19@vt.edu     
Yue Xu, xuyue24@vt.edu    
Mike Buehrer, rbuehrer@vt.edu   
__Wireless, ECE, Virginia Tech__  
![](./MISC/map.jpeg)


# Dataset
under folder `DATA`, for more data, click [DataSet](https://drive.google.com/drive/folders/1SEYScWxg7xViXz1snsuuhOBDKR7r1Npt?usp=sharing), and `./DATA/READ.md` describe the data. 



# How to run
* if need, run `./FUNCTION/mainData.m` to generate dataset
* Neural Network
  * `main_regression.py`, where labels are in [0,1] scale  
  * `main_regression_dB.py`, where labels are in log scale, [0, 40]dB
  * `main_classification.py`, where labels are in another log scale
* Random Forest
  * `mainRF.m`, in matlab. 1-dimension
  * `mainRF.py`, in python

# File Description
* `ToolBox.py` is called by `main_classification.py`.  
* Under `FUNCTIONS` run `mainGenerateData.m` to generate traing dataset.  
* `Result` folder has  all trained results when run `main*.py` file
* `Predict` folder has  all test results when run `test.py`


  

# Chat/ Hangout Group
[Chat/ Hangout Group](https://chat.google.com/dm/5UaasgAAAAE)

# Slides
[performance Google slides](https://docs.google.com/presentation/d/1VWrGMLpcMN6-HCs_-lINsF1C2rGetQ_UcSqOe9gokTs/edit?usp=sharing)



# News
(Sep 29) starter code online

# Roadmap

# Config
running on ARC VT is prefered, run python is preferred.
 * Python
   * `tensorflow`, `Keras`, `scikitlearn`  








# Running Time Reference




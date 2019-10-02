# BerPredict

Bit-Error-Rate (BER) Predict using Neural Networkin Matlab  
Aug, 2019    
Jet Yu   
jianyuan@vt.edu    
Wireless, ECE, Virginia Tech 
![](./fig//H.png)


# Slides
[performance Google slides](TODO)


# News
(Sep 4) starter code online

# Roadmap

# Config
running on ARC VT is prefered. 
 * Matlab 
   * Machine Learning Toolbox  
   * GPU
* python
  * packages
      * `tensorflow`
      * `keras`
      * `matlibplot`
      * `hdf5storage`
      * `yagmail` - send result and codes to email
  * run `conda install conda-forge [package name]`
  * for yagmail, run `conda install -c atddatms yagmail`, if not work and error like `conflict with subproccess2`, try `pip install yagmail`  


# Generate interfer type  
* awgn (additive white gaussain noise)
* awgn+tone  
* awgn+chirp  
* awgn+filtN(filtered noise, low-passed white noise) 
* copyCat Noise, with unknown modulation and pulse shaping scheme.   

# How to run
* (Matlab) execute `mainData.m` file to generate training Data, then `main.m` to launch 
* (Python) run matlab file to generate data first, then `mainDNN.py` or `mainRNN.py`.

# How to run SVM in matlab
* download [libsvm](https://github.com/cjlin1/libsvm)  
* change directory to ./matlab
* execute `make.m` file, then new `.c` files are generated  
* load your `.mat` file, then 
  ```
  model = svmtrain(training_label_vector, training_instance_matrix, ['libsvm_options']);
  ```
* predict
  ```
  [predicted_label] = svmpredict(testing_label_vector,  ...
                                testing_instance_matrix, ...
                                model, ['libsvm_options']);
    ```

# Running Time Reference
for default datasize of 16,000, data generation cost ~ 1min, and training takes ~8min.

# Bibliography
[Bibliography](./todo)

# Code Reference
[libsvm - fast SVM](https://github.com/cjlin1/libsvm)




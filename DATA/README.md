# Dataset Description
## Exp1
  * `X16k1.mat`,  16k data, 5 inputs `[ bitsPerSym, JtoS, EbNo, IntfType,duty]`  
  * `Y16k1.mat`,  16k data, 2 inputs `[ unMitBer, MitBer]`   
  Notice: data has beed shuffled.
## Exp2 
  * `X.mat`,  7k data, 12 inputs ` [  bitsPerSym,              JtoS,          EbNo,          duty, isDSSS,                     IntfType,      intParams.fc,  intParams.SweepRate,intParams.bitsPerSym, intParams.sps, intParams.eBW, intParams.BW]`  
  * `Y.mat`,  7k data, 3 outputs `[ unMitBer, MitBer-FFT_Thre, MitBer-FFT_Notch]`   
## Exp3 
  * `X.mat`,  9k data, 13 input ` [  methodID, bitsPerSym,              JtoS,          EbNo,          duty, isDSSS,                     IntfType,      intParams.fc,  intParams.SweepRate,intParams.bitsPerSym, intParams.sps, intParams.eBW, intParams.BW]`
  * `Y.mat`,  9k data, 2 outputs `[ unMitBer, MitBer]`  



## Other Dataset Link
[Full DataSet](https://drive.google.com/drive/folders/1SEYScWxg7xViXz1snsuuhOBDKR7r1Npt?usp=sharing), for Old Dataset Description, click [DataSet](https://drive.google.com/drive/folders/1SEYScWxg7xViXz1snsuuhOBDKR7r1Npt?usp=sharing) for the `Old` folder if its not seen here.
* `X1.mat` `Y1.mat`, FFT_Threshold, `Ndat` 2000, `N` number of bits 1e4
* `X2.mat` `Y2.mat`, DSSS, `Ndat` 2000, `N` number of bits 1e4   
* `X3.mat` `Y3.mat`, Notch, `Ndat` 2000, `N` number of bits 1e4 
*  `X1_n.mat` `Y1_n.mat`, FFT_Threshold, AWGN only, `Ndat` approxi 400, `N` number of bits 1e4  
*  `XLong1.mat` `YLong1.mat`, FFT_Threshold, `Ndat` 2000, `N` number of bits __1e5__
*  `XLong3.mat` `YLong3.mat`, Notch, `Ndat` 2000, `N` number of bits __1e5__
*  `XScale1.mat` `YScale1.mat`, FFT_Threshold, `Ndat` __20000__, `N` number of bits 1e4






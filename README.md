# A-Novel-Broad-Echo-State-Network
## Abstract
Time series prediction is crucial for advanced control and management of complex systems, while the actual data are usually highly nonlinear and nonstationary. A novel broad echo state network is proposed herein for the prediction problem of complex time series data. Firstly, the framework of the broad echo state network with cascade of mapping nodes (CMBESN) is designed by embedding the echo state network units into the broad learning system. Secondly, the number of enhancement layer nodes of the CMBESN is determined by proposing an incremental algorithm. It can obtain the optimal network structure parameters. Meanwhile, an optimization method is proposed based on the nonstationary statistic metrics to determine the enhancement layer. Finally, experiments are conducted both on the simulated and actual datasets. The results show that the proposed CMBESN and its optimization have good prediction capability for nonstationary time series data
## Link to paper 
https://www.mdpi.com/2076-3417/12/13/6396
## Running code
CFBLS_ESN.py is a program to implement CMBESN model.  
```
CFBLS_ESN.py   
```
GRU_duolie.py is a program that uses GRU model to predict multi column datasets.  
```
GRU_duolie.py  
```
GRU.py is a program that uses the GRU model to predict single column datasets.  
```
GRU.py  
```
CFBLS_ESN_TEST.py and CFBLS_ESN.py two programs only modify some parameters, and the framework is the same.  
```
CFBLS_ESN_TEST.py  
CFBLS_ESN.py  
```
Other.py files are explained according to the above, that is, the model name has changed.  
## data set
```
date set.rar  is the data set used in this paper.  
```

# model-visualizer

use to visualize misclassified pictures by simple pass in your model.h5 and directory that contains the same number of classes you want to test on

### What you need 

python 3.x and you'll need to have these libraries install

```
tensorflow
keras
matplotlib
numpy
```
[]
### change your variable(s) at will

| variable | description |
| --- | --- |
| test_path(str) | your folder that will be use by flow_from_directory |
| model_path(str) | path of model.h5, will be called by model.load_model(model_path) |
| save_misclassified_path(str) | path that you want to save your misclassified pictures (will be create automatically if not exist) |
| minimum_size(int) | your model input size (e.g. 256) |
| fig_title(str) | figure name to be save |
| print_misclassified(bool) | print out misclassified pictures with true label and prediction on the terminal, default is True |



### how to run

simply call 

```
python3 visualizer.py 
```


### Example of print_misclassified 

![img](https://i.imgur.com/97eOiSB.png)

### Example image

![img](https://i.imgur.com/1Oej9Wu.jpg)

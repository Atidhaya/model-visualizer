# model-visualizer

use to visualize misclassified pictures by simple pass in your model.h5 and directory that contains the same number of classes you want to test on

### What you need 

python 3.x and you'll need to have these libraries install

```
tensorflow
keras
matplotlib
```

### change your variable(s) at will

| variable | description |
| --- | --- |
| test_path | your folder that will be use by flow_from_directory |
| save_misclassified_path | path that you want to save your misclassified pictures |
| minimum_size | your model input size (e.g. 256) |
| fig_title | figure name to be save |
| print_misclassified | print out misclassified pictures with true label and prediction, default is True |
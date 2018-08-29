import os
import numpy as np
import tensorflow as tf

from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.image as img
from os import mkdir


plt.switch_backend('agg')


'''
absolute variable should be declare here
'''

test_path = "/Users/doe/Desktop/validation"  #test or validation directory path call by generator, must have sub directory equal to number of classes. 
model_path = "/Users/doe/Desktop/game_classifier/model_weights/game_classifier_val_acc_94.h5" #your .h5 model path. 
save_misclassified_path = "./misclassified" #your target folder to save misclassified picture 
minimum_size = 256 #your model size 
mispred_pict = []
pred_actual = []
pred_wrong = [] 
confident_level = []
count = 0
batch_size = 1 #recommend to set to 1
tot_img = 0

fig_title = "misclassified_game"

page_size = 20      
rows = 4
cols = 5

print_misclassified = True 



#just typical inverse mapping because it's guarantee to have different indexes
def reverse_map(model_class):
    inv_class = {v: k for k, v in model_class.items()}
    return inv_class  

#return index of the array that contains the max value
#and corresponding label name 
def find_index(y):
    max_prob = 0
    max_index = -1 
    for i in range(len(y[0])):
        if y[0][i] > max_prob :
            max_index = i
            max_prob = y[0][i] 
    return max_index 



#check if the prediction match with the actual label. 
#returnboolean and index of the wrongly predicted label 
#else return true and index of the correct predicted label 
def match(y_true, y_predict):
  if(y_predict[0][find_index(y_true)] != max(y_predict[0])):
      return False
  return True


##load and resize image to your specific minimum_size if it's smaller.
def load_image(img_sub_path):
      file_path = img_sub_path
      img_arr = img.imread(file_path)
      resized = img_arr
      try : 
          w,h, _ = img_arr.shape 
          if w < minimum_size : 
              wpercent = (minimum_size/float(w))
              h_size = int((float(h)*float(wpercent)))
              resized  = imresize(img_arr, (minimum_size, hsize))
          elif h < minimum_size :
              hpercent = (minimum_size/float(h))
              wsize = int((float(w)*float(hpercent)))
              resized = imresize(img_arr, (wsize, minimum_size))
      except :
          print("bad image found ... skipping " , img_sub_path)
      return np.array(resized) 

def make_plt(rows, cols, start_i, fig_name, title=fig_title):
    fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(15, 25))
    fig.suptitle(fig_title, fontsize=20)
    for i in range(rows):
        for j in range(cols):
            if(start_i+(j+i*cols) > len(mispred_pict)-1):
                break 
            im = img.imread(test_path+"/"+ mispred_pict[start_i+(j+i*cols)])
            ax[i][j].imshow(im)
            ec = (0, .6, .1)
            fc = (0, .7, .2)
            ax[i][j].text(0, 0, 'Actual :' + pred_actual[start_i + (j+i*cols)] , size=10, rotation=0,
                  ha="left", va="top", 
                  bbox=dict(boxstyle="round", ec=ec, fc=fc))
            ec = (1, .5, .5)
            fc = (1, .8, .8)
            ax[i][j].text(0, 50, 'Predicted : ' + pred_wrong[start_i + (j+i*cols) ] +" ("+str(confident_level[start_i + (j+i*cols) ])+")", size=10, rotation=0,
                  ha="left", va="top", 
                  bbox=dict(boxstyle="round", ec=ec, fc=fc))
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    if not os.path.exists(save_misclassified_path):
        os.makedirs(save_misclassified_path)
    print("saving..." , fig_name)
    fig.savefig(save_misclassified_path+"/"+fig_name)


def visualize(page_size,rows,col):
    for i in range((len(mispred_pict)//page_size)+1):
        start_i = i*page_size 
        fig_name = fig_title+str(start_i)+".png"
        make_plt(rows=rows,cols=cols,start_i=start_i,fig_name=fig_name)


def execute(model_path=model_path, test_path=test_path,print_misclassified=print_misclassified, batch_size=batch_size):
    global count
    model = models.load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(minimum_size,minimum_size),
        batch_size=batch_size,
        shuffle=False,
        class_mode="categorical")
    print(test_generator.class_indices)
    true_map = reverse_map(test_generator.class_indices)  
    ##iterate through the whole test or validation set, extract necessary data
    global tot_img
    tot_img = len(test_generator)
    for i in range(len(test_generator)):
        x,y = next(test_generator)
        predict = model.predict(x) 
        predicted_index = find_index(predict)
        is_match = match(y,predict)
        if not is_match:
            #print out mismatch label along with confidence level 
            if(print_misclassified):
                print(test_generator.filenames[i],"||", "should be [", true_map[find_index(y)], "] but [",true_map[predicted_index], "(",predict[0][predicted_index],")]")
            mispred_pict.append(test_generator.filenames[i])
            pred_actual.append(true_map[find_index(y)])
            pred_wrong.append(true_map[predicted_index])
            confident_level.append(predict[0][predicted_index]) 
            count += 1
    visualize(page_size,rows,cols)  


if __name__ == '__main__':
    execute()
    print("total misclassified :", count, "(",1-count/tot_img,"%)")

[label_01]: /imgs/metrics/label_hist.png "Labels"
[label_02]: /imgs/metrics/label_hist_oversamp.png "Labels"
[img_01]: /imgs/metrics/dataset_images.png "Labels"
[img_02]: /imgs/metrics/dataset_images_02.png "Labels"

# Traffic light detection

The model is based on the results given in the Traffic Sign Project. I used a LeeNet 5 as a base model.
The preprocessing includes the same strategy, to oversample the dataset to a specific amount of data. 

At first i screenshot some images of red, gree, yellow and unknown samples from the simulator. 
Additionally i added more images to a limit of 100 for each class by rotating and translating the images in the dataset.

Additional data argumentation were:

* Normalize the image
* Rescale the image to 180,180
* Center the image to 0 by subtracting 0.5 from the normalized image

The model was trained over 10 epochs, because in a run over 20 epochs the model overfits.
The model was saved as a *,h5 kears model, loaded in the simulation with same preprocessing steps like in the training phase.

Code for the ml traffic light detection net can be found in `CarND-Capstone\tf_tl_classifier`

## Trainingresults

    Epoch 1/10
    400/400 [==============================] - 22s - loss: 1.1264 - acc: 0.6500    
    Epoch 2/10
    400/400 [==============================] - 9s - loss: 0.5072 - acc: 0.8100     
    Epoch 3/10
    400/400 [==============================] - 9s - loss: 0.3808 - acc: 0.8100     
    Epoch 4/10
    400/400 [==============================] - 9s - loss: 0.3122 - acc: 0.8650     
    Epoch 5/10
    400/400 [==============================] - 9s - loss: 0.2807 - acc: 0.8950     
    Epoch 6/10
    400/400 [==============================] - 9s - loss: 0.2612 - acc: 0.8950     
    Epoch 7/10
    400/400 [==============================] - 9s - loss: 0.2453 - acc: 0.8925     
    Epoch 8/10
    400/400 [==============================] - 9s - loss: 0.2281 - acc: 0.8950     
    Epoch 9/10
    400/400 [==============================] - 10s - loss: 0.2403 - acc: 0.8925    
    Epoch 10/10
    400/400 [==============================] - 10s - loss: 0.2236 - acc: 0.8900    

## Images per Label:

Label 

    0 : Red 
    1 : Green 
    2 : Unknown
    3 : Yellow

### Example images from directory

![alt text][label_01]

![alt text][img_01]

### After oversampling step 

![alt text][label_02]

![alt text][img_02]
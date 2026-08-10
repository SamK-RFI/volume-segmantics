# VolSeg Documentation; Settings Guide

The following guide gives a comprehensive overview of the possible settings, parameters and availible function variables offered within the Volume-Segmantics Toolkit. This section should be used in conjunction with the [Functionality Documentation](TBC), where modes can be swtiched using the variables listed in the Training and Prediction .YAML files. This document is split into 3 parts for easier navigation including;

- Default Parameter Walkthrough.
- Training and Prediction Settings File Breakdown and Fine-Tuning Guide.
- Model Archetecture, Encoder and Loss Function lists.

## Default Parameter Walkthrough

After downlading the VolSeg Tookit, the training_settings.YAML and prediciton_setting.YAML will be set to their default variable inputs. If significant changes are made to them during the course of your work and you require a fresh start, default version of these settings can be copied from [here](https://github.com/rosalindfranklininstitute/volume-segmantics/blob/main/volseg-settings).

For first time users, it is reccomended to familierise yourselves with the most important inputs of each settings file as outlined in this sub-section, and how changing them in your initial tests might affect your model training and output success. From there, the fine-tuning section can then be used to further change inputs according to your user-case. 

### Training Settings YAML

These setting directly influence the model training command of the VolSeg Toolkit. These are located in the `2d_model_predict_settings.yaml` file, where the individual inputs have comments as to their use next to their specific components; these are placed as reminders for their use and options. Further clarification for some of the more prominent and important settings and their possible changes are outlined below alongside their basic use (importance order);

- **`model:type`**; The designated Model Archetecture (overall training pathway) for your training; a full list of the potential archetectures can be found listed further on in this documentation. Prominent models include "U_Net" (default and most well rounded; it is reccomended that you test using this archetecture first), "U_Net_Plus_plus", "FPN", "DeepLabV3", "DeepLabV3_Plus" etc. though some may be more suitible depending on your user case.

- **`model:encoder_name`**; The designated Encoder used within your archetectures (feature reading and extraction); a full list of the potential encoders can be found listed further on in this documentation. The most tested and stable encoders include "tu-convnextv2_base", "tu_convnext_large", "resnet34" and "resnet50", though there has also been great success with the "efficientnet-b3", "timm-resnest50d"* and DINO (e.g. "dinov2_vitb14") encoders etc.

- **`image_size`**; The size of the data patches fed into the training command. This metric deals with how your data is stored and controlled, where variables must be a multiple of 32. The default input of *512*, and works great for image-label pairs between 250-700 cubic-voxels with efficient training times and moderate compute levels. If your compute capability is lower, or your data volumes are smaller, consider reducing this number (384 or 448). If you have a higher compute capability or your dataset is much larger, consider increasing this number (576 or 640); do take intoaccount that increaseing this variable will increase the training time and also have an impact on your predicttion time aswell. 

- **`num_cyc_frozen`**/**`num_cyc_unfrozen`**; The number of runs you designate your data to train on frozen and unfrozen parameters; combined equals the length of your training session. When frozen, the 'backbone' of the model is trained, creating a 'within-task pre-trained model' from specific parameters while others are frozen in place. When unfrozen, no parameters are halted and the pre-existing 'within-task pre-trained model' is then fine-tuned using all availible information based on the original. For further information regaurding this methodology, look for 'transfer-learning' subject matter. The higher the number of epochs, the more training time the model will incorporate to reach its peek validation and evaluation conditions; though be careful of *over-prediction impacts*. e.g. When looking at the *model_loss-Plot.png* of a very successful model, the 2 plot-lines should be reduce towards a straight line at the x axis, however if the graph starts to rise again; this will mean the model is over-training and you will receive a less accurate prediction as a result and this can be visible on predicition outputs as noise, and over segmented areas. **The number of unfrozen epochs should be 2/3 the size of the frozen epochs; the default is 8/5**.

- **`loss_criterion`**; The loss calculator helping to evaluate the training of noted features through classifications and inform and guide sucessful continued model progression; a full list of loss functions can be found further on in this documentation. Depending on your user case, this should be changes per your wanted segmentation outcomes and the image-label inputs you wish to train; "CombinedCEDiceLoss" is the most adaptable and stable for validation a wide range of images however great success has also bee seen using "BCELoss", "DiceLoss", "GeneralizedDiceLoss", etc.

- **`ce_weight`**/**`dice_weight`**; Weighting for CombinedCEDiceLoss Loss critrion evaluation; ratio of the cross-entropy component (ce), helping stabalise imbalance over the segmentable classes, and diceloss component (dice), calculating the overlap between the predicted outcomes and groundtruth. This ratio must add to 1.0 or the trainign will fail under this loss function. The variables should only be changes if the image-label pair inputs are particularly large, 1000+ cubic voxels, or the wanted features are numerous or complex in nature. The default is 0.2/0.8 split, however can be moved to more equal terms to suit the user case.

- **`eval_metric`**; Evaluation metric analysing errors in segmentation model; this should be kept as MeanIoU unless your user case requires specific amendations based on its complexity or size. 

When adapting training or prediction settings, start by varying these first based on your segmentation outcomes. 

#### 2d_model_predict_settings.yaml

The settings within this file have comments as to their use next to their specific component, however, for further clarification for some of the more prominent settings and their possible changes are outlined below *(Normalisation, Augmentation, and 2.5D settings should be kept the same as those used in the train_settings.yaml file)*;

> - **`quality`**; Degree of prediction-image analysis and prediction view (Medium is default, predicting using all 3 axis, however this can be lowered or improved based on the linked GPU capability or required output needs). 

low" quality refers to prediction of the volume segmentation by taking images along a single axis (images in the (x,y) plane). For "medium" and "high" quality, predictions are done along 3 axes and in 12 directions (3 axes, 4 rotations) respectively, before being combined by maximum probability. 

> - **`output_probs`**; tbc
> - **`output_entropy`**; tbc

> [!NOTE]
>*Further information regarding settings specifics, component options and choice is summarised in the [Advanced Usage and Functionality Documentation.](TBC)*

## Training and Prediction Settings File Breakdown and Fine-Tuning Guide
### File Breakdown

The training and prediction *.yaml* files are organised for easy navigation while amending training parameters. The training file is further split into 2 sections; *Basic* and *Advanced*, with the following sub-sections;

> #### Basic
> - Image and Model output
> - Normalisation
> - Reproducibility
> - Model Architecture
> - Learning rate 
> - Finder
>   - Differential
>   - Scheduler
> - Loss Function
>   - Selection
>   - Legacy inputs
>   - Advanced Losses
> - Evaluation Metric
> - Misc

> #### Advanced
> - Augmentation
> - 2.5D slicing
> - Multi-task
> - Semi-supervised (SS)
> - Pseudo-labeling SS

The prediction file is comprised of either 'continuating' variables (ones that should be mirrored from the training settings per the model utilised in the prediction command) or prediction specific inputs, with the following sub-sections;

> - Prediction Specifics
> - Normalisation
> - Augmentation
> - 2.5D Prediction
> - Sliding Window

## Fine-Tuning Guides

The following guide outlines what each input does, the possibilities for changing that settings and the potential affects of each settings line; it aims to inform users of usage prospects relative to their individual work case and help format your training/prediction parameters specifically towards you data.

### 2d_model_train_settings.yaml
#### - Basic;
> Image and Model Output; File allocation settings
- `data_im_dirname`: *Name of folder that sliced data 2D images will be output to*  . 
  > - The data is sliced and saved in this directory. This is temporarily stored in the VolSeg working directory while training is executed, and then deleted as the training command has completed. This should not be changed
- `seg_im_out_dirname`: *Name of folder that sliced segmentation 2D images with be output to* 
  > - The segmentation is sliced and saved in this directory. This is temporarily stored in the VolSeg working directory while training is executed, and then deleted as the training command has completed. This should not be changed
- `model_output_fn`: *Suffix for the saved model filename*
  > - The model outfile file (containing your trained model) will have this suffix in the name. this can be changed to suit your prefered output name for project clarity (must contain no spaces).
- `data_hdf5_path`: *The internal HDF5 path to the image data*
  > - Internal paths for data allocation when training. This should not be changed
- `seg_hdf5_path`: *The internal HDF5 path to the label data*
  > - Internal paths for segmentation allocation when training. This should not be changed

> Normalisation Settings; ###
- `clip_data`: *Clip and rescale the image data intensities before saving to disk*
  > - ?
- `st_dev_factor`: *The number of standard deviations from the mean to clip data to.*
  > - ?
- `minmax_norm`: *Min-max normalisation/rescaling. Only needed if clip_data if False.*
  > - ?
- `use_imagenet_norm`: *If True, applies ImageNet mean/std normalisation to input images (wanted if Imagenet pretrained weights used)*
  > - ?
- `normalization_debug_mode`: *If True, enables detailed normalisation debugging in training loop.*
  > - ?

> Reproducability; Random Seed Input for deterministic training and testing
- `random_seed`: Optional integer seed.
  > - Consistent-condition integer input; When set to a number/integer the training will run under deterministic parameters relative to a 'seed', meaning that the independant and randomly generated variables will be set in place including the train/validation split, shuffling and data-loading workers reproducable. This is an ideal feature for those testing specifically the archetecture or endocer habits with their data with stable training conditions. If set to zero, the seed will be ge generated randomly. if your model output is the main focus of a run, this should be left blank, however if the archetecture or encoder behavour relative to other statistics (for testing purposes) is the main focus, choose an integer and keep this consistent to produce a seed-generated outcomes.

> Model Training Inputs; Overview of training settings
- `training_axes`: *Specify axes/single axis to train on. Choose from [All, Z, X, Y].*
  > - The axis specified for training perspective. This should be kept to 'All', meaning the model takes into account all 3 axis directions when training, however can be changed; specific direction may be needed for highlighting features or mitigate artefacts, however may also change the way the model views other comparable images. This should be used in the intermediate or testing stages of model creation.
- `image_size`: *Size of images used for training (must be multiple of 32)*
  > - This refers to the size of the data/segmentation cubes used in the training command and how it is then sliced and handled when training begins; this number must be less than or equal to the overall image/seg file voxel size (ideally a lowest multiple up until your data size or within 2/3 of its original size) and can be reduced and increased to improve model data-handling depending on image size though larger numbers used in this variable will require a higher compute load. If the image is very complex, it is also suggested that you decrease this number to allow for more training cubes to be used.
- `downsample`: *If True, data will be downsampled by 2*
  > - The data is downsized if set to true. This should only really be used if working with very large datasets for testing; data should be kept whole for best models. 
- `training_set_proportion`: *Proportion of images to use the training, rest are used for validation*
  > - The ratio used for training verses validation (15% validation is default, 85% for training) mesured as a decimal; 1.00 is 100%. Chaning this ratio may affect the amount of data availible to the training; higher validation may mean that complex scan training may produce more varied results though varying slightly depending on user case can produce better models depending on wanted outcome. 
- `cuda_device`: *The graphics card to use (between 0 and 3 for a machine with 4 GPUs)*
  > - The graphics card designation if your computing setup has more than 1 (max 4); 0 is for 1 and first GPU.
- `num_cyc_frozen`: *Number of training epochs on frozen model*
  > - The nummber of frozen training cycles (Epochs), increase/decrease per prefered training periods wanted (the ratio should be 2/3 frozen, 1/3 unfrozen); Increasing this might improve a model with incomplete training epoch leveling, however this will also increase your training time and may require more computing power.
- `num_cyc_unfrozen`: *Number of training epochs on unfrozen model*
  > - The nummber of frozen training cycles (Epochs), increase/decrease per prefered training periods wanted (the ratio should be 2/3 frozen, 1/3 unfrozen); Increasing this might improve a model with incomplete training epoch leveling, however this will also increase your training time and may require more computing power.
- `patience`: *Number of epochs to wait before early stopping if validation loss does not improve*
  > - The number of epochs the training protocol will wait if the validation score does not improve past a specific point. This will decrease the chance for training to start degrading with continued training time, though should be altered in conjunction with the epoch numbers if the data is complex. 

> Model Architecture Settings; Workflow of data-handling/processing used within the model trainer.
- *`model: Type`*: Choice of segmentation model type; 11 possible. U_Net is default and most stable (use Multitask_Unet for multitask advanced usage only)
  > - The Model Archeteture you wish to use for your training run; this can be from the range listed in the settings comment and must be exactly as described or an error will be given. *Unet* is the most diverse and effective on a wide range of images, *Unet++* is another alternative that may prove more effective on more complex label boundaries. *Multitask Unet* should only be used if the multitask functionality is being used. 
- *`model: encoder_name`*: Choice of encoder name; 19 possible. tu-convnextv2_base is default and most stable. 
  > - The Model Encoder you wish to use for your training run; this can be from the range listed in the settings comment and must be exactly as described or an error will be given. *tu-convnextv2_large* is good for larger images, *resnet* can give better generalisations and performance on low contrast images, *other encoders* can prove effective when testing to corrct specific image features. 
- *`model: encoder_weights`*: Pre-trained weights asignment. 
  > - The default pretrained weights based on libriaries within model archetectes; compatible with all encoders availible and needed for initialization. 
- *`model: encoder_depth`*: Number of feature levels trained as part of the encoder
  > - When training, the model archetecture downsamples the data to extract features; this depth limits the number of levels for the downsampling wher ehgiher steps mean more espatical resolution, and lower steps means faster training.The default is 5 and is the best for efficiency hwoever can be increaqsed uf more compute is availible or exact model is wanted. DINO encoders need should use less levels or the training may become very complex. 
Commented out: `_in_channels`: Data layers for the model input tensor
  > - Asigns the number of feature maps to expect; this is automatically asigned as per the data detection features, however can be commented out and asigned mannually; 1 for grayscale, 3 for RGB, 5 for multi-scale or multi-modal data and N for 2.5D

> Learning Rate Settings; Settings for finding the learning rate: the speed at which the model learns and makes predictions, where it finds a optimal balance betwen speed and accuracy. 
- *`lr_find_epochs`*: Number of training epochs for learning rate search
  > - Number of epochs used to automatically find the numerical learning rate through a *test training*; finding the steepest decrease of loss through a gradual increase of the learning rate ensuring the model is training efficiently. This should be kept to one as hgiher valuse make the search slower with little benefit, though can be increased if needed.
- *`lr_reduce_factor`*: Divisor for start and end LR when finding LR on reloaded model
  > - This controls the optimization of the learning rate behaviour during training; by *decreaseing* this number for *larger encoders*, it allows the learning rate to increase. This should be used in conjunction with the schelduler and multiplier settings to acheive better results. Default is 500 (medium sized encoder size, decrese per smaller encdoer size).
- *`encoder_lr_multiplier`*: Encoder LR multiplier for unfrozen training
  > - Encoders scaler for the learning rate search with respect to the decoder. A smaller number leads to slower updates to the encoder and more suited for larger encoders, and larger numbers are used for smaller encoders and faster updated to the encoder. If set to 1.0 or none, the learning rate will be the same for the encoder and decoder. This should be used in conjunction with the schelduler and reduce_factor settings to acheive better results. Default is 0.1 (larger encoder settings, increse per smaller encoder size).

> Learning Rate Scheduler; Settings directly tuning learning rate as training starts. 
- *`pct_lr_inc`*: The percentage of overall iterations where the LR is increasing.
  > - The number of total training steps spent increasing the learning rate from minimum ro maximum; controla how quickly the learning rate ramps up at the start of the training. Small values (0.05 min.)mean its peak is reached quickly leading to potential model instability, and larger numbers (0.4 max.) can stabalise training of larger models and those which data is particularly noisey. Number outside this range are not reccomended. Default is 0.3 (30% of steps used for warm up)
- *`starting_lr`*: Lower bound of learning rate search
  > - This is the initial learning rate value from whihc the scheduling or decay starts from; as the training starts this will adjust over time to suit the training conditions. a higher starting rate can speed up convergence but risks overshooting the minimum rate required, however a lower learning rate, though proving to be more stabkle does increse the trainign time substantially. *5e-5* is the default though can be adjusted per the user case. 
- *`end_lr`*: Upper Bound of learning rate search
  > - This is the final learning rate at which training will cease, where the increase from its the *starting_lr* is and its mesured loss is monitoured to find the optimal learning rate range. A final smaller ending rate should finish the training with a low learnign rate, which is better for fine-tuning, however a larger ending rate means the learning rate will remain high at the end helping to avoid overfitting but may result in reduces model stability. *1e-3* is the default though can be adjusted per the user case. 

> Loss Function Settings; Choice of criterion and relative variables
- *`loss_criterion`*: Choose from one of the 9 loss functions listed
  > - The Loss Function you wish to use for your training run; this can be from the range listed in the settings comment and must be exactly as described or an error will be given. *CombinedCEDiceLoss* is the default and works very well on single, multiclass data and for larger prediction images. Other functions may work better for more complex images, or with specific user case features. 
- *`ce_weight`*: **Used specifically for CombinedCEDiceLoss**; Weight for Cross-Entropy component (alpha in CombinedCEDiceLoss)
  > - Used in conjuction with dice_weight to form a ratio; contol loss function control depending on user case. CE; pixel-wise classification accuracy: higher number better for lower classes or equal class weight in dataset, lower for multiclass or more complex images. 
- *`dice_weight`*: **Used specifically for CombinedCEDiceLoss**; Weight for Dice component (beta in CombinedCEDiceLoss)
  > - Used in conjuction with ce_weight to form a ratio; contol loss function control depending on user case. Dice; class imbalence: higher number better for more complex and datasets with high class variability or number, lower for less lower multiclass complexities.
- *`dice_weight_mode`*: Dice weighting mode for class-weighted losses
  > - The weight normalisation for smaller classses in multi-class training; *inverse_sqrt_freq* is the default and the best for more complex and multiclass data increasing the importance of the smaller classes with respect to the larger ones, *inverse_freq"* priotirises the smaller classes ove the largr ones, and *uniform* keeps the smaller and largr classes in an unbalenced dataset equal in terms of importance. 
- *`exclude_background_from_dice`*: Flag for the inclusion of the backgroudn class in metrics. 
  > - This should be set to *True* when the background forms a large portion of the image compared to your wanted labels (roughly 80% background or more depending on image data and complexity) and helps to focus the model on the segmentable features; helps to predict accurately and reduce noise.
- *`alpha`*: When BCEDiceLoss selected, weighting for BCELoss
  > - **Used specifically for BCEDiceLoss**; prefered metric before being depreciated for CombinedCEDiceLoss. ??
- *`beta`*: When BCEDiceLoss selected, weighting for DiceLoss
  > - **Used specifically for BCEDiceLoss**; prefered metric before being depreciated for CombinedCEDiceLoss. ??

> Evaluation Metric; Performance and Validation Assesment 
- *`eval_metric`*: The Evaluation Metric used throughout training for model accuracy and validation
  > - *Mean Intersection over Union (MeanIoU)*; the mean value of the calculated IoU for each class, taking into consideration the true posatves fale posatives and false negatives from the predcition relative to the ground truth. It gives a stable and accurate value from which to interpret overlap and take into consideration class imbalance; prefect segmentation is 1.0 (100%). This is the default and the most widely used metric. 
  > - *DiceCoefficient*; Calculates a dicescore relative to the overlap between groundtruth and prediction, and better imbalanced class distributions; however cannot be used in conjuction with loss functions that include DiceLoss (CombinedCEDiceLoss, DiceLoss, GeneralizedDiceLoss and ClassWeightedDiceLoss) as it introduced biases. 
- *`dice_averaging`*: ??
  > - **Used specifically for DiceCoefficient**; Macro averages the dicescore per class to take in overall model performance. ??

> Misc. Training Settings; ###
- *`plot_lr_graph`*: ??
  > - 
- *`use_sam`*: ??
  > - 
- *`adaptive_sam`*: ??
  > - 
- *`encoder_weights_path`*: ??
  > - 
- *`full_weights_path`*: ??
  > - 

#### - Advanced;
> Augmentation Settings; Image-Label Data Training Library
- `augmentation_library`: Choose augmentation library.
  > - Assign the augmentation libruary to your training depending on user case; this is further explained in the [functionality documentation]().
- `use_monai_datasets`: Flag for specific MONAI datasets
  > - When MONAI is selected as the augmentation_library, you can choose to use its datasets alongside others intergeted into the augmentation libruaries; this can sometimes improve the segmentation of medical images. This should be set to tru when MONAI augmentation libruary is selected (will have no bearing on albumentations if selected and remains true).

> 2.5D Slicing Settings; 2.5D functionality.
- `use_2_5d_slicing`: Enables 2.5D functionality.
  > - Set to false when not using; must mirror prediction settings if model using 2.5D settings is created. Set to true when in use; will use num_slices and slice_file_format as further flags.
- `num_slices`: Number of slices to use.
  > - Choice of the number of slices that shoudl be separated before selecting a central training slice; must be an odd number (3/5/7/9 etc.). This is further explained in the [functionality documentation]().
- `slice_file_format`: File format for multi-channel slices
  > - The file format used when processing slices before training starts. Use .png for 3 slices mode (3 channels) or .tiff for 5 slices or above (over 3 channels).
- `skip_border_slices`: Flag to skip start and end slice from slice selection. 
  > - If enabled, will ski the first and last slice when choosing enacting the num_slices choice. should be kept to false as this function is not yet fully implimented; still being tested

> Multi-task Learning settings; Multi-Task Functionality; boundary maps
- `use_multitask`: Enables Multitask functionality.
  > - Set to false when not using and set to true when in use; will use subsequant function flags.
- `num_tasks`: Number of tasks
  > - Number of tasks integrated into the multitask designator; includes 0 as image and 1 as label files, and task 2 as the bounary file. This is further explained in the [functionality documentation](). 
- `decoder_sharing`: ??
  > - 
- `seg_loss_weight`: ??
  > - 
- `boundary_loss_weight`: ??
  > - 
- `task3_loss_weight`: ??
  > - 
- `boundary_loss_type`: ??
  > - 

> Semi-Supervised Learning Settings; Self-Supervised Learning Functionality; mean-teacher and pseudolabeling
- `use_semi_supervised`: Enables Self-supervised functionality.
  > - Set to false when not using and set to true when in use; will use subsequant function flags.
- `unlabeled_batch_size`: ??
  > - 
- `consistency_weight`: ??
  > - 
- `rampup_start`: ??
  > - 
- `rampup_end`: ??
  > - 
- `ema_decay`: ??
  > - 
- `mean_teacher_vis_epoch_interval`: ??

> Pseudo-labeling Semi-Supervised Learning Settings; ??
- `pseudo_label_confidence_threshold`: ??
  > - 
- `pseudo_label_confidence_method`: ??
  > - 
- `pseudo_label_min_pixels_per_class`: ??
  > - 
- `pseudo_label_use_teacher`: ??
  > - 
- `pseudo_label_weight`: ??
  > - 
- `pseudo_label_rampup_start`: ??
  > - 
- `pseudo_label_rampup_end`: ??
  > - 
- `pseudo_label_threshold_schedule`: ??
  > - 
- `pseudo_label_start_threshold`: ??
  > - 
- `pseudo_label_target_acceptance_rate`: ??
  > - 
- `pseudo_labeling_vis_epoch_interval`: ??
  > - 

~Finish after explaination/walkthrough given

### 2d_model_predict_settings.yaml
> Prediction Settings: Fundimental prediction settings
- `quality`: Degree of Prediction
  > - Quality can be set to 3 setting depending on need and compute; Low: predicts along a single axis (Specific-axis in 4 directions) giving a 2D prespective prediction from your model along a specified axis, Medium: predicts along 3 axis (Z/Y/X-axis) giving a well rounded 3D perspective prediction from your model. High: Predicts along 3 axis in 4 directions per dimension (Z/Y/X-axis in 4 directions) giving a higher quality '12-way' 3D perspective prediction from your model along all axis. Setting the quality to high will require a larger computeing threshold and take 3X longer than a notmal prediction. 
- `output_probs`: ??
  > - 
- `output_entropy`: ??
  > - 
- `cuda_device`: *The graphics card to use (between 0 and 3 for a machine with 4 GPUs)*
  > - The graphics card designation if your computing setup has more than 1 (max 4); 0 is for 1 and first GPU.
- `downsample`: *If True, data will be downsampled by 2*
  > - The predictio image data is downsized if set to true. This should only really be used if working with very large datasets for testing; data should be kept whole for best predictions. 
- `one_hot`: ??
  > - 
- `prediction_axis`: **Used for *Low Quality* prediction settings**. 
  > - Set the axis for the low quality prediction setting; can be Z, Y or X axis. 
- `output_size`: *Size of images used for prediction (must be multiple of 32)*
  > - This refers to the size of the data cubes used in the prediction command;  it should be less than the overall image file voxel size (ideally the lowest multiple up until your data size) and can be reduced and increased to improve model depending on image complexity though larger numbers used in this variable will require a higher compute load. this number can be different than that used int he model training settings used for the predict command. 
- `data_hdf5_path`: *The internal HDF5 path to the image data*
  > - Internal paths for data allocation when predicting. This should not be changed

> Sliding Window Interface; Entropy Map Variables??
- `use_sliding_window`: ??
  > - 
- `sw_roi_size`: ??
  > - 
- `sw_overlap`: ??
  > - 
- `sw_batch_size`: ??
  > - 
- `sw_mode`: ??
  > - 

> Mirror settings; Normalisation settings
- `clip_data`: ??
  > - 
- `st_dev_factor`: ??
  > - 
- `minmax_norm`: ??
  > - 
- `use_imagenet_norm`: ??
  > - 

~ check that the prediction normalisation settings match those used for training

> Mirror settings; Augmentation settings
- `augmentation_library`: Choose augmentation library.
  > - Assign the augmentation libruary to your prediction command depending on user case; this is further explained in the [functionality documentation](), though should be the same libruary used for the training model used in the prediction command. 

> Mirror settings; 2.5D Prediction settings
- `use_2_5d_prediction`: Enables 2.5D functionality.
  > - Set to false when not using; must mirror training settings if model using 2.5D settings during training is used int he prediction command. Set to true when in use; will use num_slices and prediction_padding_factor as further flags.
- `num_slices`: Number of slices to use.
  > - Choice of the number of slices that shoudl be separated before selecting a central prediction slice; must be an odd number (3/5/7/9 etc.). This is further explained in the [functionality documentation](), though should be the same number used for the training model used in the prediction command. 
- `prediction_padding_factor`: ??
  > - 

~Finish after explaination/walkthrough given

As newer versions of the toolkit are released, further settings may be added and as such this documentation guide will be updated.

## Model Architectures, Encoder and Loss Function Lists; 

There are 10 current model architectures compatible with the VolSeg Toolkit. These can be changed ion the Training Setting YAML where the availible options are listed in the comments of the *Model Architecture* input; `U-Net` is the default and the most widely-tested architecture. The full architecture list includes:

- `U-Net` (Default)
> des 
- `U-Net++`
> des
- `FPN`
> des
- `DeepLabV3`
> des
- `DeepLabV3+`
> des
- `MA-Net`
> des
- `LinkNet`
> des
- `PAN`
> des
- `SegFormer`
> des
- `Vanilla U-net`
> des

There are also 19 pre-trained encoders that can be used with these architectures: `tu-convnextv2_base` is the default and the most widely tested encoder. The full pretrained encoder list includes:

- `tu-convnextv2_base` (Default)
> des
- `tu_convnext_base`
> des
- `tu-convnextv2_large`
> des
- `tu-convnext_large`
> des
- `resNet34`
> des
- `resNet50`
> des
- `resnext50_32x4d`
> des
- `Efficientnet-b3`
> des
- `Efficientnet-b4`
> des
- `Efficientnet-b5`
> des
- `Efficientnet-b7`
> des
- `timm-resnest50d`\*
> des
- `timm-resnest101e`\*
> des
- `dinov2_`..
  - ..`vitx14`
  > des
  - ..`vitb14`
  > des
  - ..`vitl14`
  > des
  - ..`vitg14`
  > des
- dinov3_`..
  - ..`vitl16`
  > des
  - ..`vit7b16`
  > des

>Encoders with an asterisk (\*) are not compatible with PAN.

There are finally 9 Loss Functions that can be used when training these models: `CombinedCEDiceLoss` is the default and the most widely tested and useful function. The full loss function list includes:

- `CombinedCEDiceLoss` (Default)
> Loss function that combines *Cross-Entropy* and *DiceLoss*; very robust metric accounting for class imballence while mesuring the overlap between predicted and ground truth masks.

- `BCELoss`
> *Binary Cross Entropy* loss; metric that mesured the difference between predicted probabilities and true binary labels; where labels are either 0 or 1 and lower BCE means better predictions. 

- `DiceLoss`
> Dirived from a dice coefficient mesuring the similarities between the ground truth and predicted masks; focus is on region overlap rather than pixel-specific accuracy. 

- `GeneralizedDiceLoss`
> Focues on smaller inconsistencies and weights them more highly agaisnt the normal data input; rare classes are contributing more to the loss signal providing a better metric for smaller region or user case feature training. 

- `CrossEntropyLoss`
> Metric that calculates and predicts the probability of each class input and how close a models predictions are to the ground truth; thoguh this helping to inform the model of correct parts and help to supress wrong parts. This functions is great with *multiclass data*, where the loss function is minimised to guide the model to more accurate predictions. 

- `TverskyLoss`
> Funtion that deals with differentiable loss looking at pixel-wise probabilities and has been balenced to prioritise both precision and recall. This function is great with sparse-object and multiclass segmentation. 

- `BoundaryDoULoss`
> *Boundary Difference over Union* loss; ??

- `BoundaryLoss`
> Based on a distance metric and takes into consideration regional training information; this function is great with *highly unbalanced datasets* which have values orders of magnitude fom eachother.

- `ClassWeightedDiceLoss`
> Handles different classes in model training by calculating the loss for each seperatly and then averageing it by the class frequency; this gives more credit to smaller classes and compensated for imbalence in multiclass data.


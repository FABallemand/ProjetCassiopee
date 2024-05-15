import os

###########################################################################
## Mocaplab ###############################################################  
###########################################################################
from src.visualisation import create_all_animations

###########################################################################  
## RGBD Object ############################################################
###########################################################################
from src.models.rgbd_object.cnn import rgbd_object_cnn_supervised_training
from src.models.rgbd_object.autoencoder import rgbd_object_ae_unsupervised_training, rgbd_object_ae_unsupervised_contrastive_training
from src.models.rgbd_object.combined_model import rgbd_object_combined_supervised_training, rgbd_object_combined_supervised_contrastive_training

# Run with: nohup python3 -u main.py &

if __name__ == '__main__':
    # Change working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    ###########################################################################
    ## Mocaplab ###############################################################  
    ###########################################################################  
    # create_all_animations()

    ###########################################################################  
    ## RGBD Object ############################################################
    ###########################################################################
    # CNN
    rgbd_object_cnn_supervised_training()

    # AE
    # rgbd_object_ae_unsupervised_training()
    # rgbd_object_ae_unsupervised_contrastive_training()

    # Combined model
    # rgbd_object_combined_supervised_training()

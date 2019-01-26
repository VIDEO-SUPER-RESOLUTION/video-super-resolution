from __future__ import print_function, division

from keras.utils.vis_utils import plot_model
import new_models as models
import img_utils
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

if __name__ == "__main__":
    path = r"headline_carspeed.jpg"
    val_path = "val_images/"

    scale = 2

    """
    Plot the models
    """
    
    model = models.ResNetSR(scale).create_model()
    plot_model(model, to_file="architectures/ResNet.png", show_layer_names=True, show_shapes=True)
    
    """
    Train Res Net SR
    """
            
    rnsr = models.ResNetSR(scale)
    rnsr.create_model(None, None, 3, load_weights=True)
    rnsr.evaluate(val_path)
    
    """
    Evaluate ResNetSR on Set5/14
    """

    rnsr = models.ResNetSR(scale)
    rnsr.create_model(None, None, 3, load_weights=True)
    rnsr.evaluate(val_path)



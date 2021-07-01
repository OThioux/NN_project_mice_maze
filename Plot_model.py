from tensorflow import keras
from keras.utils.vis_utils import plot_model

# Doesn't work.

model = keras.models.load_model("Final_model_fold_1.h5")
plot_model(model, to_file="/NicePlots/Model,png", show_shapes=True)


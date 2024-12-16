import tensorflow as tf
import sys
sys.path.append(r'C:\Users\chaud\OneDrive\Documents\MACHINE LEARNING\utils')
from models_generator import models_generator
from load_data import load_data
from save_data import save_data

var_names = ['X_train', 'X_val', 'y_train', 'y_val']
loaded_vars = load_data(*var_names)

X_train = loaded_vars['X_train']
X_val = loaded_vars['X_val']
y_train = loaded_vars['y_train']
y_val = loaded_vars['y_val']

# models1 = models_generator(
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func='mae',
#     epochs=50,
#     neuron_combinations_array=[ [ [32], [24], [16], [8] ] ],
#     smart_skipping=True
# )

# save_data(models1=models1)

# models2 = models_generator(
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func='mae',
#     epochs=50,
#     neuron_combinations_array=[ [ [32,32], [24,24], [16,16], [8,8] ] ],
#     smart_skipping=True
# )

# save_data(models2=models2)

# models3 = models_generator(
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func='mae',
#     epochs=50,
#     neuron_combinations_array=[ [ [32,32,32], [24,24,24], [16,16,16], [8,8,8] ] ],
#     smart_skipping=True
# )

# save_data(models3=models3)

# models1_2 = models_generator(
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func='mae',
#     epochs=100,
#     neuron_combinations_array=[ [ [32] ] ],
#     smart_skipping=True,
#     learning_rates=[0.1],
#     reg_lambdas = [0.0001, 0.],
#     batch_norm=False
# )

# save_data(models1_2=models1_2)

# models2_2 = models_generator(
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func='mae',
#     epochs=100,
#     neuron_combinations_array=[ [ [32,32], [16,16] ] ],
#     smart_skipping=True,
#     learning_rates=[0.1],
#     reg_lambdas = [0.0001, 0.],
#     batch_norm=False
# )

# save_data(models2_2=models2_2)

# models3_2 = models_generator(
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func='mae',
#     epochs=100,
#     neuron_combinations_array=[ [ [32,32,32], [24,24,24] ] ],
#     smart_skipping=True,
#     learning_rates=[0.01],
#     reg_lambdas = [0.0001, 0.],
#     batch_norm=False
# )

# save_data(models3_2=models3_2)

# models1_3 = models_generator(
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func='mae',
#     epochs=100,
#     neuron_combinations_array=[ [ [24] ] ],
#     smart_skipping=True,
#     learning_rates=[0.1],
#     reg_lambdas = [0.0001, 0.],
#     batch_norm=False
# )

# save_data(models1_3=models1_3)

# models1_4 = models_generator(
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func='mae',
#     epochs=100,
#     neuron_combinations_array=[ [ [16], [8] ] ],
#     smart_skipping=True,
#     learning_rates=[0.1],
#     reg_lambdas = [0.0001, 0.],
#     batch_norm=False
# )

# save_data(models1_4=models1_4)

models1_5 = models_generator(
    X_train,
    y_train,
    X_val,
    y_val,
    total_outputs=3,
    verbose=0,
    loss_func='mae',
    epochs=100,
    neuron_combinations_array=[ [ [7], [6], [5], [4], [3], [2], [1] ] ],
    smart_skipping=True,
    learning_rates=[1., 0.1],
    reg_lambdas = [0.],
    batch_norm=False
)

# save_data(models1_5=models1_5)
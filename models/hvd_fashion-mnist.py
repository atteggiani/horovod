import time 
time_start=time.perf_counter()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam


# Initialize Horovod
hvd.init()

if hvd.rank() == 0:
    print(f"Total number of GPUs: {hvd.size()}")

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# Build model and dataset
def load_dataset():
    def normalise(pixel):
        return pixel.astype('float32') / 255.0
    
    # Load dataset.
    (train_x, train_y), (validation_x, validation_y) = fashion_mnist.load_data()

    # Reshape dataset and normalise.
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    validation_x = validation_x.reshape((validation_x.shape[0], 28, 28, 1))
    train_x = normalise(train_x)
    validation_x = normalise(validation_x)

    # One hot encode target values.
    train_y = to_categorical(train_y)
    validation_y = to_categorical(validation_y)

    return train_x, train_y, validation_x, validation_y

def get_model_definition():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    return model

def compile_model(learning_rate, momentum=None):
    model = get_model_definition()
    # compile model with Horovod DistributedOptimizer.
    # opt = hvd.DistributedOptimizer(SGD(lr=learning_rate, momentum=momentum))
    opt = hvd.DistributedOptimizer(Adam(lr=learning_rate))
    # Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses Horovod's DistributedOptimizer to compute gradients.
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    return model

def main():
    # model_filename = str(utils.model_dir.joinpath('multi-gpu'))
    # checkpoint_filename = str(utils.checkpoint_dir.joinpath('checkpoint-{epoch}.h5'))
    # tensorboard_log_dir = str(utils.tb_log_dir)


    learning_rate = 0.001
    # momentum = 0.9
    batch_size = 320
    epochs = 10

    # Increase the batch size to decrease network traffic between GPUs
    batch_size *= hvd.size()
    # Scale the learning rate (SGD Optimisers benefit from this).
    # source: Accurate, Large Minibatch SGD Training ImageNet in 1 Hour
    learning_rate *= hvd.size()

    # Callbacks
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(initial_lr=learning_rate / hvd.size(),
                                                    warmup_epochs=3,
                                                    verbose=0))

    # Save checkpoint and tensorboard files on worker 0 only.
    # The model is identical across all workers; therefore, other workers should not
    # save this information. Concurrent I/O could also corrupt output files.
    # if hvd.rank() == 0:
        # callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_filename))
        # callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir,histogram_freq=1))


    model = compile_model(learning_rate)
    train_x, train_y, val_x, val_y = load_dataset()
    num_samples = train_x.shape[0]

    # All workers compute the model together.
    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_x, val_y),
        # steps_per_epoch = num_samples // batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1 if hvd.rank() == 0 else 0,
        )
    
    # Worker 0 saves the model when complete (model identical across all workers).
    # if hvd.rank() == 0:
    #     model.save(model_filename)

if __name__ == "__main__":
    main()
if hvd.rank()==0:
    time_end = time.perf_counter()
    final_time = (time_end-time_start)
    print(f"Total elapsed time: {final_time:.4f} seconds.")
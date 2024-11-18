import time
from keras import callbacks

class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()  # Track the start time
        self.epoch_times = []  # List to store time per epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()  # Start time for each epoch

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time  # Calculate epoch duration
        self.epoch_times.append(epoch_time)  # Append epoch time to list

    def on_train_end(self, logs=None):
        self.total_training_time = time.time() - self.start_time  # Calculate total training time
        print(f"Total Training Time: {self.total_training_time:.2f} seconds")
        print(f"Average Epoch Time: {sum(self.epoch_times) / len(self.epoch_times):.2f} seconds")
        
print("Callback class imported successfully")
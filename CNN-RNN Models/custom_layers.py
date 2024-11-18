import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

'''
Improved LSTM:
    - Inherit from keras.layers to create own custom layer
    - iLSTM is used in a RNN wrapper, as the RNN layer in keras is a flexible way to 
      handle custom recurrent layers
    - Contains 3 Gates: Forget, Input, and Output Gates
'''
"""  
@register_keras_serializable()
class iLSTM(layers.Layer): 
    def __init__(self, units, **kwargs):
        super(iLSTM, self).__init__(**kwargs)
            #**kwargs allows for passing additional configuration options to parent layer class
        
        # Stores the number of units for the iLSTM Layer
        self.units = units
        
        # Define state size
        self.state_size = [self.units, self.units]  # [hidden_state, cell_state]

        # Define output size
        self.output_size = self.units

    def build(self, input_shape): # Called when layer is first used
        # Initialize Weights for Forget Gate
        # Weight Matrix for Forget Gate connecting Input to Forget Gate
        self.W_fx = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='he_normal',
                                    trainable=True)
        # Weight Matrix for Forget Gate connecting Previous Hidden State to Forget Gate
        self.W_fh = self.add_weight(shape=(self.units, self.units),
                                    initializer='he_normal',
                                    trainable=True)
        # Bias Vector for Forget Gate 
        self.b_f = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   trainable=True)

        # Initialize Weights for Input Gate
        # Weight Matrix for Input Gate connecting the Input to Input Gate
        self.W_ix = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='he_normal',
                                    trainable=True)
        # Weight Matrix for Input Gate connecting Previous Hidden State to Input Gate
        self.W_ih = self.add_weight(shape=(self.units, self.units),
                                   initializer='he_normal',
                                   trainable=True)
        # Bias Vector for Input Gate 
        self.b_i = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   trainable=True)

    def call(self, inputs, states):
        prev_hidden_state, prev_cell_state = states

        # Calculate the Forget Gate
        f_t = tf.sigmoid(tf.matmul(inputs, self.W_fx) + tf.matmul(prev_hidden_state, self.W_fh) + self.b_f)

        # Calculate the Input Gate
        i_t = tf.sigmoid(tf.matmul(inputs, self.W_ix) + tf.matmul(prev_hidden_state, self.W_ih) + prev_cell_state + self.b_i)

        # Mainline forgetting mechanism from iLSTM paper
        k_t = f_t * prev_cell_state
        
        # Conversion Information Module (CIM)
        cim_output = tf.tanh(i_t) + k_t
        
        c_t = cim_output

        # Next Hidden State (Without output gate, we directly use the updated cell state)
        h_t = tf.tanh(c_t)

        return h_t, [h_t, c_t]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        # Initialize the hidden state and cell state to zeros
        h = tf.zeros((batch_size, self.units), dtype=dtype)
        c = tf.zeros((batch_size, self.units), dtype=dtype)
        return [h, c]
    
    def get_config(self):
        base_config = super().get_config()
        config = {"units": self.units}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
"""  
@register_keras_serializable()
class iLSTM(layers.Layer): 
    def __init__(self, units, kernel_regularizer=None, **kwargs):  # Add kernel_regularizer as an argument
        super(iLSTM, self).__init__(**kwargs)
        
        # Stores Units
        self.units = units
        # Store the regularizer
        self.kernel_regularizer = kernel_regularizer  

        # Define state size
        self.state_size = [self.units, self.units]  # [hidden_state, cell_state]
        self.output_size = self.units

    def build(self, input_shape):  # Called when layer is first used
        # Initialize Weights for Forget Gate
        self.W_fx = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='he_normal',
                                    regularizer=self.kernel_regularizer,  # Apply regularizer here
                                    trainable=True)
        self.W_fh = self.add_weight(shape=(self.units, self.units),
                                    initializer='he_normal',
                                    regularizer=self.kernel_regularizer,  # Apply regularizer here
                                    trainable=True)
        self.b_f = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   trainable=True)

        # Initialize Weights for Input Gate
        self.W_ix = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='he_normal',
                                    regularizer=self.kernel_regularizer,  # Apply regularizer here
                                    trainable=True)
        self.W_ih = self.add_weight(shape=(self.units, self.units),
                                    initializer='he_normal',
                                    regularizer=self.kernel_regularizer,  # Apply regularizer here
                                    trainable=True)
        self.b_i = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   trainable=True)
        
    def call(self, inputs, states):
        prev_hidden_state, prev_cell_state = states

        # Calculate the Forget Gate
        f_t = tf.sigmoid(tf.matmul(inputs, self.W_fx) + tf.matmul(prev_hidden_state, self.W_fh) + self.b_f)

        # Calculate the Input Gate
        i_t = tf.sigmoid(tf.matmul(inputs, self.W_ix) + tf.matmul(prev_hidden_state, self.W_ih) + prev_cell_state + self.b_i)

        # Mainline forgetting mechanism from iLSTM paper
        k_t = f_t * prev_cell_state
        
        # Conversion Information Module (CIM)
        cim_output = tf.tanh(i_t) + k_t
        
        c_t = cim_output

        # Next Hidden State (Without output gate, we directly use the updated cell state)
        h_t = tf.tanh(c_t)

        return h_t, [h_t, c_t]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        # Initialize the hidden state and cell state to zeros
        h = tf.zeros((batch_size, self.units), dtype=dtype)
        c = tf.zeros((batch_size, self.units), dtype=dtype)
        return [h, c]
    
    def get_config(self):
        base_config = super().get_config()
        config = {"units": self.units}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
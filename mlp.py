import numpy as np
import pathlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, BatchNormalization, Activation, Bidirectional
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

import tensorflow.keras.backend as K
import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import*

class ITSmlp(object):
    def __init__(self):
        X, Xtest, Y, Ytest, encoder = get_data()
        self.X = X
        self.Y = Y
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.encoder = encoder
        self.num_classes = len(set(self.Y))
        print(" num locations : {}".format(self.num_classes))
        print(" Train Input Shape : {}".format(self.X.shape))
        print(" Train Label Shape : {}".format(self.Y.shape))
        print(" Test  Input Shape : {}".format(self.Xtest.shape))
        print(" Test  Label Shape : {}".format(self.Ytest.shape))

    def classifier(self, activation='relu'):
        inputs = Input(shape=(n_features,))
        x = Dense(dense1, activation=activation)(inputs)
        x = Dense(dense1)(x)
        x = Dense(dense1)(x)
        x = BatchNormalization()(x)
        x = Dense(dense2, activation=activation)(x)
        x = Dense(dense2, activation=activation)(x)
        x = Dense(dense2)(x)
        x = BatchNormalization()(x)
        x = Dense(dense3, activation=activation)(x)
        x = Dense(dense3, activation=activation)(x)
        x = Dense(dense3, activation=activation)(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)
        self.model.summary()

        # tf.keras.utils.plot_model(
        #                 self.model, 
        #                 to_file=dot_img_file, 
        #                 show_shapes=True
        #                 )

    def LSTMclassifier(self):
        inputs = Input(shape=(n_features,1))
        x = Bidirectional(LSTM(128))(inputs)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)
        self.model.summary()

        tf.keras.utils.plot_model(
                        self.model, 
                        to_file=dot_img_file, 
                        show_shapes=True
                        )

    def train(self):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate),
            metrics=['accuracy'],
        )
        callback = EarlyStopping(
                            monitor="val_loss",
                            patience=20,
                            mode="min",
                            restore_best_weights=True
                        )
        self.history = self.model.fit(
                            self.X,
                            self.Y,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_split=validation_split,
                            callbacks=[callback]
                            )
        self.plot_metrics()

    def plot_metrics(self):
        loss_train = self.history.history['loss']
        loss_val = self.history.history['val_loss']
        
        loss_train = np.cumsum(loss_train) / np.arange(1,len(loss_train)+1)
        loss_val = np.cumsum(loss_val) / np.arange(1,len(loss_val)+1)
        plt.plot(loss_train, 'r', label='Training loss')
        plt.plot(loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(loss_img)
        plt.legend()
        plt.show()
        
        acc_train = self.history.history['accuracy']
        acc_val = self.history.history['val_accuracy']
        
        acc_train = np.cumsum(acc_train) / np.arange(1,len(acc_train)+1)
        acc_val = np.cumsum(acc_val) / np.arange(1,len(acc_val)+1)
        plt.plot(acc_train, 'r', label='Training Accuracy')
        plt.plot(acc_val, 'b', label='validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(acc_img)
        plt.legend()
        plt.show()

    def save_model(self):
        self.model.save(mlp_weights)

    def load_model(self):
        loaded_model = load_model(mlp_weights)
        loaded_model.compile(
                        loss='sparse_categorical_crossentropy',
                        optimizer=Adam(learning_rate),
                        metrics=['accuracy']
                        )
        self.model = loaded_model

    def TFconverter(self): # For deployment in the mobile devices quantization of the model using tensorflow lite
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(mlp_weights)
        converter.target_spec.supported_ops = [
                                tf.lite.OpsSet.TFLITE_BUILTINS,   # Handling unsupported tensorflow Ops 
                                tf.lite.OpsSet.SELECT_TF_OPS 
                                ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]      # Set optimization default and it configure between latency, accuracy and model size
        tflite_model = converter.convert()

        mlp_converter_file = pathlib.Path(mlp_converter) 
        mlp_converter_file.write_bytes(tflite_model) # save the tflite model in byte format
 
    def TFinterpreter(self):
        self.interpreter = tf.lite.Interpreter(model_path=mlp_converter) # Load tflite model
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details() # Get input details of the model
        self.output_details = self.interpreter.get_output_details() # Get output details of the model

    def Inference(self, features):
        features = features.astype(np.float32)
        input_shape = self.input_details[0]['shape']
        assert np.array_equal(input_shape, features.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], features)

        self.interpreter.invoke() # set the inference

        output_data = self.interpreter.get_tensor(self.output_details[0]['index']) # Get predictions
        return output_data

    def predictions(self, X):
        P = self.model.predict(X)
        Ypred = P.argmax(axis=-1)
        Ypred = self.encoder.inverse_transform(Ypred)
        return Ypred

    def evaluation(self):
        model = self.model
        encoder = self.encoder
        X = self.X 
        Y = self.Y
        route_accuracy(encoder, model, X, Y)
        state_error_analysis(encoder, model, X, Y)

    def run(self):
        if os.path.exists(mlp_weights):
            print(" MLP model Loading !!")
            self.load_model()
        else:
            print(" MLP model Training !!")
            self.classifier()
            self.train()
            self.save_model()

        if not os.path.exists(mlp_converter):
            self.TFconverter()
            self.TFinterpreter()


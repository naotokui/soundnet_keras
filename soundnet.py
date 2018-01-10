from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, Input
from keras.models import Model
import numpy as np
import librosa


def preprocess(audio):
    audio *= 256.0  # SoundNet needs the range to be between -256 and 256
    # reshaping the audio data so it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1, 1))
    return audio


def load_audio(audio_file):
    sample_rate = 22050  # SoundNet works on mono audio files with a sample rate of 22050.
    audio, sr = librosa.load(audio_file, dtype='float32', sr=22050, mono=True)
    audio = preprocess(audio)
    return audio


def build_model():
    """
    Builds up the SoundNet model and loads the weights from a given model file (8-layer model is kept at models/sound8.npy).
    :return:
    """
    model_weights = np.load('models/sound8.npy').item()

    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 4, 'pool_strides': 4},

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                        {'name': 'conv8', 'num_filters': 1000, 'padding': 0,
                            'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                          'kernel_size': 8, 'conv_strides': 2},
                         ]

    inputs = Input(shape=(None, 1)) # define inputs

    x = inputs
    for layer in filter_parameters:
        if 'conv8' not in layer['name']:
            x = ZeroPadding1D(padding=layer['padding'])(x)
        else:
            x = ZeroPadding1D(padding=layer['padding'])(conv7_layer_output)

        conv_layer = Conv1D(layer['num_filters'],
                        kernel_size=layer['kernel_size'],
                        strides=layer['conv_strides'],
                        padding='valid', name=layer['name'])

        weights = model_weights[layer['name']]['weights'].reshape(conv_layer.get_weights()[0].shape)
        biases = model_weights[layer['name']]['biases']
        conv_layer.set_weights([weights, biases])

        x = conv_layer(x)

        if 'conv8' not in layer['name']: # except the last layers
            gamma = model_weights[layer['name']]['gamma']
            beta = model_weights[layer['name']]['beta']
            mean = model_weights[layer['name']]['mean']
            var = model_weights[layer['name']]['var']

            batch_norm = BatchNormalization()
            batch_norm.set_weights([gamma, beta, mean, var])
            x = batch_norm(x)
            x = Activation('relu')(x)
        if 'pool_size' in layer:
            x = MaxPooling1D(pool_size=layer['pool_size'],
                                   strides=layer['pool_strides'],
                                   padding='valid')(x)
        if layer['name'] == 'conv7':
            conv7_layer_output = x
        elif layer['name'] == 'conv8':
            imagenet_output = x
        elif layer['name'] == 'conv8_2':
            places_output = x

    model = Model(inputs=inputs,outputs=[imagenet_output, places_output])
    return model

def predict_from_audio_file(model, audio_file):
    audio = load_audio(audio_file)
    return model.predict(audio)

def predictions_to_scenes(prediction):
    scenes = []
    with open('categories/categories_places2.txt', 'r') as f:
        categories = f.read().split('\n')
        for p in range(prediction.shape[1]):
            scenes.append(categories[np.argmax(prediction[0, p, :])])
    return scenes

def predictions_to_object(prediction):
    scenes = []
    with open('categories/categories_imagenet_1000.txt', 'r') as f:
        categories = f.read().split('\n')
        for p in range(prediction.shape[1]):
            scenes.append(categories[np.argmax(prediction[0, p, :])])
    return scenes

if __name__ == '__main__':
    #  SoundNet demonstration
    model = build_model()
    model.summary()

    prediction = predict_from_audio_file(model, 'railroad_audio.wav')
    print "imagenet:", predictions_to_object(prediction[0])
    print "places:", predictions_to_scenes(prediction[1])

from keras.applications.mobilenet import MobileNet
from keras.layers import Input, GlobalAveragePooling2D, Dense, Activation
from keras.models import Model

def build(num_classes, input_size=(256,256,3)):

    feature_extractor = MobileNet(input_tensor=Input(shape=input_size), input_shape=input_size, weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(feature_extractor.output)
    x = Dense(num_classes)(x)
    x = Activation("softmax")(x)
    model = Model(inputs=feature_extractor.input, outputs=x)
    return feature_extractor, model

if __name__ == "__main__":
    _, model = build(1000)
    print('success')
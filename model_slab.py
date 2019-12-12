from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from keras.layers import MaxPooling3D, UpSampling3D
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers import ZeroPadding3D, Lambda
import keras.backend as K

def contract2d(prev_layer, n_kernel, kernel_size, pool_size, padding,act, dropout=False):
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(prev_layer)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel << 1
    if dropout:
        conv = Dropout(.2)(conv)
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    # conv = Conv2D(n_kernel, kernel_size, padding=padding)(conv)
    # conv = Activation(act)(BatchNormalization()(conv))
    pool = MaxPooling2D(pool_size=pool_size,strides=pool_size)((conv))
    return conv, pool

def expand2d(prev_layer, left_layer, n_kernel,kernel_size,pool_size, padding,act, dropout=False):
    up = Concatenate(axis=-1)([UpSampling2D(size=pool_size)(prev_layer), left_layer])
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(up)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel >> 1
    if dropout:
        conv = Dropout(.2)(conv)
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    return conv

def contract3d(prev_layer, n_kernel, kernel_size, pool_size, padding,act,pooling):
    conv = Conv3D(n_kernel, kernel_size, padding='valid')(prev_layer)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel << 1
    conv = Conv3D(n_kernel, kernel_size, padding='valid')(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    conv = ZeroPadding3D(padding=(0, 2 * (kernel_size//2), 2 * (kernel_size//2)))(conv)
    if pooling:
        pool = MaxPooling3D(pool_size=pool_size,strides=pool_size)((conv))
        return conv, pool
    else:
        return conv, None

#def expand3d(prev_layer, left_layer, n_kernel,kernel_size,pool_size, padding,act, dropout=False):
#    up = Concatenate(axis=-1)([UpSampling3D(size=pool_size)(prev_layer), left_layer])
#    conv = Conv3D(n_kernel, kernel_size, padding=padding)(up)
#    conv = Activation(act)(BatchNormalization()(conv))
#    n_kernel = n_kernel >> 1
#    if dropout:
#        conv = Dropout(.25)(conv)
#    conv = Conv3D(n_kernel, kernel_size, padding=padding)(conv)
#    conv = Activation(act)(BatchNormalization()(conv))
#    return conv

def build_model(input_ch=1, output_ch=1, dropout=True):
#    inputs = Input((512,512,input_ch))
    inputs = Input((input_ch,512,512,2))

    kernel_size = 3
    pool_size = 2
    padding='same'
    activation='relu'
    n_kernel = 16

#    pool = Lambda(lambda x: K.expand_dims(x,-1))(inputs)
    pool = inputs
    enc3ds = []
    n_3dconvs = 2
    for ic in range(n_3dconvs):
        conv, pool = contract3d(pool, n_kernel,kernel_size,pool_size,padding,activation,pooling=True)
        enc3ds.append(conv)
        n_kernel = conv.shape[-1].value

    # n_kernel <<= 1
    pool = Lambda(lambda x: K.squeeze(x,1))(pool)
    conv = pool
    encs = []
    for _ in range(4):
        conv, pool = contract2d(pool, n_kernel,kernel_size,pool_size,padding,activation)
        encs.append(conv)
        n_kernel = conv.shape[-1].value

    for i, enc in enumerate(encs[-2::-1]):
        conv = expand2d(conv, enc, n_kernel,kernel_size,pool_size,padding,activation, dropout=True)
        n_kernel = conv.shape[-1].value

    for i, enc in enumerate(enc3ds[-1::-1]):
        enc = Conv3D(n_kernel, (enc.shape[1].value,1,1))(enc)
        enc = Lambda(lambda x: K.squeeze(x,1))(enc)
        conv = expand2d(conv, enc, n_kernel,kernel_size,pool_size,padding,activation, dropout=True)
        n_kernel = conv.shape[-1].value

    output = Conv2D(output_ch, 1, padding=padding, activation='softmax')(conv)

    return Model(inputs=inputs, outputs=output)

if __name__ == "__main__":
    model = build_model(16,9)
    import utils
    utils.visualize_network(model, 'model.svg')
    model.summary()
    # for l in model.layers:
    #     if isinstance(l, Dropout):
    #         print('remove')

    #         model.layers.remove(l)
    #         l.input.output = l.output
    #         l.output.input = l.input

    # print(model.layers)
    # utils.visualize_network(Model(inputs=model.inputs,outputs=[model.layers[-1].output]), 'nodo_model.svg')

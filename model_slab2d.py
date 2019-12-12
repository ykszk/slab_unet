from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.layers import MaxPooling3D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import ZeroPadding3D, Lambda, Permute

def contract2d(prev_layer, n_kernel, kernel_size, pool_size, padding,act):
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(prev_layer)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel << 1
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    pool = MaxPooling2D(pool_size=pool_size,strides=pool_size)((conv))
    return conv, pool

def expand2d(prev_layer, left_layer, n_kernel,kernel_size,pool_size, padding,act, dropout=False):
    up = Concatenate(axis=-1)([UpSampling2D(size=pool_size)(prev_layer), left_layer])
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(up)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel >> 1
    if dropout:
        conv = Dropout(.25)(conv)
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    return conv

def build_model(input_shape, output_ch=1, dropout=True):
    inputs = Input(input_shape, name='NV_MODEL_INPUT')

    kernel_size = 3
    pool_size = 2
    padding='same'
    activation='relu'
    n_kernel = 16

    permuted = Permute((2,3,1))(inputs)
#    inputs = Permute((1,3))(inputs)

    if True:
        pool = AveragePooling2D(pool_size)(permuted)
        encs = [permuted]
    else:
        pool = permuted
        encs = []
    for _ in range(5):
        conv, pool = contract2d(pool, n_kernel,kernel_size,pool_size,padding,activation)
        encs.append(conv)
        n_kernel = conv.shape[-1].value

    for enc in encs[-2::-1]:
        conv = expand2d(conv, enc, n_kernel,kernel_size,pool_size,padding,activation, dropout=dropout)
        n_kernel = conv.shape[-1].value

    output = Conv2D(output_ch, 1, padding=padding, activation='softmax')(conv)
    output = Activation('linear', name='NV_MODEL_OUTPUT')(output)

    return Model(inputs=inputs, outputs=output)

if __name__ == "__main__":
    model = build_model((3,512,512),6)
    if True:
        import utils
        utils.visualize_network(model, 'model.png', show_layer_names=True)
    model.summary()
import time
start_time = time.time()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow
tf = tensorflow.compat.v1
tf.disable_eager_execution()
kr = tf.keras
import datetime

def main():
    # prepare dataset
    batch_size = 10
    img_shape = (20, 20)

    gen = kr.preprocessing.image.ImageDataGenerator(
        rescale=1/255.,
        rotation_range=180.,
        validation_split=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )

    gen_param = {"directory" : "imgs/train",
                 "target_size" : img_shape,
                 "class_mode" : "categorical",
                 "batch_size" : batch_size,
                 "shuffle" : True,
                 "color_mode" : "grayscale"}

    train_gen = gen.flow_from_directory(subset="training", **gen_param)
    val_gen = gen.flow_from_directory(subset="validation", **gen_param)

    # build classifier
    model = build_classifier(img_shape)
    print(model.summary())
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # train classifier
    num_of_epoch = 20

    logs_dir = "logs/fit/"
    tsb = kr.callbacks.TensorBoard(log_dir=logs_dir)

    steps_per_epoch = train_gen.n
    val_steps = val_gen.n
    
    model.fit_generator(
        generator=train_gen, 
        steps_per_epoch=steps_per_epoch,
        epochs=num_of_epoch, 
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=[tsb]
    )
    
    nowtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save("model/classifier_{}.h5".format(nowtime))

def build_classifier(img_size: tuple):
    param = {"kernel_size" : (3, 3), 
             "strides" : (1, 1), 
             "padding" : "same", 
             "activation" : "relu"}


    # 整数はフィルターをその数だけ持つ畳み込み層
    # "M"はプーリング層
    layers = [32, 32, "M"]
    
    model = kr.models.Sequential()

    model.add(kr.layers.Conv2D(filters=layers[0], input_shape=(img_size[0], img_size[1], 1), **param))
    for l in layers[1:]:
        if l == "M":
            model.add(kr.layers.MaxPooling2D(pool_size=(2, 2)))        
        else:
            model.add(kr.layers.Conv2D(filters=l, **param))

    model.add(kr.layers.Flatten())
    model.add(kr.layers.Dense(units=1024, activation="relu"))
    model.add(kr.layers.Dropout(0.2))
    model.add(kr.layers.Dense(units=1024, activation="relu"))
    model.add(kr.layers.Dropout(0.2))
    model.add(kr.layers.Dense(units=3, activation="softmax"))

    return model

if __name__ == "__main__":
    main()
    stop_time = time.time()
    print("elapsed_time : {}".format(stop_time - start_time))

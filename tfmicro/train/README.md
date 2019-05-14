# Building Test TFMicro Application

We will first build on x86 on a desktop or laptop to check that our model
works. The first model we will test is a small MNIST model trained in
Tensorflow and converted into TFLite.

## Building C++ TFLite Library

Make sure to clone tensorflow and follow their instructions to download
and set up bazel.

```bash
git clone https://www.github.com/tensorflow/tensorflow
cd tensorflow
bazel build //tensorflow/contrib/lite:all
git clone https://www.github.com/google/flatbuffers
```

Then, we can train a MNIST model, and then convert it into a TFLite model. Then,
to use this in TF-Micro, we will directly represent the model binary as a C++
array and load it as a model.

```bash
python3 train.py
python3 convert.py
xxd -i lite_model_file.tflite > tf_micro_model.cpp
```


### Troubleshooting

Building simple C++ applications with TFMicro on an x86 laptop or desktop was
very annoying, here are some links that could be helpful.

* [Build TF-Life and Link](https://github.com/tensorflow/tensorflow/issues/16219)

#### Test cpp script doesn't work!

Make sure that tensorflow is built fron source. Then, navigate to the directory

```bash
ls $TF_HOME/bazel-bin/tensorflow/lite
```

you should be able to see `libtensorflowlite.so`. Make sure that this path is added to your `LD_LIBRARY_PATH` before you execute the test script so we can find that library.

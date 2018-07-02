# Tensorflow.jsx Demo

This is a demo project to quickly get started with tfjsx.
Check out the [tfjsx project here](https://github.com/ModelDepot/tfjsx).

![out of the box training visualization](https://github.com/ModelDepot/tfjsx/blob/master/assets/demo.png?raw=true)

# Getting Started

## Clone and Install Dependencies

```
$ git clone https://github.com/ModelDepot/tfjsx.git
$ cd tfjsx
$ yarn
```

## Start Dev Server
```
$ yarn start
```

Visit http://localhost:1234 to see it running!

# What's Inside

Inside `src/index.js` there's two models and some basic code to demonstrate
what you can do with tfjsx.

## MyModel

`MyModel` is a simple linear regression model, the
[exact same one](https://js.tensorflow.org/#getting-started) found
on Tensorflow.js's home page. It includes a `trainDataGenerator` that has
7 hard coded samples that the model will try to fit.

## MnistModel

`MnistModel` is a more complex CNN found in
[Tensorflow.js's MNIST tutorial](https://js.tensorflow.org/tutorials/mnist.html).
It includes a validation data generator to show how you can visualize validation
metrics on every epoch.

The model training can be paused using the `train` prop for the `<Train>`
component.

Pressing the Predict button will output predictions for three samples
(1, 3 and 9) in the JS console.

## MyApp

`MyApp` component drives the model training and holds the final trained
model state. You can swap out rendering `MnistModel` to `MyModel` if desired.

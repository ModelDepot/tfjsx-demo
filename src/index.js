import React from 'react';
import ReactDOM from 'react-dom';
import { Train, Model, Dense, Conv2D, MaxPooling2D, Flatten } from 'tfjsx';
import * as tf from '@tensorflow/tfjs';
import mnist from 'mnist';

function* trainDataGenerator() {
  yield { x: 1, y: 1 };
  yield { x: 2, y: 2 };
  yield { x: 3, y: 3 };
  yield { x: 4, y: 4 };
  yield { x: 5, y: 5 };
  yield { x: 6, y: 6 };
  yield { x: 11, y: 11 };
}

class MyModel extends React.Component {
  render() {
    return (
      <Train
        trainData={trainDataGenerator}
        epochs={15}
        batchSize={5}
        samples={7}
        onTrainEnd={this.props.onTrainEnd}
        train
        display
      >
        <Model optimizer='sgd' loss='meanSquaredError'>
          <Dense units={1} inputShape={[1]} />
        </Model>
      </Train>
    );
  }
}

const data = mnist.set(1000, 100);
const train = data.training;
const test = data.test;

function* mnistTrainDataGenerator() {
  for (let sample of train) {
    const square_sample = tf.tensor1d(sample.input).reshape([28, 28]);
    yield { x: tf.tensor1d(sample.input).reshape([28, 28, 1]), y: sample.output };
  }
}

function* mnistTestDataGenerator() {
  for (let sample of test) {
    const square_sample = tf.tensor1d(sample.input).reshape([28, 28]);
    yield { x: tf.tensor1d(sample.input).reshape([28, 28, 1]), y: sample.output };
  }
}

class MnistModel extends React.Component {
  render() {
    return (
      <Train
        trainData={mnistTrainDataGenerator}
        samples={1000}
        validationData={mnistTestDataGenerator}
        onBatchEnd={metrics => console.log(metrics)}
        epochs={15}
        batchSize={128}
        onTrainEnd={this.props.onTrainEnd}
        train={this.props.train}
        display
      >
        <Model
          optimizer={tf.train.sgd(0.15)}
          loss='categoricalCrossentropy'
          metrics={['accuracy']}>
          <Conv2D
            inputShape={[28, 28, 1]}
            kernelSize={5}
            filters={8}
            strides={1}
            activation='relu'
            kernelInitializer='VarianceScaling' />
          <MaxPooling2D poolSize={[2, 2]} strides={[2, 2]} />
          <Conv2D
            kernelSize={5}
            filters={16}
            strides={1}
            activation='relu'
            kernelInitializer='VarianceScaling' />
          <MaxPooling2D poolSize={[2, 2]} strides={[2, 2]} />
          <Flatten />
          <Dense units={10} kernelInitializer='VarianceScaling' activation='softmax' />
        </Model>
      </Train>
    );
  }
}

class MyApp extends React.Component {
  state = {
    model: null,
    training: false,
  };

  render() {
    const one = tf.tensor1d(mnist[1].get()).reshape([28, 28, 1]);
    const three = tf.tensor1d(mnist[3].get()).reshape([28, 28, 1]);
    const nine = tf.tensor1d(mnist[9].get()).reshape([28, 28, 1]);

    const test = tf.stack([one, three, nine]);

    // const test = tf.tensor([ 1, 3, 25, 99 ], [4, 1]);

    return (
      <div>
        <button onClick={() => {
          this.state.model.predict(test).print();
        }}>
          Predict!
        </button>
        <button onClick={() => this.setState({ training: !this.state.training })}>
          {this.state.training ? 'Pause Training' : 'Start Training'}
        </button>
        <MnistModel
          onTrainEnd={model => this.setState(model)}
          train={this.state.training}/>
      </div>
    );
  }
}

ReactDOM.render(<MyApp />, document.getElementById('app'));

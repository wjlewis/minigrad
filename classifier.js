import { NN } from './nn.js';
import { Value } from './index.js';

// This sample data resembles a circular target, with the 1s region enclosd by
// the 0s. The two sets cannot be separate by a line.
const trainingData = [
  [0.0, 0.0, 1.0],
  [0.74, 0.0, 1.0],
  [0.46, 0.8, 1.0],
  [0.48, -0.6, 1.0],
  [-0.33, 0.7, 1.0],
  [-0.42, -0.2, 1.0],
  [-0.89, -0.1, 1.0],
  [-0.23, -0.78, 1.0],

  [0.0, 1.0, 0.0],
  [0.62, 1.3, 0.0],
  [1.3, 0.45, 0.0],
  [1.34, 0.0, 0.0],
  [1.78, -0.7, 0.0],
  [1.1, -0.8, 0.0],
  [0.3, -1.78, 0.0],
  [0.1, -1.5, 0.0],
  [-0.63, -1.3, 0.0],
  [-1.37, -1.01, 0.0],
  [-1.94, 0.0, 0.0],
  [-1.38, 0.38, 0.0],
  [-0.3, 1.29, 0.0],
];

function train() {
  const mlp = new NN(2, 4, 4, 1);
  const learningRate = 0.01;

  for (let n = 0; n < 1_000; n++) {
    const loss = trainingData
      .map(([x, y, expected]) => {
        const received = mlp.forward([x, y])[0];
        return Value.of(expected).minus(received).pow(2);
      })
      .reduce((sum, diff) => sum.plus(diff), Value.of(0));

    console.log(loss.data);

    loss.zeroGrad();
    loss.backward();

    for (let p of mlp.params()) {
      p.data += learningRate * -p.grad;
    }
  }
}

train();

import { Value } from './index.js';

export class NN {
  constructor(nInputs, ...nSizes) {
    this._layers = nSizes.map((nOutputs, i) => {
      const nIn = i === 0 ? nInputs : nSizes[i - 1];
      return new Layer(nIn, nOutputs);
    });
  }

  /**
   * Perform forward propagation with the JavaScript values `xs`. These inputs
   * should _not_ be instances of `Value`.
   */
  forward(xs) {
    return this._layers.reduce(
      (ys, layer) => layer.forward(ys),
      xs.map(Value.of)
    );
  }

  /**
   * Return all of the parameters (weights and biases) of `this` network.
   */
  params() {
    return this._layers.flatMap(l => l.params());
  }
}

class Layer {
  constructor(nInputs, nOutputs) {
    this._neurons = range(nOutputs).map(() => new Neuron(nInputs));
  }

  forward(vs) {
    return this._neurons.map(n => n.forward(vs));
  }

  params() {
    return this._neurons.flatMap(n => n.params());
  }
}

class Neuron {
  constructor(nInputs) {
    this._weights = range(nInputs).map(unitRandom).map(Value.of);
    this._bias = Value.of(unitRandom());
  }

  forward(vs) {
    const activation = zip(vs, this._weights)
      .map(([v, w]) => v.times(w))
      .reduce((sum, y) => sum.plus(y), this._bias);
    return activation.tanh();
  }

  params() {
    return [this._bias, ...this._weights];
  }
}

/**
 * Construct an array of numbers from `0` to `to`.
 */
function range(to) {
  const out = [];
  for (let n = 0; n < to; n++) {
    out.push(n);
  }
  return out;
}

/**
 * Produce a uniformly generated random number in the interval `[-1, 1]`.
 */
function unitRandom() {
  return Math.random() * 2 - 1;
}

/**
 * Zip two arrays together, producing an array of pairs.
 */
function zip(xs, ys) {
  const out = [];
  const len = Math.max(xs.length, ys.length);
  for (let i = 0; i < len; i++) {
    out.push([xs[i], ys[i]]);
  }
  return out;
}

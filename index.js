/**
 * A small autograd system, inspired by `micrograd`.
 * Values are automatically propagated forward through the expression, and
 * gradients are backpropagated by calling `backward`.
 */
export class Value {
  constructor(data, deps, backward, opName) {
    this.data = data;
    this.grad = 0;
    this._deps = deps;
    this._backward = () => backward(this.grad, this.data);
    this._opName = opName;
  }

  /**
   * Construct a `Value` from a JS value. At the moment, only Numbers are
   * supported.
   */
  static of(x) {
    return new Value(x, [], () => {}, 'scalar');
  }

  /**
   * Add `rhs` to `this`, producing a new `Value`.
   */
  plus(rhs) {
    return new Value(
      this.data + rhs.data,
      [this, rhs],
      grad => {
        this.grad += 1 * grad;
        rhs.grad += 1 * grad;
      },
      '+'
    );
  }

  /**
   * Negate `this`, producing a new `Value`.
   */
  neg() {
    return new Value(
      -this.data,
      [this],
      grad => {
        this.grad += -1 * grad;
      },
      '-'
    );
  }

  /**
   * Subtract `rhs` from `this`, producing a new `Value`.
   */
  minus(rhs) {
    return this.plus(rhs.neg());
  }

  /**
   * Multiply `this` times `rhs`, producing a new `Value`.
   */
  times(rhs) {
    return new Value(
      this.data * rhs.data,
      [this, rhs],
      grad => {
        this.grad += rhs.data * grad;
        rhs.grad += this.data * grad;
      },
      '*'
    );
  }

  /**
   * Exponentiate `this`, producing a new `Value`.
   */
  exp() {
    return new Value(
      Math.exp(this.data),
      [this],
      (grad, data) => {
        this.grad += data * grad;
      },
      'exp'
    );
  }

  /**
   * Raise `this` to the `n`th power, where `n` is a JS number.
   */
  pow(n) {
    return new Value(
      this.data ** n,
      [this],
      grad => {
        this.grad += n * this.data ** (n - 1) * grad;
      },
      `^${n}`
    );
  }

  /**
   * Divide `this` by `rhs`, producing a new `Value`.
   */
  div(rhs) {
    return this.times(rhs.pow(-1));
  }

  /**
   * Return the hyperbolic tangent of `this`, producing a new `Value`.
   */
  tanh() {
    const exp = this.times(Value.of(2)).exp();
    return exp.minus(Value.of(1)).div(exp.plus(Value.of(1)));
  }

  /**
   * Zero any computed gradients that are part of this `Value`.
   */
  zeroGrad() {
    this.grad = 0;
    for (const dep of this._deps) {
      dep.zeroGrad();
    }
  }

  /**
   * Compute the gradients of all dependencies with respect to `this`.
   */
  backward() {
    const sorted = this._sort();
    this.grad = 1;
    for (const value of sorted) {
      value._backward();
    }
  }

  /**
   * Return a string describing the structure of `this`.
   */
  debug(level = 0) {
    const indent = ' '.repeat(level * 2);
    const data = this.data.toPrecision(4);
    const grad = this.grad.toPrecision(4);
    const info = `${indent}data = ${data}; grad = ${grad}`;
    if (this._deps.length === 0) {
      return info;
    }
    const opInfo = `${indent}${this._opName}`;
    const depInfo = this._deps.map(dep => dep.debug(level + 1));
    return [info, opInfo, ...depInfo].join('\n');
  }

  _sort() {
    const seen = new Set();
    const sorted = [];

    function addValue(v) {
      if (seen.has(v)) {
        return;
      }
      seen.add(v);
      for (const dep of v._deps) {
        addValue(dep);
      }
      sorted.push(v);
    }

    addValue(this);
    sorted.reverse();
    return sorted;
  }
}

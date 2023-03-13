import { Value } from './index.js';

const v = Value.of(2).times(Value.of(3)).div(Value.of(4)).exp();

console.log(v.debug());

console.log('\ncomputing gradients...\n');

v.backward();
console.log(v.debug());

console.log('\nzeroing computed gradients...\n');

v.zeroGrad();
console.log(v.debug());

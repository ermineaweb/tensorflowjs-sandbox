const tf = require("@tensorflow/tfjs-node-gpu");

// const learningRate = [0.0001, 0.00001, 0.000001];
const learningRate = [null];

// const activation = ["relu", "sigmoid"];
const activation = ["relu"];

const optimizer = [
  { label: "adam", fn: (lr) => (lr ? tf.train.adam(lr) : "adam") },
  { label: "sgd", fn: (lr) => (lr ? tf.train.sgd(lr) : "sgd") },
  // (lr) => (lr ? tf.train.sgd(lr) : "sgd"),
  // (lr) => (lr ? tf.train.sgd(lr) : "sgd"),
];

const loss = ["meanSquaredError"];

// const units = [10, 30, 50, 80];
const units = [80];

// Generate cartesian product of given iterables:
function* cartesian(head, ...tail) {
  let remainder = tail.length ? cartesian(...tail) : [[]];
  for (let r of remainder) for (let h of head) yield [h, ...r];
}

const hyperparams = [...cartesian(optimizer, learningRate, loss, activation, units)];

module.exports = { optimizer, loss, activation, learningRate, hyperparams };

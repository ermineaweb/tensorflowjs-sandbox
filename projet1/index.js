const tf = require("@tensorflow/tfjs-node-gpu");
const { testData, trainData } = require("./data");

function run() {
  // create tensors
  const trainTensors = {
    sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(trainData.timeSec, [20, 1]),
  };
  const testTensors = {
    sizeMB: tf.tensor2d(testData.sizeMB, [6, 1]),
    timeSec: tf.tensor2d(testData.timeSec, [6, 1]),
  };

  // create model
  const model = tf.sequential();
  // create hidden layer
  model.add(tf.layers.dense({ inputShape: [1], units: 50, activation: "sigmoid" }));
  model.add(tf.layers.dense({ inputShape: [50], units: 1 }));
  // compile the model
  model.compile({ optimizer: "adam", loss: "meanSquaredError" });

  (async () => {
    // train the model
    await model.fit(trainTensors.sizeMB, trainTensors.timeSec, { epochs: 50 });
    // evaluate the model with test data
    model.evaluate(testTensors.sizeMB, testTensors.timeSec).print();
    // predict on new data
    // 0.435 - 0744
    model.predict(tf.tensor2d([[5], [10]])).print();
  })();
}

module.exports = run;

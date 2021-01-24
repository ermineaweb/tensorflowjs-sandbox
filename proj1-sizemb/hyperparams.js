const tf = require("@tensorflow/tfjs-node-gpu");
const { HyperParams, optimizers } = require("../HyperParams");
const { testData, trainData } = require("./data");

async function run() {
  // Generate data
  const train = {
    sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(trainData.timeSec, [20, 1]),
  };
  const test = {
    sizeMB: tf.tensor2d(testData.sizeMB, [6, 1]),
    timeSec: tf.tensor2d(testData.timeSec, [6, 1]),
  };
  // Search best hyperparams
  const hyperParams = new HyperParams(
    { input: train.sizeMB, label: train.timeSec, inputSize: 1 },
    {
      epochs: [100, 200, 400, 600],
      units: [15, 20, 25, 30, 35, 40, 45],
      loss: ["meanSquaredError"],
      optimizer: [optimizers.adam, optimizers.sgd],
    }
  );
  await hyperParams.findHyperParams();
  const model = hyperParams.getModel();
  console.log(hyperParams.getHyperparams());

  // Evaluation
  const eval = await model.evaluate(test.sizeMB, test.timeSec);
  eval.print();

  // Prediction
  model.predict(tf.tensor2d([[5]])).print(); //0.435
  model.predict(tf.tensor2d([[10]])).print(); //0.744
}

module.exports = run;

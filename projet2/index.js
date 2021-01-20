const tf = require("@tensorflow/tfjs-node-gpu");

const trainDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/projet2/train.csv";
const testDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/projet2/test.csv";

async function run() {
  // We want to predict the column "medv", which represents a median value of a home (in $1000s), so we mark it as a label.
  const trainDataset = tf.data.csv(trainDataCsvUrl, { columnConfigs: { medv: { isLabel: true } } });
  const testDataset = tf.data.csv(testDataCsvUrl, { columnConfigs: { medv: { isLabel: true } } });

  // Number of features (remove label column)
  const numOfFeatures = (await trainDataset.columnNames()).length - 1;

  // Prepare data
  // Convert xs(features) and ys(labels) from object form (keyed by column name) to array form.
  const flattenedTrainDataset = trainDataset
    .map(({ xs, ys }) => ({ xs: Object.values(xs), ys: Object.values(ys) }))
    .batch(10);
  const flattenedTestDataset = testDataset
    .map(({ xs, ys }) => ({ xs: Object.values(xs), ys: Object.values(ys) }))
    .batch(10);

  // flattenedDataset.forEachAsync((res) => {
  //   console.log(res);
  // });

  // Define the model (best params with hyperparameters finder)
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [numOfFeatures], units: 80, activation: "relu" }));
  model.add(tf.layers.dense({ inputShape: [80], units: 80, activation: "relu" }));
  model.add(tf.layers.dense({ inputShape: [80], units: 1 }));
  model.compile({ optimizer: "adam", loss: "meanSquaredError" });

  await model.fitDataset(flattenedTrainDataset, {
    epochs: 400,
    callbacks: {
      // onTrainBegin: async (epoch, logs) => console.log(`Start training`),
      // onEpochBegin: async (epoch, logs) => console.log(`Start epoch : ${epoch}`),
      onEpochEnd: async (epoch, logs) => console.log(`End epoch : ${epoch} Loss : ${logs.loss}`),
      // onTrainEnd: async (epoch, logs) => console.log(`End training : ${epoch}`),
    },
  });
  // Evaluation
  const eval = await model.evaluateDataset(flattenedTestDataset);
  eval.print();

  // Prediction
  // [0.06129,20,3.33,1,0.4429,7.645,49.7,5.2119,5,216,14.9,3.01], ==> 46
  // [0.05425,0,4.05,0,0.51,6.315,73.4,3.3175,5,296,16.6,6.29], ==> 24.6
  // [2.63548,0,9.9,0,0.544,4.973,37.8,2.5194,4,304,18.4,12.64], ==> ,16.1
  model.predict(tf.tensor2d([0.06129, 20, 3.33, 1, 0.4429, 7.645, 49.7, 5.2119, 5, 216, 14.9, 3.01], [1, 12])).print();
  model.predict(tf.tensor2d([0.05425, 0, 4.05, 0, 0.51, 6.315, 73.4, 3.3175, 5, 296, 16.6, 6.29], [1, 12])).print();
  model.predict(tf.tensor2d([2.63548, 0, 9.9, 0, 0.544, 4.973, 37.8, 2.5194, 4, 304, 18.4, 12.64], [1, 12])).print();
}

module.exports = run;

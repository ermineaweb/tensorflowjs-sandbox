const tf = require("@tensorflow/tfjs-node-gpu");

const trainDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/titanic/train.csv";
const testDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/titanic/test.csv";

const optimizers = {
  adam: (lr) => tf.train.adam(lr),
  sgd: (lr) => tf.train.sgd(lr),
  rmsprop: (lr) => tf.train.rmsprop(lr),
};

const losses = ["meanSquaredError", "binaryCrossEntropy"];

const learningRates = [0.1, 0.01, 0.001, 0.0001, 0.00001];

async function run() {
  const { flattenedDataset: trainDataset, numOfFeatures } = await createDatasetFromCSV(trainDataCsvUrl);
  const model = createModel({ numOfFeatures, learningRate: 0.01 });
  const history = await trainModel({ model, trainDataset, epochs: 5 });
  console.log(history)

  // Evaluation
  // const eval = await model.evaluateDataset(flattenedTestDataset);
  // eval.print();

  // Prediction
  // model.predict(tf.tensor2d([892,3,"Kelly, Mr. James","male",34.5,0,0,330911,7.8292,,"Q"], [1, 12])).print();
}

function createModel({ numOfFeatures, learningRate }) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal",
    })
  );
  model.add(
    tf.layers.dense({
      inputShape: [50],
      units: 1,
    })
  );
  model.compile({
    optimizer: tf.train.sgd(learningRate),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function trainModel({ model, trainDataset, epochs }) {
  for (let i = 0; i <= epochs; i++) {
    await model.fitDataset(trainDataset, {
      epochs: 1,
      callbacks: {
        onEpochEnd: async (epoch, logs) => console.log(`End epoch : ${epoch} Loss : ${logs.loss}`),
        // onTrainEnd: async (epoch, logs) => console.log(`End training : ${epoch}`),
      },
    });
  }
}

async function createDatasetFromCSV(url) {
  // We want to predict the column "survived", which represents 0 = death, 1 = life
  const dataset = tf.data.csv(url, { columnConfigs: { Survived: { isLabel: true } } });
  // Number of features (remove label column)
  const numOfFeatures = (await dataset.columnNames()).length - 1;
  // Convert xs(features) and ys(labels) from object form (keyed by column name) to array form.
  const flattenedDataset = dataset.map(({ xs, ys }) => ({ xs: Object.values(xs), ys: Object.values(ys) })).batch(50);
  // if we want to show tensors
  // await flattenedDataset.forEachAsync((res) => {
  //   console.log(res);
  // });
  return { flattenedDataset, numOfFeatures };
}

async function findHyperparameters() {
  let bestModel = null;
  let loss = 0;
  const { flattenedDataset: trainDataset, numOfFeatures } = await createDatasetFromCSV(trainDataCsvUrl);
  for (const learningRate of learningRates) {
    const model = createModel({ numOfFeatures, learningRate });
    const history = await trainModel({ model, trainDataset, epochs: 1 });
  }
  console.log(history);
}

module.exports = run;

const tf = require("@tensorflow/tfjs-node-gpu");
const { hyperparams } = require("../hyperparams");

const trainDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/hyperparams/train.csv";
const testDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/hyperparams/test.csv";

async function run() {
  const { flattenedDataset, numOfFeatures } = await createDatasetFromCSV({ url: trainDataCsvUrl });
  const { flattenedTestDataset } = await createDatasetFromCSV({ url: testDataCsvUrl });
  const { params, model } = await findHyperparameters({ flattenedDataset, numOfFeatures, hyperparams });
  console.table(params);

  // Evaluation
  // const eval = await model.evaluateDataset(flattenedTestDataset);
  // eval.print();
  // test data
  // [0.06129,20,3.33,1,0.4429,7.645,49.7,5.2119,5,216,14.9,3.01], ==> 46
  // [0.05425,0,4.05,0,0.51,6.315,73.4,3.3175,5,296,16.6,6.29], ==> 24.6
  // [2.63548,0,9.9,0,0.544,4.973,37.8,2.5194,4,304,18.4,12.64], ==> ,16.1
  model.predict(tf.tensor2d([0.06129, 20, 3.33, 1, 0.4429, 7.645, 49.7, 5.2119, 5, 216, 14.9, 3.01], [1, 12])).print();
  model.predict(tf.tensor2d([0.05425, 0, 4.05, 0, 0.51, 6.315, 73.4, 3.3175, 5, 296, 16.6, 6.29], [1, 12])).print();
  model.predict(tf.tensor2d([2.63548, 0, 9.9, 0, 0.544, 4.973, 37.8, 2.5194, 4, 304, 18.4, 12.64], [1, 12])).print();
}

async function findHyperparameters({ flattenedDataset, numOfFeatures, hyperparams }) {
  let bestParams = { bestloss: 0, learningRate: null, optimizer: null, loss: null, units: null };
  let bestloss = 1000;
  let finalModel = null;

  for (const params of hyperparams) {
    const [optimizer, learningRate, loss, activation, units] = params;
    const model = createModel({ numOfFeatures, activation, learningRate, loss, optimizer: optimizer.fn, units });
    const history = await trainModel({ model, trainDataset: flattenedDataset, epochs: 400 });
    if (history.loss[0] < bestloss) {
      bestloss = history.loss[0];
      finalModel = model;
      bestParams = { ...bestParams, bestloss, learningRate, activation, optimizer: optimizer.label, loss, units };
    }
  }

  return { params: bestParams, model: finalModel };
}

function createModel({ numOfFeatures, learningRate, activation, loss, optimizer, units }) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [numOfFeatures], units, activation }));
  model.add(tf.layers.dense({ inputShape: [units], units: 1 }));
  model.compile({
    optimizer: typeof optimizer === "function" ? optimizer(learningRate) : optimizer,
    loss,
  });

  return model;
}

async function trainModel({ model, trainDataset, epochs }) {
  let history = null;
  for (let i = 0; i <= epochs; i++) {
    history = await model.fitDataset(trainDataset, {
      epochs: 1,
      callbacks: {
        onEpochEnd: async (epoch, logs) => console.log(`End epoch : ${i} Loss : ${logs.loss}`),
        // onTrainEnd: async (epoch, logs) => console.log(`End training : ${epoch}`),
      },
    });
  }
  return history.history;
}

async function createDatasetFromCSV({ url, labelColumn }) {
  // We want to predict the column "survived", which represents 0 = death, 1 = life
  const dataset = tf.data.csv(url, { columnConfigs: { medv: { isLabel: true } } });
  // Number of features (remove label column)
  const numOfFeatures = (await dataset.columnNames()).length - 1;
  // Convert xs(features) and ys(labels) from object form (keyed by column name) to array form.
  const flattenedDataset = dataset.map(({ xs, ys }) => ({ xs: Object.values(xs), ys: Object.values(ys) })).batch(10);
  // if we want to show tensors
  // await flattenedDataset.forEachAsync((res) => {
  //   console.log(res);
  // });
  return { flattenedDataset, numOfFeatures };
}

module.exports = run;

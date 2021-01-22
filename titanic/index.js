const tf = require("@tensorflow/tfjs-node-gpu");
const HyperParamsTitanic = require("./HyperParamsTitanic");
const { optimizers } = require("../HyperParams");

const trainDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/titanic/train.csv";
const testDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/titanic/test.csv";

async function run() {
  const { dataSet: trainDataset, numOfFeatures } = await createDatasetFromCSV(trainDataCsvUrl);
  const { dataSet: testDataset } = await createDatasetFromCSV(trainDataCsvUrl);

  // Search best hyperparams
  const hyperParams = new HyperParamsTitanic(
    { dataset: trainDataset, inputSize: 6 },
    {
      epochs: [120, 160],
      units: [50],
      activation: ["sigmoid"],
      loss: ["meanSquaredError"],
      // optimizer: [optimizers.adam, optimizers.sgd],
    }
  );
  /*
  {
  bestloss: 0.11855467408895493,
  learningRate: null,
  optimizer: 'adam',
  loss: 'meanSquaredError',
  units: 50,
  epochs: 160,
  activation: 'sigmoid'
}
*/

  await hyperParams.findHyperParams();
  const model = hyperParams.getModel();
  console.log(hyperParams.getHyperparams());

  // const model = tf.sequential();
  // model.add(tf.layers.dense({ inputShape: [numOfFeatures], units: 15, activation: "relu" }));
  // model.add(tf.layers.dense({ units: 15, activation: "relu" }));
  // model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  // model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });
  // await model.fitDataset(trainDataset, {
  //   epochs: 10,
  //   callbacks: {
  //     onEpochEnd: async (epoch, logs) => console.log(`Epoch : ${epoch} Loss : ${logs.loss}`),
  //   },
  // });

  // Evaluation
  const eval = await model.evaluateDataset(testDataset, {});

  // Prediction
  model.predict(tf.tensor2d([3, 1, 34.5, 0, 0, 7.8292], [1, 6])).print();
}

async function createDatasetFromCSV(url) {
  // We want to predict the column "survived", which represents 0 = death, 1 = life
  const dataset = tf.data.csv(url, { columnConfigs: { Survived: { isLabel: true } } });
  // Number of features (remove label column)
  // const numOfFeatures = (await dataset.columnNames()).length - 1;
  const numOfFeatures = 6;
  // Convert xs(features) and ys(labels) from object form (keyed by column name) to array form.
  const flattenedDataset = dataset
    .shuffle(10)
    // remove undefined AGE
    .filter(({ xs }) => xs.Age)
    // remap data
    .map(({ xs, ys }) => {
      const { PassengerId, Name, Ticket, Cabin, Embarked, ...oldXs } = xs;
      const newXs = { ...oldXs, Sex: oldXs.Sex === "male" ? 0 : 1 };
      return { xs: Object.values(newXs), ys: Object.values(ys) };
    })
    .batch(10);
  return { dataSet: flattenedDataset, numOfFeatures };
}

module.exports = run;

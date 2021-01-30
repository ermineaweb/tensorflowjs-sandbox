const tf = require("@tensorflow/tfjs-node-gpu");
const fs = require("fs");
const HyperParamsHousePrices = require("./HyperParamsHousePrices");
const { createTrainData, createTestData } = require("./generateData");
const { optimizers } = require("../../HyperParams");

const trainDataCsvUrl =
  "file:///home/romain/workspace/tensorflowjs/kaggle/houseprices/data/train.csv";
const testDataCsvUrl =
  "file:///home/romain/workspace/tensorflowjs/kaggle/houseprices/data/test.csv";

async function run() {
  const { dataSet: trainDataset, numOfFeatures } = await createTrainData(trainDataCsvUrl);
  // const { dataSet: validateDataset } = await createDatasetFromCSV(trainDataCsvUrl, {
  //   validate: true,
  // });
  // const { dataSet: testDataset } = await createDatasetFromCSV(testDataCsvUrl, { test: true });

  // trainDataset.forEachAsync((t) => {
  //     console.log(t.xs.arraySync())
  // });
  // Search best hyperparams
  const hyperParams = new HyperParamsHousePrices(
    { dataset: trainDataset, inputSize: numOfFeatures },
    {
      epochs: [200],
      units: [60],
      activation: ["relu"],
      // meanAbsoluteError
      loss: ["meanSquaredError", "meanAbsoluteError"],
      learningRate: [0.1,0.01,0.001],
      // optimizer: [optimizers.sgd, optimizers.adam],
    }
  );
  await hyperParams.findHyperParams();
  const model = hyperParams.getModel();
  console.log(hyperParams.getHyperparams());

  // const model = tf.sequential();
  // model.add(tf.layers.dense({ inputShape: [numOfFeatures], units: 100, activation: "relu" }));
  // model.add(tf.layers.dense({ units: 100, activation: "relu" }));
  // model.add(tf.layers.dense({ units: 50, activation: "relu" }));
  // model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  // model.compile({
  //   optimizer: "adam",
  //   loss: "binaryCrossentropy",
  //   // learningRate: 0.001,
  //   metrics: ["accuracy"],
  // });
  // await model.fitDataset(trainDataset, {
  //   epochs: 600,
  //   validationData: validateDataset,
  //   callbacks: {
  //     onEpochEnd: async (epoch, logs) => {
  //       console.log(`Epoch:${epoch} Loss:${logs.loss} Acc:${logs.acc}`);
  //     },
  //   },
  // });
  //
  // let n = 892;
  // let out = `PassengerId,Survived\r\n`;
  // // Prediction
  // await testDataset.forEachAsync((t) => {
  //   console.log(t);
  //   const res = model.predict(t).arraySync()[0] > 0.5 ? 1 : 0;
  //   out += `${n++},${res}\r\n`;
  // });
  //
  // fs.writeFile(__dirname + "/res.csv", out, (err) => {
  //   console.log(err || "done");
  // });
}

module.exports = run;

const tf = require("@tensorflow/tfjs-node-gpu");
const fs = require("fs");
const HyperParamsTitanic = require("./HyperParamsTitanic");
const { createTrainData, createTestData } = require("./generateData");

const trainDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/kaggle/titanic/data/train.csv";
const testDataCsvUrl = "file:///home/romain/workspace/tensorflowjs/kaggle/titanic/data/test.csv";

async function run() {
  const { dataSet: trainDataset, numOfFeatures } = await createTrainData(trainDataCsvUrl);
  // const { dataSet: validateDataset } = await createDatasetFromCSV(trainDataCsvUrl, {
  //   validate: true,
  // });
  const { dataSet: testDataset } = await createTestData(testDataCsvUrl);

  // testDataset.forEachAsync((t) => {
  //   console.log(t);
  // });
  // Search best hyperparams
  // const hyperParams = new HyperParamsTitanic(
  //   { dataset: trainDataset, inputSize: numOfFeatures },
  //   {
  //     epochs: [200],
  //     units: [120],
  //     activation: ["relu"],
  //     //binaryCrossentropy meanSquaredError
  //     loss: ["binaryCrossentropy"],
  //     learningRate: [0.01],
  //     // optimizer: [optimizers.adam, optimizers.sgd],
  //   }
  // );
  // await hyperParams.findHyperParams();
  // const model = hyperParams.getModel();
  // console.log(hyperParams.getHyperparams());

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [numOfFeatures], units: 80, activation: "relu" }));
  model.add(tf.layers.dense({ units: 80, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    learningRate: 0.01,
    metrics: ["accuracy"],
  });
  await model.fitDataset(trainDataset, {
    epochs: 250,
    // validationData: validateDataset,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch:${epoch} Loss:${logs.loss} Acc:${logs.acc}`);
      },
    },
  });

  let n = 892;
  let out = `PassengerId,Survived\r\n`;
  // Prediction
  await testDataset.forEachAsync((t) => {
    const res = model.predict(t).arraySync()[0] > 0.5 ? 1 : 0;
    out += `${n++},${res}\r\n`;
  });

  fs.writeFile(__dirname + "/res.csv", out, (err) => {
    console.log(err || "done");
  });
}

module.exports = run;

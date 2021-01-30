const tf = require("@tensorflow/tfjs-node-gpu");

const mapValue = (val) => {
  return {
    ...val,
    Sex: val.Sex === "male" ? 0 : 1,
    Age: val.Age ? val.Age : 29,
    Fare: Math.log1p(val.Fare),
    // Name: findTitle(val.Name),
  };
};

async function createTrainData(url) {
  const columnConfigs = {
    Survived: { isLabel: true },
    Pclass: { isLabel: false },
    Sex: { isLabel: false },
    Age: { isLabel: false },
    SibSp: { isLabel: false },
    Parch: { isLabel: false },
    Fare: { isLabel: false },
    // Name: { isLabel: false },
  };
  const numOfFeatures = Object.values(columnConfigs).length - 1;
  const dataset = tf.data.csv(url, {
    columnConfigs,
    configuredColumnsOnly: true,
  });

  const flattenedDataset = dataset
    // .shuffle(1000)
    .map(({ xs, ys }) => {
      return { xs: Object.values(mapValue(xs)), ys: Object.values(ys) };
    })
    .batch(20)
    .map(({ xs, ys }) => {
      return {
        xs: xs.sub(xs.min()).div(xs.max().sub(xs.min())),
        ys,
      };
    });

  // flattenedDataset.forEachAsync((d) => {
  //   console.log(d.xs.arraySync());
  // });

  return { dataSet: flattenedDataset, numOfFeatures };
}

async function createTestData(url) {
  const columnConfigs = {
    // Survived: { isLabel: true },
    Pclass: { isLabel: false },
    Sex: { isLabel: false },
    Age: { isLabel: false },
    SibSp: { isLabel: false },
    Parch: { isLabel: false },
    Fare: { isLabel: false },
    // Name: { isLabel: false },
  };
  const numOfFeatures = Object.values(columnConfigs).length - 1;
  const dataset = tf.data.csv(url, {
    columnConfigs,
    configuredColumnsOnly: true,
  });

  const flattenedDataset = dataset
    // .shuffle(1000)
    .map((xs) => {
      return Object.values(mapValue(xs));
    })
    .batch(1)
    .map((xs) => {
      return xs.sub(xs.min()).div(xs.max().sub(xs.min()));
    });

  return { dataSet: flattenedDataset, numOfFeatures };
}

function findTitle(name) {
  switch (true) {
    case name.search(/mr\./i) > -1:
      return 1;
    case name.search(/master\./i) > -1:
      return 2;
    case name.search(/col\./i) > -1:
      return 3;
    case name.search(/miss\./i) > -1:
      return 4;
    case name.search(/mrs\./i) > -1:
      return 5;
    case name.search(/dr\./i) > -1:
      return 6;
    default:
      return 0;
  }
}

function normalizeMeanStd(value, mean, std) {
  return (value - mean) / std;
}

function normalizeMinMax(value, min, max) {
  return (value - min) / (max - min);
}

async function meanAndStdDevOfDatasetRow(dataset, columnName) {
  let totalSamples = 0;
  let sum = 0;
  let mean = 0;
  let squareDiffFromMean = 0;

  await dataset.forEachAsync(({ xs, ys }) => {
    const x = xs[columnName];
    if (x != null || x !== undefined) {
      totalSamples += 1;
      sum += x;
      mean = sum / totalSamples;
      squareDiffFromMean = (mean - x) * (mean - x);
    }
  });

  const variance = squareDiffFromMean / totalSamples;
  const std = Math.sqrt(variance);

  return { mean, std };
}

async function minMax(dataset, columnName) {
  let min = 0;
  let max = 0;

  await dataset.forEachAsync(({ xs, ys }) => {
    if (xs[columnName] < min) min = xs[columnName];
    if (xs[columnName] > max) max = xs[columnName];
  });

  return { min, max };
}

module.exports = { createTrainData, createTestData };

const tf = require("@tensorflow/tfjs-node-gpu");

async function createDatasetFromCSV(url, { test, validate } = { test: false, validate: false }) {
  // const numOfFeatures = (await dataset.columnNames()).length - 1;
  const numOfFeatures = 9;
  const configColumns = {
    Survived: { isLabel: true },
    Pclass: { isLabel: false },
    Sex: { isLabel: false },
    Age: { isLabel: false },
    SibSp: { isLabel: false },
    Parch: { isLabel: false },
    Fare: { isLabel: false },
    Name: { isLabel: false },
  };

  const mapXs = (xs, min, max, meanAge, std) => {
    return {
      Sex: xs.Sex === "male" ? 0 : 1,
      Fare: Math.log1p(xs.Fare),
      Age: xs.Age
        ? normalizeMeanStd(xs.Age, meanAge, std)
        : normalizeMeanStd(meanAge, meanAge, std),
      SibSp: xs.SibSp,
      Parch: xs.Parch,
      Name: findTitle(xs.Name),
      Pclass1: xs.Pclass === 1 ? 1 : 0,
      Pclass2: xs.Pclass === 2 ? 1 : 0,
      Pclass3: xs.Pclass === 3 ? 1 : 0,
    };
  };

  if (test) {
    const { Survived, ...columnConfigs } = configColumns;
    const dataset = tf.data.csv(url, {
      columnConfigs,
      configuredColumnsOnly: true,
    });

    const { mean: meanAge, std } = await meanAndStdDevOfDatasetRowTest(dataset, "Age");
    const { min, max } = await minMaxTest(dataset, "Age");

    const flattenedDataset = dataset
      // .take(3)
      .map((xs) => Object.values(mapXs(xs, min, max, meanAge, std)))
      .batch(1);

    return { dataSet: flattenedDataset, numOfFeatures };
  }

  const dataset = tf.data.csv(url, {
    columnConfigs: configColumns,
    configuredColumnsOnly: true,
  });

  const { mean: meanAge, std } = await meanAndStdDevOfDatasetRow(dataset, "Age");
  const { min, max } = await minMax(dataset, "Age");

  // Convert xs(features) and ys(labels) from object form (keyed by column name) to array form.
  const flattenedDataset = dataset
    .shuffle(1000)
    .skip(validate ? 700 : 0)
    .map(({ xs, ys }) => ({
      xs: Object.values(mapXs(xs, min, max, meanAge, std)),
      ys: Object.values(ys),
    }))
    .batch(32);

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

async function minMaxTest(dataset, columnName) {
  let min = 0;
  let max = 0;

  await dataset.forEachAsync((xs) => {
    if (xs[columnName] < min) min = xs[columnName];
    if (xs[columnName] > max) max = xs[columnName];
  });

  return { min, max };
}

async function meanAndStdDevOfDatasetRowTest(dataset, columnName) {
  let totalSamples = 0;
  let sum = 0;
  let mean = 0;
  let squareDiffFromMean = 0;

  await dataset.forEachAsync((xs) => {
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

function normalizeDataset() {
  let sampleSoFar = 0;
  let sumSoFar = 0;
  return (x) => {
    sampleSoFar += 1;
    sumSoFar += x;
    const estimatedMean = sumSoFar / sampleSoFar;
    return x - estimatedMean;
  };
}

module.exports = createDatasetFromCSV;

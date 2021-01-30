const tf = require("@tensorflow/tfjs-node-gpu");
const {
  MSZoning,
  Street,
  Alley,
  LotShape,
  LandContour,
  Utilities,
  LotConfig,
  LandSlope,
  RoofStyle,
} = require("./mapColumns");

const configColumns = {
  SalePrice: { isLabel: true },
  // MSSubClass: { isLabel: false },
  // MSZoning: { isLabel: false },
  // // LotFrontage: { isLabel: false },
  // LotArea: { isLabel: false },
  // Street: { isLabel: false },
  // Alley: { isLabel: false },
  // LotShape: { isLabel: false },
  // LandContour: { isLabel: false },
  // Utilities: { isLabel: false },
  // LotConfig: { isLabel: false },
  // LandSlope: { isLabel: false },
  // // ...
  // OverallQual: { isLabel: false },
  // OverallCond: { isLabel: false },
  // YearBuilt: { isLabel: false },
  // YearRemodAdd: { isLabel: false },
  // RoofStyle: { isLabel: false },
  // //...
  // firstFlrSF: { isLabel: false },
  // secondFlrSF: { isLabel: false },
  // LowQualFinSF: { isLabel: false },
  // GrLivArea: { isLabel: false },
  // //...
  // TotRmsAbvGrd: { isLabel: false },
  // //...
  // Fireplaces: { isLabel: false },
  // //...
  // GarageArea: { isLabel: false },
  // //...
  // PoolArea: { isLabel: false },
  TotalBsmtSF:{isLabel: false}
};

const mapValue = (val) => {
  return {
    ...val,
    // MSSubClass: val.MSSubClass,
    // MSZoning: MSZoning(val.MSZoning),
    // // LotFrontage: xs.LotFrontage,
    // LotArea: val.LotArea,
    // Street: Street(val.Street),
    // Alley: Alley(val.Alley),
    // LotShape: LotShape(val.LotShape),
    // LandContour: LandContour(val.LandContour),
    // Utilities: Utilities(val.Utilities),
    // LotConfig: LotConfig(val.LotConfig),
    // LandSlope: LandSlope(val.LandSlope),
    // // ...
    // OverallQual: val.OverallQual,
    // OverallCond: val.OverallCond,
    // YearBuilt: val.YearBuilt,
    // YearRemodAdd: val.YearRemodAdd,
    // RoofStyle: RoofStyle(val.RoofStyle),
    // //...
    // firstFlrSF: val.firstFlrSF,
    // secondFlrSF: val.secondFlrSF,
    // LowQualFinSF: val.LowQualFinSF,
    // GrLivArea: val.GrLivArea,
    // //...
    // TotRmsAbvGrd: val.TotRmsAbvGrd,
    // //...
    // Fireplaces: val.Fireplaces,
    // //...
    // GarageArea: val.GarageArea,
    // //...
    // PoolArea: val.PoolArea,
  };
};

const numOfFeatures = Object.values(configColumns).length - 1;

async function createTrainData(url) {
  const dataset = tf.data.csv(url, {
    columnConfigs: { ...configColumns },
    configuredColumnsOnly: true,
  });

  const flattenedDataset = dataset
    // .shuffle(1000)
    .map(({ xs, ys }) => {
      return { xs: Object.values(mapValue(xs)), ys: Object.values(ys) };
    })
    .batch(30)
    .map(({ xs, ys }) => {
      return {
        xs: xs.sub(xs.min()).div(xs.max().sub(xs.min())),
        ys,
      };
    });

  return { dataSet: flattenedDataset, numOfFeatures };
}

async function createTestData(url) {
  const { SalePrice, ...columnConfigs } = configColumns;

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

module.exports = { createTrainData, createTestData };

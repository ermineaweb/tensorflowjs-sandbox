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

async function createDatasetFromCSV(url, { test, validate } = { test: false, validate: false }) {
  const configColumns = {
    SalePrice: { isLabel: true },
    MSSubClass: { isLabel: false },
    MSZoning: { isLabel: false },
    // LotFrontage: { isLabel: false },
    LotArea: { isLabel: false },
    Street: { isLabel: false },
    Alley: { isLabel: false },
    LotShape: { isLabel: false },
    LandContour: { isLabel: false },
    Utilities: { isLabel: false },
    LotConfig: { isLabel: false },
    LandSlope: { isLabel: false },
    // ...
    OverallQual: { isLabel: false },
    OverallCond: { isLabel: false },
    YearBuilt: { isLabel: false },
    YearRemodAdd: { isLabel: false },
    RoofStyle: { isLabel: false },
    //...
    firstFlrSF: { isLabel: false },
    secondFlrSF: { isLabel: false },
    LowQualFinSF: { isLabel: false },
    GrLivArea: { isLabel: false },
    //...
    TotRmsAbvGrd: { isLabel: false },
    //...
    Fireplaces: { isLabel: false },
    //...
    GarageArea: { isLabel: false },
    //...
    PoolArea: { isLabel: false },
  };
  const numOfFeatures = Object.values(configColumns).length - 1;

  const mapXs = (xs, min, max, meanAge, std) => {
    return {
      MSSubClass: xs.MSSubClass,
      MSZoning: MSZoning(xs.MSZoning),
      // LotFrontage: xs.LotFrontage,
      LotArea: xs.LotArea,
      Street: Street(xs.Street),
      Alley: Alley(xs.Alley),
      LotShape: LotShape(xs.LotShape),
      LandContour: LandContour(xs.LandContour),
      Utilities: Utilities(xs.Utilities),
      LotConfig: LotConfig(xs.LotConfig),
      LandSlope: LandSlope(xs.LandSlope),
      // ...
      OverallQual: xs.OverallQual,
      OverallCond: xs.OverallCond,
      YearBuilt: xs.YearBuilt,
      YearRemodAdd: xs.YearRemodAdd,
      RoofStyle: RoofStyle(xs.RoofStyle),
      //...
      firstFlrSF: xs.firstFlrSF,
      secondFlrSF: xs.secondFlrSF,
      LowQualFinSF: xs.LowQualFinSF,
      GrLivArea: xs.GrLivArea,
      //...
      TotRmsAbvGrd: xs.TotRmsAbvGrd,
      //...
      Fireplaces: xs.Fireplaces,
      //...
      GarageArea: xs.GarageArea,
      //...
      PoolArea: xs.PoolArea,
    };
  };

  if (test) {
    const { SalePrice, ...columnConfigs } = configColumns;
    const dataset = tf.data.csv(url, {
      columnConfigs,
      configuredColumnsOnly: true,
    });

    // const { mean: meanAge, std } = await meanAndStdDevOfDatasetRowTest(dataset, "Age");
    // const { min, max } = await minMaxTest(dataset, "Age");

    const flattenedDataset = dataset
      // .take(3)
      .map((xs) => Object.values(mapXs(xs)))
      .batch(1);

    return { dataSet: flattenedDataset, numOfFeatures };
  }

  const dataset = tf.data.csv(url, {
    columnConfigs: configColumns,
    configuredColumnsOnly: true,
  });

  // const { mean: meanAge, std } = await meanAndStdDevOfDatasetRow(dataset, "Age");
  // const { min, max } = await minMax(dataset, "Age");

  const flattenedDataset = dataset
    .shuffle(3000)
    // .skip(validate ? 700 : 0)
    .map(({ xs, ys }) => ({
      xs: Object.values(mapXs(xs)),
      ys: Object.values(ys),
    }))
    .batch(30);

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

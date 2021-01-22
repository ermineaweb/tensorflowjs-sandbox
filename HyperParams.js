const tf = require("@tensorflow/tfjs-node-gpu");

const optimizers = {
  adam: { label: "adam", fn: (lr) => (lr ? tf.train.adam(lr) : "adam") },
  sgd: { label: "sgd", fn: (lr) => (lr ? tf.train.sgd(lr) : "sgd") },
};

class HyperParams {
  constructor(
    { input, label, inputSize, dataset },
    {
      learningRate = [null],
      activation = ["relu"],
      optimizer = [optimizers.adam],
      loss = ["meanSquaredError"],
      units = [30],
      epochs = [40],
    }
  ) {
    this.learningRate = learningRate;
    this.activation = activation;
    this.optimizer = optimizer;
    this.loss = loss;
    this.units = units;
    this.epochs = epochs;
    this.model = null;
    this.dataset = dataset;
    this.input = input;
    this.label = label;
    this.inputSize = inputSize;
    this.hyperparams = {
      bestloss: 0,
      learningRate: null,
      optimizer: null,
      loss: null,
      units: null,
      epochs: null,
    };
  }

  createModel({ numOfFeatures, learningRate, activation, loss, optimizer, units }) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [numOfFeatures], units, activation }));
    model.add(tf.layers.dense({ inputShape: [units], units: 1 }));
    model.compile({
      optimizer: typeof optimizer === "function" ? optimizer(learningRate) : optimizer,
      loss,
    });
    return model;
  }

  async trainModel({ model, epochs }) {
    let history = null;
    for (let i = 1; i <= epochs; i++) {
      console.log(`Epoch n° ${i} / ${epochs}`);
      if (this.input && this.label) {
        history = await model.fit(this.input, this.label, { epochs: 1 });
      }
      if (this.dataset) {
        history = await model.fitDataset(this.dataset, { epochs: 1 });
      }
    }
    return history.history;
  }

  async findHyperParams() {
    let bestloss = 10000;

    for (const params of this.generateCombinations()) {
      const [optimizer, learningRate, loss, activation, units, epochs] = params;
      const model = this.createModel({
        numOfFeatures: this.inputSize,
        learningRate,
        activation,
        loss,
        optimizer: optimizer.fn,
        units,
      });
      const history = await this.trainModel({ model, epochs });
      if (history.loss[0] < bestloss) {
        bestloss = history.loss[0];
        this.model = model;
        this.hyperparams = {
          ...this.hyperparams,
          bestloss,
          learningRate,
          activation,
          optimizer: optimizer.label,
          loss,
          units,
          epochs,
        };
      }
    }
  }

  *cartesian(head, ...tail) {
    let remainder = tail.length ? this.cartesian(...tail) : [[]];
    for (let r of remainder) for (let h of head) yield [h, ...r];
  }

  generateCombinations() {
    return [
      ...this.cartesian(
        this.optimizer,
        this.learningRate,
        this.loss,
        this.activation,
        this.units,
        this.epochs
      ),
    ];
  }

  getModel() {
    return this.model;
  }

  getHyperparams() {
    return this.hyperparams;
  }

  getOptimizers() {
    return optimizers;
  }
}

module.exports = { HyperParams, optimizers };

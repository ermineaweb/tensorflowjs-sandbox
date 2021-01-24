const tf = require("@tensorflow/tfjs-node-gpu");
const { optimizers } = require("../HyperParams");
const { HyperParams } = require("../HyperParams");

class HyperParamsTitanic extends HyperParams {
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
    super(
      { input, label, inputSize, dataset },
      {
        learningRate,
        activation,
        optimizer,
        loss,
        units,
        epochs,
      }
    );
  }

  createModel({ numOfFeatures, learningRate, activation, loss, optimizer, units }) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [numOfFeatures], units, activation }));
    model.add(tf.layers.dense({ units, activation }));
    // care with classification we need last layer activation sigmoid
    model.add(tf.layers.dense({ activation: "sigmoid", units: 1 }));
    model.compile({
      optimizer: typeof optimizer === "function" ? optimizer(learningRate) : optimizer,
      loss,
      metrics: ["accuracy"],
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
    return history.history.acc[0];
  }
}

module.exports = HyperParamsTitanic;

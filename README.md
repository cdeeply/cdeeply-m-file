# cdeeply-m-file
MATLAB / Octave interface to C Deeply's neural network generators.

Put cdeeply_neural_network.m into a reachable directory, then:

1) Define a class instance, e.g. `myNN = cdeeply_neural_network`.
2) Call `myNN.tabular_regressor(...)` or `myNN.tabular_encoder(...)` to train a neural network in supervised or unsupervised mode.  *This step requires an internet connection*! as the training is done server-side.
3) Call `myNN.runSample(...)` as many times as you want to process new data samples -- one sample per function call.

**Function definitions:**

`sampleOutputs = tabular_regressor(trainingSamples, sampleTableTranspose,`  
`        outputRowOrColumnList, importances,`  
`        maxWeights, maxHiddenNeurons, maxLayers, maxWeightDepth, maxActivationRate,`  
`        maxWeightsHardLimit, maxHiddenNeuronsHardLimit, maxActivationsHardLimit, allowedAFs,`  
`        ifQuantizeWeights, wQuantBits, wQuantZeroInt, wQuantRange,`  
`        ifQuantizeActivations, yQuantBits, yQuantZeroInt, yQuantRange,`  
`        sparseWeights, allowNegativeWeights, ifNNhasBias, ifAllowingInputOutputConnections)`

Generates a x->y prediction network using *supervised* training on `trainingSamples`.
* `trainingSamples` is a matrix having dimensions `numFeatures` and `numSamples`, containing *both* inputs and target outputs.
  * Set `sampleTableTranspose` to `"FEATURE_SAMPLE_ARRAY"` for `trainingSamples(feature, sample)` array ordering, or `"SAMPLE_FEATURE_ARRAY"` for `trainingSamples(sample, feature)` array ordering.
  * The rows/columns in `trainingSamples` corresponding to the target outputs are specified by `outputRowOrColumnList`.
* The optional `importances` argument weights the cost function of the target outputs.  Pass as a matrix having dimensions `numTargetOutputs` and `numSamples` (ordered according to `sampleTableTranspose`), or `[]` if this parameter isn't being used.
* Optional parameters `maxWeights`, `maxHiddenNeurons` and `maxLayers` limit the size of the neural network, and `maxWeightDepth` limits the depth of layer-to-layer connections.  Set unused parameters to `"NO_MAX"`.  The corresponding `maxWeightsHardLimit` and `maxHiddenNeuronsHardLimit` parameters should be either `true` or `false`.
* `allowedAFs` is a length-5 Boolean vector corresponding to the allowed activation functions:  (step, ReLU, ReLU1, sigmoid, tanh).  Set each Boolean according to whether the activation function should be considered for a given layer of the network.
* `To quantize weights or neural activations, set `ifQuantizeWeights` or `ifQuantizeActivations` to `true` and give values to the following three fields; otherwise set the respective `ifQuantize` to `false`.
* Set `sparseWeights` to either `true` or `false`, depending on whether we are generating sparse weight matrices.
* The parameter `maxActivationRate` should be set to `1.` unless sparsifying the neural activations, in which case this can be a lower positive value.  Sparse activations do not work with sigmoid or tanh activation functions.  Set `maxActivationsHardLimit` to either `true` or `false`.
* The `allowNegativeWeights` and `ifNNhasBias` parameters are either `true` or `false`.  The bias is a constant input to each neuron.
* Set `ifAllowingInputOutputConnections` to `"ALLOW_IO_CONNECTIONS"` or `"NO_IO_CONNECTIONS"` depending on whether to allow the input layer to feed directly into the output layer.  (Outliers in new input data might cause wild outputs).
* `sampleOutputs` is a matrix having dimensions `numTargetOutputs` and `numSamples`, to which the training output *as calculated by the server* will be written.  This is mainly a check that the data went through the pipes OK.  If you don't care, ignore the return value.

`sampleOutputs = tabular_encoder(trainingSamples, sampleTableTranspose, importances,`  
`        ifDoEncoder, ifDoDecoder, numEncodingFeatures, numVariationalFeatures, variationalDist,`  
`        maxWeights, maxHiddenNeurons, maxLayers, maxWeightDepth, maxActivationRate,`  
`        maxWeightsHardLimit, maxHiddenNeuronsHardLimit, maxActivationsHardLimit, allowedAFs,`  
`        ifQuantizeWeights, wQuantBits, wQuantZeroInt, wQuantRange,`  
`        ifQuantizeActivations, yQuantBits, yQuantZeroInt, yQuantRange,`  
`        sparseWeights, allowNegativeWeights, ifNNhasBias)`

Generates an autoencoder (or an encoder or decoder) using *unsupervised* training on `trainingSamples`.
* `trainingSamples` is a matrix having dimensions `numFeatures` and `numSamples`, in the order determined by `sampleTableTranspose`.
* `sampleTableTranspose` and `importances` are set the same way as for `tabular_regressor(...)`.
* The size of the encoding is determined by `numEncodingFeatures`.
  * So-called variational features are extra randomly-distributed inputs used by the decoder, analogous to the extra degrees of freedom a variational autoencoder generates.
  * `variationalDist` is set to `"UNIFORM_DIST"` if the variational inputs are uniformly-(0, 1)-distributed, or `"NORMAL_DIST"` if they are normally distributed (zero mean, unit variance).
* Set `ifDoEncoder` to either `"DO_ENCODER"` or `"NO_ENCODER"`, the latter being for a decoder-only network.
* Set `ifDoDecoder` to either `"DO_DECODER"` or `"NO_DECODER"`, the latter being for an encoder-only network.
* The remaining parameters are set the same way as for `tabular_regressor(...)`.

`sampleOutput = runSample(sampleInput [, sampleVariationalInput])`

Runs the neural network on a *single* sample, and returns the network output.
* If it is an autoencoder or decoder with variational features, sample the variational features from the appropriate distribution and pass them as a second argument.
* The return value is simply a copy of the last layer of the network `myNN.y[myNN.numLayers]`.

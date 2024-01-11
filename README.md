# cdeeply-m-file
MATLAB / Octave interface to C Deeply's generators.

Put cdeeply_neural_network.m into a reachable directory, then:

1) Define a class instance, e.g. `myNN = cdeeply_neural_network`.
2) Call `myNN.tabular_regressor(...)` or `myNN.tabular_encoder(...)` to train a neural network in supervised or unsupervised mode.  *This step requires an internet connection*! as the training is done server-side.
3) Call `myNN.runSample(...)` as many times as you want to process new data samples -- one sample per function call.

**Function definitions:**

`sampleOutputs = tabular_regressor(numInputs, numTargetOutputs, numSamples,`  
`        trainingSamples, importances, sampleTableTranspose, outputRowOrColumnList,`  
`        maxWeights, maxHiddenNeurons, maxLayers, maxLayerSkips,`  
`        ifNNhasBias, ifAllowingInputOutputConnections)`

Generates a x->y prediction network using *supervised* training on `trainingSamples`.
* `trainingSamples` is a matrix having dimensions `(numInputs+numTargetOutputs)` and `numSamples`.
  * Set `sampleTableTranspose` to `"FEATURE_SAMPLE_ARRAY"` for `trainingSamples(input_output, sample)` array ordering, or `"SAMPLE_FEATURE_ARRAY"` for `trainingSamples(sample, input_output)` array ordering.
  * The rows/columns in `trainingSamples` corresponding to the target outputs are specified by `outputRowOrColumnList`.
* The optional `importances` argument weights the cost function of the target outputs.  Pass as a matrix having dimensions `numTargetOutputs` and `numSamples` (ordered according to `sampleTableTranspose`), or `[]` if this parameter isn't being used.
* Optional parameters `maxWeights`, `maxHiddenNeurons` and `maxLayers` limit the size of the neural network, and `maxLayerSkips` limits the depth of layer-to-layer connections.  Set unused parameters to `"NO_MAX"`.
* Set `ifNNhasBias` to `"HAS_BIAS"` unless you don't want to allow a bias (i.e. constant) term in each neuron's input, in which case set this to `"NO_BIAS"`.
* Set `ifAllowingInputOutputConnections` to `"ALLOW_IO_CONNECTIONS"` or `"NO_IO_CONNECTIONS"` depending on whether to allow the input layer to feed directly into the output layer.  (Outliers in new input data might cause wild outputs).
* `sampleOutputs` is a matrix having dimensions `numTargetOutputs` and `numSamples`, to which the training output *as calculated by the server* will be written.  This is mainly a check that the data went through the pipes OK.  If you don't care, ignore the return value.

`sampleOutputs = tabular_encoder(numInputs, numFeatures, numVariationalFeatures, numSamples,`  
`        trainingSamples, importances, sampleTableTranspose,`  
`        ifDoEncoder, ifDoDecoder, variationalDist,`  
`        maxWeights, maxHiddenNeurons, maxLayers, maxLayerSkips, ifNNhasBias)`

Generates an autoencoder (or an encoder or decoder) using *unsupervised* training on `trainingSamples`.
* `trainingSamples` is a matrix having dimensions `numInputs` and `numSamples`, in the order determined by `sampleTableTranspose`.
* `importances` and `sampleTableTranspose` are set the same way as for `tabular_regressor(...)`.
* The size of the encoding is determined by `numFeatures`.
  * So-called variational features are extra randomly-distributed inputs used by the decoder, analogous to the extra degrees of freedom a variational autoencoder generates.
  * `variationalDist` is set to `"UNIFORM_DIST"` if the variational inputs are uniformly-(0, 1)-distributed, or `"NORMAL_DIST"` if they are normally distributed (zero mean, unit variance).
* Set `ifDoEncoder` to either `"DO_ENCODER"` or `"NO_ENCODER"`, the latter being for a decoder-only network.
* Set `ifDoDecoder` to either `"DO_DECODER"` or `"NO_DECODER"`, the latter being for an encoder-only network.
* The last 5 parameters are set the same way as for `tabular_regressor(...)`.

`outputArray = runSample(inputArray [, variationalInputArray])`

Runs the neural network on a *single* input sample, and returns the network output.
* If it is an autoencoder or decoder with variational features, sample the variational features from the appropriate distribution and pass them as a second argument.
* The return value is simply a copy of the last layer of the network `myNN.y[myNN.numLayers]`.

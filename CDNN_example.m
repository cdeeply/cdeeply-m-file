numInputs = 10;
numSamples = 100;
NNtypes = { "encoder", "regressor" };
NN = cdeeply_neural_network;

for c2 = 1:2
    
    trainingData = zeros(numSamples, numInputs+(c2-1));
    for s = 1:numSamples
        iSum = 0;                               % the last row is just some function
        for io = 1:numInputs+(c2-1)
            trainingData(s, io) = rand();
            iSum = iSum + trainingData(s, io) * sin(io) / numInputs;
        end
        trainingData(s, end) = cos(iSum);
    end
    
    disp([ "Generating " NNtypes{c2} ])
    if c2 == 1
        sampleOutputs = NN.tabular_encoder( ...
                numInputs, 1, 0, numSamples, trainingData, [], "SAMPLE_FEATURE_ARRAY", ...
                true, true, "NORMAL_DIST", "NO_MAX", "NO_MAX", "NO_MAX", "NO_MAX", true);
    else
        sampleOutputs = NN.tabular_regressor(
                numInputs, 1, numSamples, trainingData, [], "SAMPLE_FEATURE_ARRAY", [ numInputs+1 ], ...
                "NO_MAX", "NO_MAX", "NO_MAX", "NO_MAX", true, true);
    end
    
    out1 = NN.runSample(trainingData(1, 1:numInputs))(1);
    if abs(out1-sampleOutputs(1)) > .0001
        error([ "  ** Network problem?  Sample 1 output was calculated as " ...
                        num2str(out1) " locally vs " num2str(sampleOutputs(1)) " by the server" ]);
    end
    
    disp([ "  Output on sample #1 was " num2str(out1) ])
end
        
        

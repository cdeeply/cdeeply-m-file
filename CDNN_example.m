numFeatures = 10;
numSamples = 100;
NNtypes = { "autoencoder with 1 latent feature", "regressor" };
NN = cdeeply_neural_network;


    % generate a training matrix that traces out some noisy curve in Nf-dimensional space (noise ~ 0.1)

dependentVar = rand(numSamples+1, 1);
trainTestMat = 0.1 * randn(numSamples+1, numFeatures);
for cf = 1:numFeatures
    featurePhase = 2*pi*rand(1);
    featureCurvature = 2*pi*rand(1);
    trainTestMat(:, cf) = trainTestMat(:, cf) + sin(featureCurvature*dependentVar + featurePhase);
end


for c2 = 1:2
    
    disp([ "Generating " NNtypes{c2} ])
    if c2 == 1
        outputsComputedByServer = NN.tabular_encoder( ...
                trainTestMat(1:numSamples, :), "SAMPLE_FEATURE_ARRAY", [], ...
                true, true, 1, 0, "NORMAL_DIST", "NO_MAX", "NO_MAX", "NO_MAX", "NO_MAX", true);
        firstSampleOutputs = NN.runSample(trainTestMat(1, :));
        testSampleOutputs = NN.runSample(trainTestMat(end, :));
    else
        outputsComputedByServer = NN.tabular_regressor(
                trainTestMat(1:numSamples, :), "SAMPLE_FEATURE_ARRAY", [ numFeatures ], [], ...
                "NO_MAX", "NO_MAX", "NO_MAX", "NO_MAX", true, true);
        firstSampleOutputs = NN.runSample(trainTestMat(1, 1:(numFeatures-1)));
        testSampleOutputs = NN.runSample(trainTestMat(end, 1:(numFeatures-1)));
    end
    
    if abs(firstSampleOutputs(1)-outputsComputedByServer(1)) > .0001
        error([ "  ** Network problem?  Sample 1 output was calculated as " ...
                        num2str(firstSampleOutputs(1)) " locally vs " num2str(outputsComputedByServer(1)) " by the server" ]);
    end
    
    if c2 == 1
        targetValue = trainTestMat(end, 1);
        targetDescription = "  Reconstructed test sample, feature 1";
    else
        targetValue = trainTestMat(end, numFeatures);
        targetDescription = "  Test sample output";
    end
    disp([ targetDescription " was " num2str(testSampleOutputs(1)) "; target value was " num2str(targetValue) ])
end
        
        

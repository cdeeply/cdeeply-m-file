numFeatures = 10;
numSamples = 100;
noiseAmplitude = 0.1;

NNtypes = { "autoencoder with 1 latent feature", "regressor" };


    % generate a training matrix that traces out some noisy curve in Nf-dimensional space (noise ~ 0.1)

disp("Training data along a 1D curve in feature space")
disp(["  * ", num2str(numSamples), " samples, ", num2str(numFeatures), " features; feature variance ~1 + Gaussian noise ~", num2str(noiseAmplitude)])
dependentVar = rand(numSamples+1, 1);
trainTestMat = noiseAmplitude * randn(numSamples+1, numFeatures);
for cf = 1:numFeatures
    featurePhase = 2*pi*rand(1);
    featureCurvature = 2*pi*rand(1);
    trainTestMat(:, cf) = trainTestMat(:, cf) + sin(featureCurvature*dependentVar + featurePhase);
end


NN = cdeeply_neural_network;

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
    
    
    if max(abs(firstSampleOutputs'-outputsComputedByServer(1, :))) > .0001
        error([ "  ** Network problem?  Sample 1 output was calculated as " ...
                        num2str(firstSampleOutputs') " locally vs " num2str(outputsComputedByServer(1, :)) " by the server" ]);
    end
    
    
        % run the network on the test sample
    
    if c2 == 1
        targetValue = trainTestMat(end, 1);
        targetDescription = "reconstructed feature 1";
    else
        targetValue = trainTestMat(end, numFeatures);
        targetDescription = "output";
    end
    disp([ "  Test sample:  " targetDescription " was " num2str(testSampleOutputs(1)) "; target value was " num2str(targetValue) ])
end
        
        

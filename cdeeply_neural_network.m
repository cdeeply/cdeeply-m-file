% 
% cdeeply.c - interfaces to neural network generator
% 
% C Deeply
% Copyright (C) 2023 C Deeply, LLC
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
% 

% usage:
% 
% 0) Create a class instance:
% 
% myNN = cdeeply_neural_network;
% 
% 
% 1) Generate a neural network, using either of:
% 
% myNN.tabular_regressor( trainingSamples, indexOrder, outputIndices, importances,
%               maxWeights, maxHiddenNeurons, maxLayers, maxLayerSkips, hasBias, allowIOconnections )
% 
% myNN.tabular_encoder( trainingSamples, indexOrder, importances,
%               doEncoder, doDecoder, numEncodingFeatures, numVariationalFeatures, variationalDistribution,
%               maxWeights, maxHiddenNeurons, maxLayers, maxLayerSkips, hasBias )
% 
% * indexOrder="SAMPLE_FEATURE_ARRAY" for trainingSamples[sampleNo][featureNo] indexing,
%       or "FEATURE_SAMPLE_ARRAY" for trainingSamples[featureNo][sampleNo] indexing.
% * For supervised x->y regression, the sample table contains BOTH 'x' and 'y', the latter specified by outputIndices[].
% * The importances table, if not empty, has dimensions numOutputFeatures and numSamples (ordered by indexOrder),
%       and weights the training cost function:  C = sum(Imp*dy^2).
% * Weight/neuron/etc limits are either positive integers or "NO_MAX".
% * variationalDistribution is either "UNIFORM_DIST" ([0, 1]) or "NORMAL_DIST" (mean=0, variance=1).
% * doEncoder, doDecoder, hasBias, and allowIOconnections are all Booleans.
% Both functions return the network outputs from the training data, if you care to check that it agrees with what's computed locally.
% 
% 
% 2) Run the network on a (single) new sample
% 
% oneSampleOutput = myNN.runSample(oneSampleInput [, oneSampleVariationalInput])
% 
% where oneSampleInput is a list of length numInputFeatures, and oneSampleOutput is a list of length numOutputFeatures.
% * If it's an autoencoder (encoder+decoder), length(oneSampleInput) and length(oneSampleOutput) equal the size of the training sample space.
%       If it's just an encoder, length(oneSampleOutput) equals numEncodingFeatures; if decoder only, length(oneSampleInput) must equal numEncodingFeatures.
% * If it's a decoder or autoencoder network having numVariationalFeatures > 0, then oneSampleVariationalInput is a list
%       of length numVariationalFeatures containing random numbers drawn from variationalDistribution.


classdef cdeeply_neural_network < handle
properties
    numLayers
    encoderLayer
    variationalLayer
    layerSize
    layerAFs
    layerInputs
    weights
    y
    
    fs = { ...
        @(x) x, ...
        @(x) x, ...
        @(x) x, ...
        @(x) x, ...
        @(x) tanh(x) };
end

methods (Access = public)
    
    function sampleOutputs = tabular_regressor(self, ...
            trainingSamples, indexOrder, outputIndices, importances, ...
            maxWeights, maxHiddenNeurons, maxLayers, maxLayerSkips, hasBias, allowIOconnections)
        
        [ numFeatures, numSamples ] = self.getDims(size(trainingSamples), indexOrder);
        numOutputs = length(outputIndices);
        numInputs = numFeatures - numOutputs;
        
        [ sampleString, rowcolString ] = self.CDNN_data2table(trainingSamples, numInputs+numOutputs, numSamples, indexOrder, 1);
        if length(importances) > 0
            importancesString = self.CDNN_data2table(trainingSamples, numInputs, numSamples, indexOrder, 1);
        else
            importancesString = "";
        end
        
        orcStrings = cell(size(outputIndices));
        for rc = 1:length(outputIndices)
            orcStrings{rc} = num2str(outputIndices(rc));
        end
        outputRowsColumnsString = strjoin(orcStrings, ',');
        
        response = urlread("https://cdeeply.com/myNN.php", "post", { ...
            "samples", sampleString, ...
            "importances", importancesString, ...
            "rowscols", rowcolString, ...
            "rowcolRange", outputRowsColumnsString, ...
            "maxWeights", self.maxString(maxWeights), ...
            "maxNeurons", self.maxString(maxHiddenNeurons), ...
            "maxLayers", self.maxString(maxLayers), ...
            "maxSkips", self.maxString(maxLayerSkips), ...
            "hasBias", self.ifChecked(hasBias), ...
            "allowIO", self.ifChecked(allowIOconnections), ...
            "submitStatus", "Submit", ...
            "NNtype", "regressor", ...
            "formSource", "MATLAB_API" });
    
        sampleOutputs = self.buildCDNN(response, numOutputs, numSamples, indexOrder);
    end
    
    
    function sampleOutputs = tabular_encoder(self, ...
            trainingSamples, indexOrder, importances, ...
            doEncoder, doDecoder, numEncodingFeatures, numVariationalFeatures, variationalDist, ...
            maxWeights, maxHiddenNeurons, maxLayers, maxLayerSkips, hasBias)
        
        [ numFeatures, numSamples ] = self.getDims(size(trainingSamples), indexOrder);
        
        [ sampleString, rowcolString ] = self.CDNN_data2table(trainingSamples, numFeatures, numSamples, indexOrder, 2);
        if length(importances) > 0
            importancesString = self.CDNN_data2table(trainingSamples, numFeatures, numSamples, indexOrder, 2);
        else
            importancesString = "";
        end
        
        if strcmp(variationalDist, "UNIFORM_DIST"), variationalDistStr = "uniform";
        elseif strcmp(variationalDist, "NORMAL_DIST"), variationalDistStr = "normal";
        else, error("Variational distribution must be either \"UNIFORM_DIST\" or \"NORMAL_DIST\"");
        end
        
        response = urlread("https://cdeeply.com/myNN.php", "post", { ...
            "samples", sampleString, ...
            "importances", importancesString, ...
            "rowscols", rowcolString, ...
            "numFeatures", num2str(numEncodingFeatures), ...
            "doEncoder", self.ifChecked(doEncoder), ...
            "doDecoder", self.ifChecked(doDecoder), ...
            "numVPs", num2str(numVariationalFeatures), ...
            "variationalDist", variationalDistStr, ...
            "maxWeights", self.maxString(maxWeights), ...
            "maxNeurons", self.maxString(maxHiddenNeurons), ...
            "maxLayers", self.maxString(maxLayers), ...
            "maxSkips", self.maxString(maxLayerSkips), ...
            "hasBias", self.ifChecked(hasBias), ...
            "submitStatus", "Submit", ...
            "NNtype", "autoencoder", ...
            "formSource", "MATLAB_API" });
        
        if doDecoder, numOutputs = numFeatures;
        else, numOutputs = numEncodingFeatures;
        end
        
        sampleOutputs = self.buildCDNN(response, numOutputs, numSamples, indexOrder);
    end
    
    
    function NNoutput = runSample(self, NNinput, NNvariationalInput)
        
        self.y{1}(:) = 1;
        self.y{2}(:) = NNinput;
        if self.variationalLayer > 0
            self.y{self.variationalLayer}(:) = NNvariationalInput;
        end
        
        for l = 3:self.numLayers
        if l != self.variationalLayer
            self.y{l}(:) = 0;
            for li = 1:length(self.layerInputs{l})
                l0 = self.layerInputs{l}(li);
                self.y{l} = self.y{l} + self.weights{l}{li}*self.y{l0};
            end
            self.y{l} = self.fs{self.layerAFs(l)}(self.y{l});
        end, end
        
        NNoutput = self.y{end};
    end
end


methods (Access = private)
    
    function checkStr = ifChecked(self, checkedBool)
        if checkedBool, checkStr = "on";
        else, checkStr = "";
        end
    end
    
    function maxStr = maxString(self, maxVar)
        if strcmp(maxVar, "NO_MAX"), maxStr = "";
        else, maxStr = num2str(maxVar);
        end
    end
    
    
    function [ numFeatures, numSamples ] = getDims(self, sampleTableSize, transpose)
        
        if (strcmp(transpose, "FEATURE_SAMPLE_ARRAY"))
            numFeatures = sampleTableSize(1);
            numSamples = sampleTableSize(2);
        elseif (strcmp(transpose, "SAMPLE_FEATURE_ARRAY"))
            numFeatures = sampleTableSize(2);
            numSamples = sampleTableSize(1);
        else
            error("transpose must be either \"FEATURE_SAMPLE_ARRAY\" or \"SAMPLE_FEATURE_ARRAY\"");
        end
    end
    
    
    function [ tableStr, rowcol ] = CDNN_data2table(self, data, numIOs, numSamples, transpose, NNtype)
        
        rowcolStrings = { "rows", "columns" };
        
        if (strcmp(transpose, "FEATURE_SAMPLE_ARRAY"))
            dim1 = numIOs;
            dim2 = numSamples;
            rowcol = rowcolStrings{NNtype};
        elseif (strcmp(transpose, "SAMPLE_FEATURE_ARRAY"))
            dim1 = numSamples;
            dim2 = numIOs;
            rowcol = rowcolStrings{3-NNtype};
        end
        
        rowElStrings = cell(1, dim2);
        tableRowStrings = cell(1, dim1);
        
        for i1 = 1:dim1
            for i2 = 1:dim2
                rowElStrings(i2) = num2str(data(i1, i2));
            end
            tableRowStrings(i1) = strjoin(rowElStrings, ',');
        end
        
        tableStr = strjoin(tableRowStrings, '\n');
    end
    
    
    function sampleOutputs = buildCDNN(self, NNdata, numOutputs, numSamples, transpose)
        
        firstChar = double(NNdata(1));
        if firstChar < double('0') || firstChar > double('9')
            error(NNdata);
            sampleOutputs = [];
            return;
        end
        
        NNdataRows = strsplit(NNdata, ";", "CollapseDelimiters", false);
        
        NNheader = strsplit(NNdataRows{1}, ",");
        self.numLayers = str2double(NNheader{1});
        self.encoderLayer = str2double(NNheader{2});
        self.variationalLayer = str2double(NNheader{3});
        
        self.layerSize = loadNumArray(NNdataRows{2});
        self.layerAFs = loadNumArray(NNdataRows{3})+1;
        numLayerInputs = loadNumArray(NNdataRows{4});
        
        allLayerInputs = loadNumArray(NNdataRows{5})+1;
        self.layerInputs = cell(1, self.numLayers);
        idx = 0;
        for l = 1:self.numLayers
            self.layerInputs{l} = allLayerInputs(idx+1:idx+numLayerInputs(l));
            idx = idx + numLayerInputs(l);
        end
        
        allWs = loadNumArray(NNdataRows{6});
        self.weights = cell(1, self.numLayers);
        idx = 0;
        for l = 1:self.numLayers
            self.weights{l} = cell(1, numLayerInputs(l));
            for li = 1:numLayerInputs(l)
                l0 = self.layerInputs{l}(li);
                numWeights = self.layerSize(l0)*self.layerSize(l);
                self.weights{l}{li} = reshape(allWs(idx+1:idx+numWeights), self.layerSize(l0), self.layerSize(l))';
                idx = idx + numWeights;
            end
        end
        
        sampleOutputs = reshape(loadNumArray(NNdataRows{7}), numSamples, numOutputs);
        if strcmp(transpose, "FEATURE_SAMPLE_ARRAY")
            sampleOutputs = sampleOutputs';
        end
        
        self.y = cell(1, self.numLayers);
        for l = 1:self.numLayers
            self.y{l} = zeros(self.layerSize(l), 1);
        end
        
        
        function numArray = loadNumArray(numericString)
            numStrings = strsplit(numericString, ",");
            numArray = zeros(size(numStrings));
            for n = 1:length(numStrings)
                numArray(n) = str2double(numStrings(n));
            end
        end
    end
end
end

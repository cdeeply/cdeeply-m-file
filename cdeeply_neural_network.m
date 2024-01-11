% usage:

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
            numInputs, numOutputs, numSamples, ...
            samples, importances, sampleTableTranspose, outputRowsColumns, ...
            maxWeights, maxHiddenNeurons, maxLayers, maxLayerSkips, hasBias, allowIOconnections)
        
        [ sampleString, rowcolString ] = self.CDNN_data2table(samples, numInputs+numOutputs, numSamples, sampleTableTranspose, 1);
        if length(importances) > 0
            importancesString = self.CDNN_data2table(samples, numInputs, numSamples, sampleTableTranspose, 1);
        else
            importancesString = "";
        end
        
        orcStrings = cell(size(outputRowsColumns));
        for rc = 1:length(outputRowsColumns)
            orcStrings{rc} = num2str(outputRowsColumns(rc));
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
            "fromWebpage", "off" });
    
        sampleOutputs = self.buildCDNN(response, numOutputs, numSamples, sampleTableTranspose);
    end
    
    
    function sampleOutputs = tabular_encoder(self, ...
            numInputs, numFeatures, numVariationalFeatures, numSamples, ...
            samples, importances, sampleTableTranspose, ...
            doEncoder, doDecoder, variationalDist, ...
            maxWeights, maxHiddenNeurons, maxLayers, maxLayerSkips, hasBias)
        
        [ sampleString, rowcolString ] = self.CDNN_data2table(samples, numInputs, numSamples, sampleTableTranspose, 2);
        if length(importances) > 0
            importancesString = self.CDNN_data2table(samples, numInputs, numSamples, sampleTableTranspose, 2);
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
            "numFeatures", num2str(numFeatures), ...
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
            "fromWebpage", "off" });
        
        if doDecoder, numOutputs = numInputs;
        else, numOutputs = numFeatures;
        end
        
        sampleOutputs = self.buildCDNN(response, numOutputs, numSamples, sampleTableTranspose);
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
    
    
    function [tableStr, rowcol] = CDNN_data2table(self, data, numIOs, numSamples, transpose, NNtype)
        
        rowcolStrings = { "rows", "columns" };
        
        if (strcmp(transpose, "FEATURE_SAMPLE_ARRAY"))
            dim1 = numIOs;
            dim2 = numSamples;
            rowcol = rowcolStrings{NNtype};
        elseif (strcmp(transpose, "SAMPLE_FEATURE_ARRAY"))
            dim1 = numSamples;
            dim2 = numIOs;
            rowcol = rowcolStrings{3-NNtype};
        else
            error("transpose must be either \"FEATURE_SAMPLE_ARRAY\" or \"SAMPLE_FEATURE_ARRAY\"");
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

clc;close all;
%LOAD DATA
digitDatasetPath = fullfile('Dataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%Split the data into 80% training and 20% validation
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');
%Show random images from datapath
figure;
perm = randperm(100,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
%Image size of model input layer
inputSize = lgraph.Layers(1).InputSize;
%augmentation 
augmenter = imageDataAugmenter( ...
    'RandXReflection' , true , ...
    'RandYReflection' , true );
%resize the training and test images and operations to perform on the training images
augimdsTrain      = augmentedImageDatastore([inputSize(1),inputSize(2),3],imdsTrain,'DataAugmentation',augmenter);
augimdsValidation = augmentedImageDatastore([inputSize(1),inputSize(2),3],imdsValidation);
%Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',5, ...
    'MaxEpochs',50, ...
    'ValidationFrequency',100, ...
    'ValidationData',augimdsValidation, ...
    'Verbose',false, ...
    'Plots','training-progress');
%Train Model
net = trainNetwork(augimdsTrain,lgraph,options);
%Validation Accuracy
[YPred,scores] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);
%Confusion Matrix
plotconfusion(imdsValidation.Labels,YPred)
%Predict random images
idx = randperm(numel(imdsValidation.Files),16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(scores(idx(i),:)),3) + "%");
end
%Save Model
mobilenet100 = net;
save mobilenet100
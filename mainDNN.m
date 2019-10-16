% ======== Bit-Error Rate Prediction using NN ========
% ======== part 2: traning and testing data ====
% ======== matlab layer model  ====

clear all;

load('./DATA/XLong3.mat');
load('./DATA/YLong3.mat');

Y = - 10*(log(Y/1000)/log(10));


ID = 2;   % 1 for iMBer, 2 for MitBer
figure;
histogram(Y(:,ID))
title('histrogram of MitBER');
xlabel('MitBER (x10^3)')

[len,inputSize] = size(X);

trainLen = floor(len * 0.80 );


Xtrain = X(1:trainLen,:);
Xtest = X(trainLen+1:end,:);
Ytrain = Y(1:trainLen,ID);
Ytest = Y(trainLen+1:end,ID);


% ======= convert ==========

[Ntrain,~] = size(Ytrain);
[Ntest,~] = size(Ytest);

for i = 1:Ntrain
    X_train(i) = mat2cell(Xtrain(i,:)',[inputSize]);
end
for i = 1:Ntest
    X_test(i) = mat2cell(Xtest(i,:)',inputSize);
end
Xtrain = X_train;
Xtest = X_test;

for i = 1:Ntrain
    Y_train(i) = mat2cell(Ytrain(i,:)',[1]);
end
for i = 1:Ntest
    Y_test(i) = mat2cell(Ytest(i,:)',[1]);
end

Ytrain = Y_train;
Ytest = Y_test;


% ======= build ==========
    layers = [ ...
        sequenceInputLayer(inputSize)
            fullyConnectedLayer(128)
            fullyConnectedLayer(128)
            fullyConnectedLayer(128)
        fullyConnectedLayer(numPara)
        regressionLayer];
    
    
    options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'GradientThreshold',10, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Verbose',1, ...
    'Plots','training-progress', ...
    'InitialLearnRate',0.01,...
    'OutputFcn',@(info)savetrainingplot(info)...
    );

% ======= train ==========
tic;
[net, trainInfo]  = trainNetwork(Xtrain,Ytrain,layers,options);
toc;
disp('training over!');

% ======== Test  ===========
Ypred = predict(net,Xtest);


% ======== Eval  ===========
if isRNN == 0
    Ypred = cell2mat(Ypred);
    Ypred = reshape(Ypred,1,[])'; 
    Ytest = cell2mat(Ytest)';
    
end


err = abs(Ytest - Ypred);

figure;
scatter(Ytest, Ypred);
clear title xlabel ylabel;
grid on;
title('Ytest vs ML estimate')
xlabel('Ytest');
ylabel('Ypred');
saveas(gcf,'scatter.png');

figure;
histogram(err);
clear title xlabel ylabel;
title('estimate error histogram');
xlabel('estimate error ');
ylabel('count');
grid on;
saveas(gcf,'hist.png');
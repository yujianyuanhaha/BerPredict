% ======== Bit-Error Rate Prediction using NN ========
% ======== part 2: traning and testing data ====


clear all;

load('./DATA/XLong3.mat');
load('./DATA/YLong3.mat');

Y = - 10*(log(Y/1000)/log(10));

% === !  trim the Inf in Y ======

Y( isinf(Y)) = 100;



% X = X_n;
% Y = Y_n;

ID = 2;   % 1 for iMBer, 2 for MitBer
figure;
histogram(Y(:,ID))
title('histrogram of MitBER');
xlabel('MitBER (x10^3)')

[len,~] = size(X);

trainLen = floor(len * 0.80 );



Xtrain = X(1:trainLen,:);
Xtest = X(trainLen+1:end,:);
Ytrain = Y(1:trainLen,ID);
Ytest = Y(trainLen+1:end,ID);


hiddenLayerSize = [32 32 32];




net = fitnet(hiddenLayerSize);
net.trainParam.Mu = 10^(-100)


[net, tr] = train(net, Xtrain', Ytrain');   % using LM method

Ypred = net(Xtest');
Ypred = Ypred';
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
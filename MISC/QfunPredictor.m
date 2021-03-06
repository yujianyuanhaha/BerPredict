addpath(genpath('./'));

clear all;

EbNo = -30:0.2:10;
Len = length(EbNo);
trainLen = floor( Len*0.80);
for  i = 1:Len
    ideal(i) = q(sqrt(2*10.^(0.1*EbNo(i))));
end

figure;
semilogy(EbNo, ideal); 
title('therotic Q function');
xlabel('EbNo  (dB)');
ylabel('BER');
grid on;

X = EbNo;
Y = ideal;




for i = 1:2
    permID = randperm(Len);
    X = X(permID);
    Y = Y(permID);
end


Xtrain = X(1:trainLen);
Xtest = X(trainLen+1:end);
Ytrain = Y(1:trainLen);
Ytest = Y(trainLen+1:end);


hiddenLayerSize = [32 32 32];
net = fitnet(hiddenLayerSize);
[net, tr] = train(net, Xtrain, Ytrain);

Ypred = net(Xtest);
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



X2 = EbNo;
Y2pred = net(X2);
figure;
semilogy(EbNo(100:end), ideal(100:end)); 
hold on;
semilogy(X2(100:end), Y2pred(100:end)); 
title('NN approximate');
xlabel('EbNo  (dB)');
ylabel('BER');
grid on;
legend('theoretic','NN aproximate')

save('./DATA/NN_approximate.mat','Y2pred');
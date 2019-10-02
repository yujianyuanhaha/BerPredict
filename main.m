addpath(genpath('./'));

% --------- settings -----
folderID   = './32k_N8k_S(-5,10)';
note = 'ref signal, 81k';
machine = 'MBP';
hiddenLayerSize =[32 32 32];
isRadBas = 0;
isBayes = 0;
isSend = 1;

% load data
load(strcat(folderID,'/Xtrain.mat'));
load(strcat(folderID,'/Ytrain.mat'));
load(strcat(folderID,'/Xtest.mat'));
load(strcat(folderID,'/Ytest.mat'));



% =================== train ============
net = fitnet(hiddenLayerSize);

if isRadBas &&  ~isBayes
    numLayer = length(hiddenLayerSize);
    for i = 1:numLayer
        net.layers{i}.transferFcn = 'radbas';
    end
    [net, tr] = train(net, Xtrain, Ytrain);
elseif ~isRadBas && isBayes
    net.trainFcn = 'trainbr';
    net.trainParam.epochs=40;
    [net, tr] = train(net, Xtrain, Ytrain);
elseif isRadBas && isBayes
    numLayer = length(hiddenLayerSize);
    for i = 1:numLayer
        net.layers{i}.transferFcn = 'radbas';
    end
    net.trainFcn = 'trainbr';
    [net, tr] = train(net, Xtrain, Ytrain);
else
    [net, tr] = train(net, Xtrain, Ytrain,'useGPU','yes');
end

%save
save(strcat(folderID,'/net.mat'),'net');

% =================== test ============
Ypred = net(Xtest);


% =================== eval ============
err = abs(Ytest - Ypred);
mse    = immse(Ytest, Ypred);
errM = mean(err');
errMdeg = errM*180/pi
errVar = var(err);

fileID = fopen( strcat(folderID,'/sum.txt'),'w');
fprintf(fileID,'errMdeg: \n');
fprintf(fileID,'%s ',errMdeg);
fclose(fileID);

save(strcat(folderID,'/errMdeg.mat'  ),'errMdeg');
save(strcat(folderID,'/err.mat'  ),'err');
save(strcat(folderID,'/Ytest.mat'),'Ytest');
save(strcat(folderID,'/Ypred.mat'),'Ypred');



% % ======== save fig & send email =====s=======


figure;
scatter(Ytest(1,:), Ypred(1,:));
hold on;
scatter(Ytest(2,:), Ypred(2,:));
clear title xlabel ylabel;
grid on;
title('Ytest vs ML estimate')
xlabel('Ytest');
ylabel('Ypred');
saveas(gcf,strcat(folderID,'/scatter.png'));

figure;
histogram(err);
clear title xlabel ylabel;
title('estimate error histogram');
xlabel('estimate error (rad)');
ylabel('count');
grid on;
saveas(gcf,strcat(folderID,'/hist.png'));

% % ========  send email =====s=======
title = strcat(machine,' , ' ,note,' , ' ,folderID,...
                '  ,mean error',num2str(errMdeg));
nnSetting = sprintf('TrainSize = %d, time = %.2f, trainFcn = %s, isRadBas = %d, isBayes = %d ', ...
    trainSize, tr.time(end),tr.trainFcn,isRadBas, isBayes);
content = strcat( nnSetting);
attachment = {strcat(folderID,'/scatter.png'),strcat(folderID,'/hist.png'),strcat(folderID,'/net.mat')...
    strcat(folderID,'/err.mat'),strcat(folderID,'/Ytest.mat'),strcat(folderID,'/Ypred.mat'),...
    };
if isSend
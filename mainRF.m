clear all;
load('./Data/all2/X.mat')
load('./Data/all2/Y.mat')

% convert Y into dB
Y = -10 * log(Y)/log(10);
figure;
histogram(Y)

% split
X = array2table(X);
Xtest = X(1:1e3,:);
Xtrain = X(1e3+1:end,:);
Ytest = Y(1:1e3,:);
Ytrain = Y(1e3+1:end,:);

tic;
for i = 1:5
    y_train = Ytrain(:,i);
    y_test = Ytest(:,i);

    t = templateTree('NumVariablesToSample','all',...
        'PredictorSelection','interaction-curvature','Surrogate','on');
    rng(1); % For reproducibility
    Mdl = fitrensemble(Xtrain,y_train,'Method','Bag','NumLearningCycles',200, ...
        'Learners',t);
    B = compact(Mdl);
    Ypred(:,i) = predict(B, Xtest);
    toc;
    i
end









% plot
T = {'unMit','FFT2','D3S','Notch','FRFT'};
figure;
for i = 1:5
    subplot(2,3,i)
    scatter(Ytest(:,i), Ypred(:,i))
    grid on
    xlabel('ground BER (dB)')
    ylabel('predicted BER (dB)')
    title(T(i))
end

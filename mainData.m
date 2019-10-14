% --------------------------------------------------------
% Bit-Error-Rate (BER) Predict using Neural Network
% Jun, 2019
% Jet Yu, ECE, Virginia Tech
% jianyuan@vt.edu
%
%
% function approximate
% Input:
% 1. SNR
% 2. SIR
% 3. Modulation level: M
% 4. Pulse Shape type, sample per symbol (sps): sps
% 5. Window size: W
% Output: BER
% --------------------------------------------------------



% ======= generate intf type ========
% 4 type: awgn, tone, chirp, filtN(filtered noise, low-passed white noise)
% 1. awgn: SNR
% 2. tone: Ac#, theta_c #, freq_c
% 3. chirp: Ac#, theta_c #, freq_s #, freq_t
% 4. filtN: a

clear all;
close all;
% hardcode for fast compute
addpath(genpath('./'));
rand('seed',0);

t = load('./DATA/pulse1.mat');
p1 = t.p1;
t = load('./DATA/pulse2.mat');
p2 = t.p2;

intfName = { 'awgn', 'tone', 'chirp', 'filtN'};

% X.bitsPerSym = 2;
% X.sps = 2;

iterMAX = 3200;
bitLen = 10000; % 1000/2*4


tic;
for i = 1:iterMAX
    
    X.sps = 4;
    X.bitsPerSym = 2;
    rxLen  = bitLen/X.bitsPerSym * X.sps ;
    
    % --- intference setting --
    X.SNRdB = -10 + rand*20;   % [-10 10]
    X.SIRdB = -10 + rand*20;         
    freq_c = 0.01 :(0.40-0.01)/(iterMAX-1) :0.40;
    f_s    = 0.001;
    f_t    = 0.002:(0.40-0.002)/(iterMAX-1):0.40;
    a      = 0.01 :(0.50-0.01)/(iterMAX-1) :0.50;
    
    X.intfType = intfName(1);
    
    if strcmp( X.intfType,'awgn')
        Interf = 1/sqrt(2*10^(X.SNRdB(i)/10))*(randn(1, rxLen) + 1j*randn(1, rxLen));
        
    elseif strcmp( X.intfType,'tone')
        SIRdB = rand*20-10;
        Interf = 1/sqrt(2*10^(SIRdB/10))*exp(1j* 2*pi * freq_c(i) * [1:rxLen])+...
            1/sqrt(2*10^(X.SNRdB(i)/10))*(randn(1, rxLen) + 1j*randn(1, rxLen));
       
    elseif strcmp( X.intfType,'chirp')
        SIRdB = rand*20-10;
        Interf = 1/sqrt(2*10^(SIRdB/10))*myChirp(f_s,f_t,rxLen)+...
            1/sqrt(2*10^(X.SNRdB(i)/10))*(randn(1, rxLen) + 1j*randn(1, rxLen));
       
    elseif strcmp( X.intfType,'filtN')
        SIRdB = rand*20-10;
        Interf = 10 * 1/sqrt(2*10^(SIRdB/10)) * filter(a(i),[1 a(i)-1],randn(1,rxLen)+1j*randn(1,rxLen))+...
            1/sqrt(2*10^(X.SNRdB(i)/10))*(randn(1, rxLen) + 1j*randn(1, rxLen));
       
    elseif strcmp( X.intfType,'copyCat')
        SIRdB = rand*20-10;
        sps = 16;
        sigLen = rxLen/X.sps;
        bitsPerSym = 2;
        bits = floor( rand(1,sigLen)*2 );
        s = myMod(bits,bitsPerSym);
        s = conv(upsample(s,sps),p2);
        copyCat = s(1:rxLen/bitsPerSym);
        copyCat = (copyCat-mean(copyCat))/var(copyCat);
        Interf = 1/sqrt(2*10^(SIRdB/10)) * copyCat +...
            1/sqrt(2*10^(X.SNRdB(i)/10))*(randn(1, rxLen) + 1j*randn(1, rxLen));

    end


    Tx = floor( rand(1,bitLen)*2 );   % input info bits
    SoI = myMod(Tx,X.bitsPerSym);
    SoI = conv(upsample(SoI,X.sps),p1);
    SoI = SoI(1:bitLen*X.sps/X.bitsPerSym);
    
    Rx = Interf + SoI;
    
    % mitigation
    % RxClean = myFFT(Rx, calculate_threshold(Rx));
    RxClean = SoI;   % no noise/ no intf test

    % find sync after pulse shape
    t = reshape(RxClean,4,[]);
    [~,idx] = max( sum(t,2));
    RxClean2 = RxClean([idx:X.sps:end]);
    RxClean2 = [ RxClean2, zeros(1, 5000 - length(RxClean2))];
    
    % demod
    RxDemod = zeros(2,bitLen/2);
    for k = 1:bitLen/2
        [~, minId] = min(abs( RxClean2(k) - [1j, -1, -1j,1] ));
        RxDemod(1,k) = floor( (minId-1)/2);
        RxDemod(2,k) = mod(minId-1,2);
    end
    RxDemod = reshape(RxDemod,1,[]);
    
    Y(i) = checkBER(Tx,RxDemod) ;
    
end



% if mod((i-1)*numClass+j,50) == 0
%     processMsg = sprintf('data generating %.2f %%', i*100.0/(iterMAX*numClass));
%     disp(processMsg);
%     toc;
% end
%
%
%
% save('X.mat','X');
% save('Y.mat','Y');
%
% Y= categorical(Y);
%
% Size = numel(X);
%
% % permute
% for i = 1:100
%     permID = randperm(Size);
%     X = X(permID);
%     Y = Y(permID);
% end
%
% SizeTrain = floor(Size*0.80);
% Xtrain = X(1:SizeTrain);
% Ytrain = Y(1:SizeTrain);
%
% Xval = X(SizeTrain+1:Size*0.90);
% Yval = Y(SizeTrain+1:Size*0.90);
%
% Xtest = X(Size*0.90+1:Size);
% Ytest = Y(Size*0.90+1:Size);






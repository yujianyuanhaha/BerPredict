% --------------------------------------------------------
% Inteference Signal Classification using Neural Network
% Jun, 2019
% Jet Yu, ECE, Virginia Tech
% jianyuan@vt.edu
% --------------------------------------------------------



% ======= generate intf type ========
% 4 type: awgn, tone, chirp, filtN(filtered noise, low-passed white noise)
% 1. awgn: SNR
% 2. tone: Ac#, theta_c #, freq_c
% 3. chirp: Ac#, theta_c #, freq_s #, freq_t
% 4. filtN: a

clear all;
close all;


dataTag = 'PSD_RNN_';

iterMAX =100000;
bitLen = 2000;
maxEpochs     = 20;



% intfName = { 'awgn', 'tone', 'chirp', 'filtN','copyCat'...
%     'toneChirp','toneFiltN','chirpFiltN','toneChirpFiltN'};
intfName = { 'awgn', 'tone', 'chirp', 'filtN'};

numClass = length(intfName);
inputSize      =  500; % ================


% --- intference setting --
SNRdB    = -10  :(20-(-10))/(iterMAX-1)  :20;
freq_c = 0.01 :(0.40-0.01)/(iterMAX-1) :0.40;
f_s    = 0.001;
f_t    = 0.002:(0.40-0.002)/(iterMAX-1):0.40;
a      = 0.01 :(0.50-0.01)/(iterMAX-1) :0.50;

Y = [];

tic;
for i = 1:iterMAX
    for j = 1:numClass
        intfType = intfName(j);
       
        if strcmp(intfType,'awgn')
            x = 1/sqrt(2*10^(SNRdB(i)/10))*(randn(1, bitLen) + 1j*randn(1, bitLen));
            y = 1;
        elseif strcmp(intfType,'tone')
            SIRdB = rand*20-10;
            x = 1/sqrt(2*10^(SIRdB/10))*exp(1j* 2*pi * freq_c(i) * [1:bitLen])+...
                1/sqrt(2*10^(SNRdB(i)/10))*(randn(1, bitLen) + 1j*randn(1, bitLen));
            y = 2;
        elseif strcmp(intfType,'chirp')
            SIRdB = rand*20-10;
            x = 1/sqrt(2*10^(SIRdB/10))*myChirp(f_s,f_t,bitLen)+...
                1/sqrt(2*10^(SNRdB(i)/10))*(randn(1, bitLen) + 1j*randn(1, bitLen));
            y = 3;
        elseif strcmp(intfType,'filtN')
            SIRdB = rand*20-10;
            x = 10 * 1/sqrt(2*10^(SIRdB/10)) * filter(a(i),[1 a(i)-1],randn(1,bitLen)+1j*randn(1,bitLen))+...
                1/sqrt(2*10^(SNRdB(i)/10))*(randn(1, bitLen) + 1j*randn(1, bitLen));
            y = 4;
        elseif strcmp(intfType,'copyCat')
            SIRdB = rand*20-10;
            sps = 16;
            sigLen = bitLen/sps;
            s = sign(randn(1,sigLen))+1j*sign(randn(1,sigLen));
%             copyCat = myPulseShape2(s,sps,4,0.25,'sqrt');
            p2=[0.0133294045829483	0.00777030651457686	0.00123897509528740	-0.00604791284466589	-0.0138177598264617	-0.0217492345025039	-0.0294819671605503	-0.0366284653541104	-0.0427878522077968	-0.0475609348036246	-0.0505660333237387	-0.0514549472913709	-0.0499284063815163	-0.0457503518520167	-0.0387604214655622	-0.0288840653685093	-0.0161398010386978	-0.000643219182246266	0.0173925246735259	0.0376598611280455	0.0597635831382631	0.0832310824424753	0.107525737263410	0.132063008823302	0.156228686818484	0.179398628047102	0.200959262245959	0.220328098172476	0.236973453125156	0.250432651146802	0.260327988429086	0.266379846900254	0.268416445313014	0.266379846900254	0.260327988429086	0.250432651146802	0.236973453125156	0.220328098172476	0.200959262245959	0.179398628047102	0.156228686818484	0.132063008823302	0.107525737263410	0.0832310824424753	0.0597635831382631	0.0376598611280455	0.0173925246735259	-0.000643219182246266	-0.0161398010386978	-0.0288840653685093	-0.0387604214655622	-0.0457503518520167	-0.0499284063815163	-0.0514549472913709	-0.0505660333237387	-0.0475609348036246	-0.0427878522077968	-0.0366284653541104	-0.0294819671605503	-0.0217492345025039	-0.0138177598264617	-0.00604791284466589	0.00123897509528740	0.00777030651457686	0.0133294045829483];        
            s = conv(upsample(s,sps),p2);
            copyCat = s(1:bitLen);
            copyCat = (copyCat-mean(copyCat))/var(copyCat);
            x = 1/sqrt(2*10^(SIRdB/10)) * copyCat +...
                1/sqrt(2*10^(SNRdB(i)/10))*(randn(1, bitLen) + 1j*randn(1, bitLen));
            y = 5;
%             figure;
%             subplot(3,1,1)
%             plot(abs(fftshift(fft(copyCat))));



        elseif strcmp(intfType,'toneChirp')
            SIRdB = rand*20-10;
            x1 = 1/sqrt(2*10^(SIRdB/10))*exp(1j* 2*pi * freq_c(i) * [1:bitLen]);            
            SIRdB = rand*20-10;
            x2 = 1/sqrt(2*10^(SIRdB/10))*myChirp(f_s,f_t,bitLen);              
            x = x1 + x2 +  1/sqrt(2*10^(SNRdB(i)/10))*(randn(1, bitLen) + 1j*randn(1, bitLen));
            y = 6;
        elseif strcmp(intfType,'toneFiltN')
            SIRdB = rand*20-10;
            x1 = 1/sqrt(2*10^(SIRdB/10))*exp(1j* 2*pi * freq_c(i) * [1:bitLen]);                
            SIRdB = rand*20-10;
            x3 = 10 * 1/sqrt(2*10^(SIRdB/10)) * filter(a(i),[1 a(i)-1],randn(1,bitLen)+1j*randn(1,bitLen));
            x = x1+x3 + +...
                1/sqrt(2*10^(SNRdB(i)/10))*(randn(1, bitLen) + 1j*randn(1, bitLen));
             y = 7;
        elseif strcmp(intfType,'chirpFiltN')
            SIRdB = rand*20-10;
            x2 = 1/sqrt(2*10^(SIRdB/10))*myChirp(f_s,f_t,bitLen);
            SIRdB = rand*20-10;
            x3 = 10 * 1/sqrt(2*10^(SIRdB/10)) * filter(a(i),[1 a(i)-1],randn(1,bitLen)+1j*randn(1,bitLen));               
            x = x2+x3+1/sqrt(2*10^(SNRdB(i)/10))*(randn(1, bitLen) + 1j*randn(1, bitLen));
           y = 8;

        elseif strcmp(intfType,'toneChirpFiltN')
            SIRdB = rand*20-10;
            x1 = 1/sqrt(2*10^(SIRdB/10))*exp(1j* 2*pi * freq_c(i) * [1:bitLen]);           
            SIRdB = rand*20-10;
            x2 = 1/sqrt(2*10^(SIRdB/10))*myChirp(f_s,f_t,bitLen);
            SIRdB = rand*20-10;
            x3 = 10 * 1/sqrt(2*10^(SIRdB/10)) * filter(a(i),[1 a(i)-1],randn(1,bitLen)+1j*randn(1,bitLen));               
            x = x1 + x2 + x3 +1/sqrt(2*10^(SNRdB(i)/10))*(randn(1, bitLen) + 1j*randn(1, bitLen));
             y = 9;
        end
        
        %    figure;
        %    plot(abs(fftshift(fft(x))));
        sps = 4;
        sigLen = bitLen/sps;
        s = sign(randn(1,sigLen))+1j*sign(randn(1,sigLen));
%         s = myPulseShape2(s,sps,4,0.25,'sqrt');
% hardcore, reduce computation
        p1 =[0.0266468801359843	-0.0276231535739088	-0.0855374118147732	-0.0998121297879238	-0.0322651579086280	0.119473681399052	0.312317558188603	0.473734829004615	0.536592673759239	0.473734829004615	0.312317558188603	0.119473681399052	-0.0322651579086280	-0.0998121297879238	-0.0855374118147732	-0.0276231535739088	0.0266468801359843];
        s = conv(upsample(s,sps),p1);
        s = s(1:bitLen);
        
        x = x + s;
        
%          subplot(3,1,2)
%             plot(abs(fftshift(fft(s))));
%              subplot(3,1,3)
%             plot(abs(fftshift(fft(x))));
        
        % downsample to reduce data size
        d = 4;
        x = downsample(x,d);

        x = fftshift(fft(x));
        x = x.*conj(x); %psd

% alpha-profile
        Nw = 256;
        da = 10;
        a1 = 51;
        a2 = 400;

%         x = alphaProfile(x,Nw, da,a1,a2);


        x = reshape(x,[],1);
%         x = normalize(x);
        x = (x - mean(x))/std(x);
        X((i-1)*numClass+j) = mat2cell(x,[inputSize]);
        Y = [Y;y];
        
        if mod((i-1)*numClass+j,50) == 0
            processMsg = sprintf('data generating %.2f %%', i*100.0/(iterMAX*numClass));
            disp(processMsg);
            toc;
        end

    end
end

save(strcat('L:\IM Classification\data\',dataTag,'X.mat'),'X');
save(strcat('L:\IM Classification\data\',dataTag,'Y.mat'),'Y');

Y= categorical(Y);

Size = numel(X);

% permute
for i = 1:100
    permID = randperm(Size);
    X = X(permID);
    Y = Y(permID);
end 

SizeTrain = floor(Size*0.80);
Xtrain = X(1:SizeTrain);
Ytrain = Y(1:SizeTrain);

Xval = X(SizeTrain+1:Size*0.90);
Yval = Y(SizeTrain+1:Size*0.90);

Xtest = X(Size*0.90+1:Size);
Ytest = Y(Size*0.90+1:Size);


save(strcat('L:\IM Classification\data\',dataTag,'Xtrain.mat'),'Xtrain');
save(strcat('L:\IM Classification\data\',dataTag,'Ytrain.mat'),'Ytrain');
save(strcat('L:\IM Classification\data\',dataTag,'Xtest.mat'),'Xtest');
save(strcat('L:\IM Classification\data\',dataTag,'Ytest.mat'),'Ytest');






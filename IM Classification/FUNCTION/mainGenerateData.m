% ======== Bit-Error Rate Prediction using NN ========
% ======== part 1: generate traning data ====

clear variables
clear all;
clc;


mID = 1    % method applied, 1 for FFT-Threshold, 2 for DSSSS ,3 for Notch 
N = 1e4        % number of bits run per iteration
Ndat = 200     % number of dataset size
Xname =  strcat('./DATA/Xtest',num2str(mID),'.mat');     % name of saved data
Yname =  strcat('./DATA/Ytest',num2str(mID),'.mat');

%====================Add Path==========================%
addpath(genpath('../'));
rand('seed',0);
randn('seed',0);



%====================Environment Setting=====================%
isTimeOffset = 1;  % random time offset?  0 = no, 1 = yes
isPhaseOffset = 1; % random phase offset?  0 = no, 1 = yes
isFreqOffset = 1;  % random frequency offset?  0 = no, 1 = yes
f_fo_max = 20;    % maximum frequency offset
NumIter = 1;    % number of iterations to run per JtoS value;
% randomizes impact of frequency offset

%===============Windowing Setting======================%
windowSize = 4096; % window size: specific number of 'large'
%====================SOI Parameters=====================%

bitsPerSym = 1; % '1' is BPSK; '2' is QPSK
isDSSS = 0; % to use DSSS set to '1' (should only be used with
% 'FFT-Thresh','DsssNbiCancel' cancellation options
fd = 100e3;  %symbol rate (only matters from a relative perpsective)
sps = 4;  %samples per symbol
fs = fd*sps; %sample rate
% Filter parameters for SOI
eBW = 0.25;
span  = 4;    % duration
beta  = 0.25;
addLength = 0;


%====================Channel Parameters=====================%
EbNo = 8;
chan = 'AWGN';

%====================Interference Parameters=====================%
% intParams.type = 'CW';   %'NONE''CW','CHIRP','MOD','FiltNoise'
INTFERENCE = {'NONE','CW','CHIRP','MOD','FiltNoise'};
intParams.type = INTFERENCE(1);
intParams.fc = fd/100;  % center frequency of interference
intParams.BW = fs/50;   % bandwidth of filtered noise or chirp
intParams.SweepRate = 500e3;  %sweep rate for chirps (Hz/sec)
intParams.duty = 1;       %duty cycle for interference
intParams.bitsPerSym = 1; %bit per symbol for interferer
% this is only used with 'MOD' interference type
intParams.sps = 100;       % samples per symbol for interferer
% this is only used with 'MOD' interference type
intParams.eBW = 0.25;    % excess BW for interferer
% this is only used with 'MOD' interference type
JtoS = -10:5:10; % Jammer to Signal ratio


%===============Frequency offset mitigation setting======================%
%===============Only applies to DSSS-based filtering======================%
checkUpbd = 20; % Max frequency that will be examined for correction
precision = 1; %   resolution of frequency offset estimation

%===============Mitigation Method======================%

allMethods = {'FFT-Thresh','DsssNbiCancel','NotchFilter','fFT'};
method = allMethods{mID}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    No changes beyond this point.  All operator controlled %%%%
%%%    variables are listed above this point.                 %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if bitsPerSym == 1
    ModTag = ',BPSK,';
else
    ModTag = ',QPSK,';
end

%===============Method Parameters Setting======================%
if(strcmp(method,'DsssNbiCancel')||(strcmp(method,'FFT-Thresh') && isDSSS == 1))
    qq = 0.9; %tradeoff parameter, balance the previous correlation matrix and current one.
    %result = qq * current + (1-qq) * previous
    FilterLength = 20;
    Ns = 7; % spreading gain
else
    Ns = 1;
    PoleRadius = 0.999;
end

% want window size to be a power of 2. Change to closest power of 2
windowSize = correctWinSize(windowSize,Ns,N,sps);
tag = strcat(ModTag,intParams.type,',EbNo = ',num2str(EbNo),'dB',',windowsize = ',num2str(windowSize));


X = [];
Y = [];
tic;


for k = 1:Ndat
    
    
    %          Y = [Y;unMitBER, MitBER];
    %     X = [X; isPhaseOffset, f_fo,offset, bitsPerSym, sps, JtoS, EbNo, IntfType,duty ]
    %  X = [X; po, f_fo,offset, bitsPerSym, sps, JtoS, EbNo, IntfType,duty ]
    % ----- rewrite -----
    isPhaseOffset = randi(2)-1;
    
    bitsPerSym = randi(2);
    %     sps = randi(4);
    JtoS = -10 + rand*20;
    EbNo = 0 + rand*20;
    IntfType = randi(5);
    intParams.type = INTFERENCE(IntfType);
    duty = 0.2 + 0.6*rand;
    intParams.duty = duty;
    
    
    
    for ii=1:length(JtoS)
        
        for iii=1:NumIter
            % ===============generate TX signal======================%
            bits = round(rand(1,N));
            %xpsk_bits = Xpsk(bitsPerSym, bits);
            %===============Modulation part ======================%
            if(strcmp(method,'DsssNbiCancel')||(strcmp(method,'FFT-Thresh') && isDSSS == 1))
                [sig, pn] = psk_modDSSS(beta, span, sps, Ns, bits, bitsPerSym);
            else
                [sig,pulse] = psk_mod(bitsPerSym, sps, eBW, bits);
            end
            
            %============ Channel Simulation part=================%
            % add phase random offset
            if isPhaseOffset
                po = rand;
                sig = sig.*exp(-1j*2*pi*po);  %
            else
                po = 0;
            end
            
            % add random frequency offset
            if isFreqOffset
                f_fo = f_fo_max*rand;
                f_fo = f_fo / Ns;       % adjust to lower relative offset for DSSS
                sig = sig.* exp(-1j*2*pi*[1:length(sig)]*f_fo/fs);
                
            end
            
            % add random time offset
            if isTimeOffset
                offset = floor(rand*30);
                sig = [sig(offset+1:end), zeros(1,offset)];
            else
                offset = 0;
            end
            
            % add noise/multipath
            % adjust noise level for DSSS case since higher samples per bit
            if (strcmp(method,'DsssNbiCancel')||(strcmp(method,'FFT-Thresh') && isDSSS == 1))
                rChan = channel(sig, EbNo, chan, Ns, bitsPerSym);
            else
                rChan = channel(sig, EbNo, chan, sps, bitsPerSym);
            end
            
            % add interference
            int = addInterf(sig, JtoS(ii), intParams, fs);  %send it sig so we calculate J/S vs signal power, not signal + noise power
            r = rChan  + int;
            SNRdB =  EbNo - 10*log10(sps) + 10*log10(bitsPerSym);
            SNR = 10^(SNRdB/10);
            
            
            
            %=============Mitigation Part ========================%
            
            if(strcmp(method,'FFT-Thresh')||strcmp(method,'DsssNbiCancel'))
                
                if (strcmp(method,'DsssNbiCancel')||(strcmp(method,'FFT-Thresh') && isDSSS == 1))
                    rClean = FreqDomainNotch(r,2048);
                    addLength = sps*N*Ns/bitsPerSym - length(rClean);
                    rClean = [rClean, zeros(1,addLength)];
                    
                else
                    rClean = FreqDomainNotch(r,windowSize);
                end
                
                
                
            end
            
            if(strcmp(method,'DsssNbiCancel'))
                [sig,EstOffset] = time_align_DSSS(r, beta, span, sps, Ns, pn);
                [sig,EstFost] = frequencyCompDsss(sig,sps,pn,fs,checkUpbd, precision, Ns, N,bitsPerSym);
                x_end1 = sig;
                x_end2 = downsample(x_end1,sps);
                x_end3 = windowProcessDsss(rClean,windowSize, qq ,sps,FilterLength,  beta, span, Ns, pn, N, fs,checkUpbd, precision, bitsPerSym);
                
            end
            
            if(strcmp(method,'FFT-Thresh') && isDSSS == 1)
                %==================Frequency offset mitigate========%
                [sig,EstOffset] = time_align_DSSS(r, beta, span, sps, Ns, pn);
                [sig,EstFost] = frequencyCompDsss(sig,sps,pn,fs,checkUpbd, precision, Ns, N,bitsPerSym);
                x_end1 = sig;
                x_end2 = downsample(x_end1,sps);
                [sig,EstOffset] = time_align_DSSS(rClean, beta, span, sps, Ns, pn);
                [sig,EstFost] = frequencyCompDsss(sig,sps,pn,fs,checkUpbd, precision, Ns, N,bitsPerSym);
                x_end1 = sig;
                x_end3 = downsample(x_end1,sps);
                
            end
            
            
            
            if(strcmp(method,'fFT'))
                Nfft = 10000;
                nBlks = floor(length(r)/Nfft);
                rClean = zeros(1,Nfft*nBlks);
                for i=1:nBlks
                    rClean((i-1)*Nfft+1:i*Nfft) = myChirpCancel(r((i-1)*Nfft+1:i*Nfft));
                end
                
            end
            
            
            if(strcmp(method,'NotchFilter'))%MA:  this really needs to be %%switched to operate on windows, right now it takes a gigantic %FFT to find
                %the interference frequency and then
                %assumes it never moves.  It's a simple
                %thing to make it operate on blocks and
                %estimate the interference frequency
                %every time, but is it stable to keep
                %changing the IIR filter? You will need
                %to record the input sample history and
                %re-settle the filter feedback each time
                %you change the pole location
                rClean = windowProcessNotch(r,PoleRadius,windowSize);
            end
            
            
            
            % ============check results part========================%
            ideal(ii) = q(sqrt(2*10.^(0.1*EbNo)));
            
            if(strcmp(method,'DsssNbiCancel')||(strcmp(method,'FFT-Thresh') && isDSSS == 1))
                if iii == 1
                    unMitBER(ii) = 1/NumIter*psk_demodDSSS(x_end2, pn, Ns, N, bitsPerSym,eBW, bits, 5000, sps, addLength);
                    MitBER(ii) = 1/NumIter*psk_demodDSSS(x_end3, pn, Ns, N, bitsPerSym,eBW, bits, 5000, sps, addLength);
                else
                    unMitBER(ii) = unMitBER(ii) + 1/NumIter*psk_demodDSSS(x_end2, pn, Ns, N, bitsPerSym,eBW, bits, 5000, sps, addLength);
                    MitBER(ii) = MitBER(ii) + 1/NumIter*psk_demodDSSS(x_end3, pn, Ns, N, bitsPerSym,eBW, bits, 5000, sps, addLength);
                end
                
            else
                if iii == 1
                    unMitBER(ii) = 1/NumIter*psk_demod(r, bitsPerSym, sps, eBW,bits,0);
                else
                    unMitBER(ii) = unMitBER(ii) + 1/NumIter*psk_demod(r, bitsPerSym, sps, eBW,bits,0);
                end
                
                if iii == 1
                    MitBER(ii) = 1/NumIter* psk_demod(rClean, bitsPerSym, sps, eBW,bits,0);
                else
                    MitBER(ii) = MitBER(ii) + 1/NumIter* psk_demod(rClean, bitsPerSym, sps, eBW,bits,0);
                end
                
            end
            
        end % iteration loop
    end     % JtoS loop
    
    
    
%     assert(isnan(MitBER)==0,'Nan here')
    
    %     if isnan(MitBER) ~=0
    
    Y = [Y;unMitBER * 1000, MitBER * 1000];
    X = [X; po, f_fo,offset, bitsPerSym, sps, JtoS, EbNo, IntfType,duty ];
    %     else
    %         [po, f_fo,offset, bitsPerSym, sps, JtoS, EbNo, IntfType,duty ];
    %     end
    % method ID, windowSize fixed for now
    if mod(k,10) == 0
        k
        toc;
        % [num,~] = size(X)
    end
    
    
    
end


save(Xname  ,'X');
save( Yname  ,'Y');



%         figure
%         semilogy(JtoS, unMitBER, 'k-x')
%         hold on
%         semilogy(JtoS, ideal, 'k-o')
%         hold on
%         semilogy(JtoS, MitBER, 'r-o')
%         xlabel('J/S (dB)')
%         ylabel('BER')
%         legend('WIthout Mitigation','Ideal','With Mitigation')
%         grid on;
%         axis([JtoS(1) JtoS(end) 1e-5 1])
%         title(strcat(method,tag))
%
%         figName = sprintf(strcat(method,'_',intParams.type,'.png'));
%         saveas(gcf,figName)


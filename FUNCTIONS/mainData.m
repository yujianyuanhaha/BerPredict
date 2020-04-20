% == Data generation for BER prediciton using Random Forest/ Neural Network =====================
% Jianyuan Yu, jianyuan@vt.edu, Virginia Tech
%
% interference type: AWGN, CW, CHIRP, MOD, FiltNoise
%        |  FFT2  |   DSSS  |  Notch  |  FRFT  |
% narrow |   Y    |   N/A   |   Y     |   Y    |
% spread |   Y    |   Y     |   Y     |   Y    |



clear all;
addpath(genpath('./'));

% ================= global setting ==================

folder = './Data/demo/'   %  folder where data to be save
mkdir(folder)
N             = 1e5   % number of bits run per iteration
Ndat          = 20   % number of dataset size
isIncludeFRFT = 0     % isFRFT included? 1 for include
isDSSS        = 0
%
isTimeOffset  = 0     % random time offset?  0 = no, 1 = yes
isPhaseOffset = 0     % random phase offset?  0 = no, 1 = yes
isFreqOffset  = 0     % random frequency offset?  0 = no, 1 = yes
f_fo_max      = 20;   % maximum frequency offset
NumIter       = 1     % number of iterations to run per JtoS value;
% randomizes impact of frequency offset

%===============Windowing Setting======================%
windowSize = 2048 % window size: specific number of 'large'
%====================SOI Parameters=====================%

bitsPerSym = 1; % '1' is BPSK; '2' is QPSK
fd         = 100e3;  %symbol rate (only matters from a relative perpsective)
sps        = 4;  %samples per symbol
fs         = fd*sps; %sample rate
eBW        = 0.25;  % Filter parameters for SOI
span       = 4;    % duration
beta       = 0.25;
addLength  = 0;


%====================Channel Parameters=====================%
EbNo = 8;
chan = 'AWGN';

%====================Interference Parameters=====================%
INTFERENCE = {'NONE','CW','CHIRP','MOD','FiltNoise'};
intParams.type = INTFERENCE(1);
intParams.fc = fd/100;  % center frequency of interference
intParams.fc_a = 1.0;  % for 2 tone
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
methodSet = [1,2,3];
if isIncludeFRFT == 1
    methodSet = [1,2,3,4];
end
allMethods = {'FFT2','D3S','Notch','FRFT'};


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





X = [];
Y = [];
tic;

for k = 1:Ndat
    
%     isDSSS         = randi(2)-1;           % spread signal or not
    duty           = 0.2 + 0.1*randi(8);   % duty cycle
    intParams.duty = duty;
    IntfType       = randi(4)+1;           % interference type
    intParams.type = INTFERENCE(IntfType);
    bitsPerSym     = randi(2);             % BPSK or QPSK
    J2SRange       = [-20,-15,-10:1:10];
    JtoS           = J2SRange(randi(length(J2SRange)));  % J2S
    EbNo           = 9 + randi(3);
    intParams.fc_a = 0.0;                  % default, only change when 'CW'
    intParams.fc_p = 0.0;
    
    % CW
    if IntfType == 2
        intParams.fc   = 1e3 * randi(20);
        intParams.fc_a = 1e3 * randi(20);  % 2nd tone frequency
        intParams.fc_p = -6 + randi(6);    % 2nd tone power level, based on 1st tone
    end
    % CHIRP
    if IntfType == 3
        intParams.SweepRate = 1e3 * randi(500);
    end
    % Mod
    if IntfType == 4
        intParams.bitsPerSym = randi(2);
        intParams.sps        = 100* randi(8);
        intParams.eBW        = 0.25 * (0.1+0.9*rand);
    end
    % filtNoise
    if IntfType == 5
        intParams.BW = 8e3 *  (0.1+0.9*rand);
    end
    
    
    
    for ii=1:length(JtoS)
        for iii=1:NumIter
            % ===============generate TX signal======================%
            bits = round(rand(1,N));
            %=============Mitigation Part ========================%
            Temp_BER = [];
            flag_noise = 0;
            rn = randi(100000);
            
            for iMethod = 1:length(methodSet)
                method = allMethods{iMethod};

                if isDSSS == 1
                    qq = 0.9; %tradeoff parameter, balance the previous correlation matrix and current one.
                    FilterLength = 20;
                    Ns = 7; % spreading gain
                else
                    Ns = 1;
                end
                PoleRadius = 0.999;
                windowSize = correctWinSize(windowSize,Ns,N,sps);
                
                %===============Modulation part ======================%
                if isDSSS == 1
                    [sig, pn] = psk_modDSSS(beta, span, sps, Ns, bits, bitsPerSym);
                else
                    [sig,pulse] = psk_mod(bitsPerSym, sps, eBW, bits);
                end
                
                
                %============ Channel Simulation part=================%
                % add phase random offset
                if isPhaseOffset
                    po = rand;
                    sig = sig.*exp(-1j*2*pi*po);  %
                    
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
                
                if isDSSS == 1
                    rChan = channel(sig, EbNo, chan, Ns, bitsPerSym);
                else
                    rChan = channel(sig, EbNo, chan, sps, bitsPerSym);
                end
                
                % add interference
                int = addInterf(sig, JtoS(ii), intParams, fs);  %send it sig so we calculate J/S vs signal power, not signal + noise power
                r = rChan  + int;
                SNRdB =  EbNo - 10*log10(sps) + 10*log10(bitsPerSym);
                SNR = 10^(SNRdB/10);
                
                
                if(strcmp(method,'FFT2')||strcmp(method,'D3S'))
                    if (strcmp(method,'D3S')||(strcmp(method,'FFT2') && isDSSS == 1))
                        rClean = FreqDomainNotch(r,2048);
                        addLength = sps*N*Ns/bitsPerSym - length(rClean);
                        rClean = [rClean, zeros(1,addLength)];
                    else
                        rClean = FreqDomainNotch(r,windowSize);
                    end
                end
                
                if(strcmp(method,'D3S'))
                    [sig_um,EstOffset_um] = time_align_DSSS(r, beta, span, sps, Ns, pn);
                    [sig_um,EstFost_um] = frequencyCompDsss(sig_um,sps,pn,fs,checkUpbd, precision, Ns, N,bitsPerSym);
                    [sig,EstOffset] = time_align_DSSS(rClean, beta, span, sps, Ns, pn);
                    [sig,EstFost] = frequencyCompDsss(sig,sps,pn,fs,checkUpbd, precision, Ns, N,bitsPerSym);
                    x_end1 = sig_um;
                    x_end2 = downsample(x_end1,sps);
                    x_end3 = windowProcessDsss2(rClean,windowSize, qq ,sps,FilterLength,  beta, span, Ns, pn, N, fs,checkUpbd, precision, bitsPerSym,EstOffset,EstFost);
                end
                
                
                
                if(strcmp(method,'FRFT'))
                    Nfft = 10000;
                    nBlks = floor(length(r)/Nfft);
                    rClean = zeros(1,Nfft*nBlks);
                    for i=1:nBlks
                        rClean((i-1)*Nfft+1:i*Nfft) = myChirpCancel(r((i-1)*Nfft+1:i*Nfft));
                    end
                end

                if(strcmp(method,'Notch'))
                    rClean = windowProcessNotch(r,PoleRadius,windowSize);
                end
                
                if( ( strcmp(method,'FFT2') || strcmp(method,'Notch') )  && isDSSS == 1)
                    %==================Frequency offset mitigate========%
                    [sig_um,EstOffset_um] = time_align_DSSS(r, beta, span, sps, Ns, pn);
                    [sig_um,EstFost_um] = frequencyCompDsss(sig_um,sps,pn,fs,checkUpbd, precision, Ns, N,bitsPerSym);
                    x_end1 = sig_um;
                    x_end2 = downsample(x_end1,sps);
                    [sig,EstOffset] = time_align_DSSS(rClean, beta, span, sps, Ns, pn);
                    [sig,EstFost] = frequencyCompDsss(sig,sps,pn,fs,checkUpbd, precision, Ns, N,bitsPerSym);
                    x_end1 = sig;
                    x_end3 = downsample(x_end1,sps);
                end
                
                
                % ============check results part========================%
                ideal(ii) = q(sqrt(2*10.^(0.1*EbNo)));
                if isDSSS == 1
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
                
                Temp_BER = [Temp_BER,unMitBER , MitBER];
                
            end  % method loop
        end    % iteration loop
    end   % JtoS loop
    
    
    Y = [Y; Temp_BER ];
    temp =  [  bitsPerSym,              JtoS,          EbNo,          duty, ...
                isDSSS,                  IntfType,      intParams.fc,  intParams.SweepRate, ...
                intParams.bitsPerSym, intParams.sps, intParams.eBW, intParams.BW,...
                 intParams.fc_a, intParams.fc_p];
    X = [X;temp];
    
    % =================== proceoss control ============
    if mod(k,1) == 0
        k
        toc;
    end
    if mod(k,20) == 0   % frequently save down in case it broke halfway
        Y_raw = Y;
        save( strcat(folder,'Y_raw.mat')  ,'Y_raw');
        Y( isnan(Y))     = 1;
        Y( Y < 1/N )     = 1/N;
        Y( imag(Y) ~= 0) = 1;
        save( strcat(folder,'Y.mat')  ,'Y');
        save( strcat(folder,'X.mat')  ,'X');
    end
    
end

% save down
Y_raw = Y;
save( strcat(folder,'Y_raw.mat')  ,'Y_raw');
Y( isnan(Y))     = 1;
Y( Y < 1/N )     = 1/N;
Y( imag(Y) ~= 0) = 1;
save( strcat(folder,'Y.mat')  ,'Y');
save( strcat(folder,'X.mat')  ,'X');







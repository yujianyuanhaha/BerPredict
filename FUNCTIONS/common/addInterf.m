function int = addInterf(sig, JtoS, intParams, fs)
% function int = addInterf(sig, JtoS, intParams, fs)
%
% This function creates an interferer based on the parameters specified in
% intParams.  The interferer is to be added to the received signal outside
% of this function.
%
% Inputs
%   sig - the signal of interest (SOI); this is used only to calibrate JtoS
%   JtoS - Jammer to Signal ratio (in dB) ; used to determine the strength
%           of the interference
%   intPrams - this is a struct used to control the interfernece parameters
%            - intParams.type = type of interferer {'NONE','CW',
%            'CHIRP', 'FiltNoise', or 'MOD'
%            - intParams.fc = center frequency of CW interference 
%            - intParams.BW = bandwidth of filtered noise or chirp
%            - intParams.SweepRate = sweep rate for chirps (Hz/sec)
%            - intParams.duty = duty cycle for interference
%            - intParams.bitsPerSym = bit per symbol for interferer
%                          this is only used with 'MOD' interference type
%            - intParams.sps = samples per symbol for interferer                        
%                          this is only used with 'MOD' interference type
%            - intParams.eBW = excess BW for interferer 
%                          this is only used with 'MOD' interference type
%   fs - sampling rate
%
% Outputs
%   int - complex samples of the generated interference 




if(strcmp(intParams.type,'NONE'))
    int = zeros(size(sig));
elseif(strcmp(intParams.type,'CW'))
    int = exp(1j*2*pi*intParams.fc*(0:length(sig)-1)./fs);
elseif(strcmp(intParams.type,'FiltNoise'))
    PssBndFreq = intParams.BW/fs*0.9;
    StpBndFreq = intParams.BW/fs;
    lpFilt = designfilt('lowpassfir', 'PassbandFrequency', PssBndFreq,...
             'StopbandFrequency', StpBndFreq, 'PassbandRipple', 0.25, ...
             'StopbandAttenuation', 65, 'DesignMethod', 'kaiserwin');
    tmp = 1/sqrt(2)*randn(length(sig),1)+j/sqrt(2)*randn(length(sig),1);
    int = filter(lpFilt,tmp); 
    int = int(1:length(sig)).';
    
elseif(strcmp(intParams.type,'CHIRP'))
    tt = (0:length(sig)-1)/fs;
    phase = 2*pi*rand;
    sawFc = intParams.SweepRate / intParams.BW;
    s = (intParams.BW/2)*sawtooth(2*pi*sawFc*tt + phase,0);   %sawtooth wave
%     s = (intParams.BW/2)*square(2*pi*5*tt);
%     s = (intParams.BW/2)*sin(2*pi*FM_rate*tt);
%     s = (intParams.BW)*(rand(size(tt))-.5);
    int_s = 2*pi*cumsum(s)/fs;
    int = cos(int_s) + 1j*sin(int_s);
elseif(strcmp(intParams.type,'MOD'))
    %int = ones(size(sig));
    bits = round(rand(1,length(sig)/intParams.sps));
    [int,pulse] = psk_mod(intParams.bitsPerSym, intParams.sps, intParams.eBW, bits);
    %random phase offset
    int = int*exp(j*2*pi*rand);
    % random symbol timing offset
    start_samp = ceil(rand*intParams.sps);
    if length(int) >= (length(sig)+ intParams.sps)
        int = int(start_samp:start_samp+length(sig)-1);
    else
        int = int(start_samp:end);
        int = [int,zeros(1,length(sig)-length(int))];
    end
else
    error('Unimplemented interference type');
end

if(intParams.duty < 1)
    minDwell = .01;  %let's just say it's a 10ms minimum dwell time
    blockLen = round(minDwell*fs);
    nBlocks = floor(length(sig)/(blockLen));
    x = -rand(1,nBlocks) + intParams.duty;  %params.duty represents the expected duty cycle 0 <= params.duty <= 1;
    multiplier = kron(x>0,ones(1,blockLen));
    multiplier = [multiplier, zeros(1,length(sig)-length(multiplier))];
else
    multiplier = ones(1,length(sig));
end

% J/S = 20log10(J_rms/S_rms)
% 10^(J/S/20) = J_rms / S_rms
% S_rms*(10^(J/S/20)) = J_rms
if(~strcmp(intParams.type,'NONE'))
    scalar = rms(sig)*10^(JtoS/20)/rms(int);
    int = scalar.*int.*multiplier;
%     for the sake of test vectors, let's add a little fading onto the
%     interferer also
    rayChan = comm.RayleighChannel(...
                                    'SampleRate',fs, ...
                                    'PathDelays',[0], ...
                                    'AveragePathGains',[0], ...
                                    'NormalizePathGains',true, ...
                                    'MaximumDopplerShift',.5, ...
                                    'RandomStream','mt19937ar with seed', ...
                                    'Seed',22, ...
                                    'PathGainsOutputPort',true);
%     int = rayChan(int(:)).';
end

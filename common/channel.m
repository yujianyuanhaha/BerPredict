function r = channel(sig,ebno,chanType, sampsPerSym, bitsPerSym)
% function r = channel(sig,ebno,chanType, sampsPerSym, bitsPerSym)
%
% This function adds AWGN to signal to achieve specified Eb/No
%
% Inputs
%   sig - Input signal that noise will be added to
%   ebno - Eb/No in dB
%   chanType - 'AWGN' is only channel type supported at this point.
%            - later other channels may be added
%   sampsPerSym - number of samples per symbol.  Needed to conver Eb/No to
%                 SNR
%   bitsPerSym - number of bits per symbol. Needed to conver Eb/No to
%                 SNR
%
% Outputs
%   r - complex samples of the signal plus noise
%

    if(strcmp(chanType,'AWGN') || strcmp(chanType,'awgn'))
        r = sig;
    else
        error(sprintf('Channel Type %s not implemented',chanType));
    end
    
    SNR = ebno - 10*log10(sampsPerSym) + 10*log10(bitsPerSym);
    std_dev = 10^(-SNR/20)*rms(sig)/sqrt(2);
    n = std_dev*randn(size(sig)) + 1j*std_dev*randn(size(sig));
    r = r + n;
end


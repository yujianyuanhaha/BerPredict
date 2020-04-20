function [EstFost] = frequencyCompDsss2(sig,sps,pn,fs, checkFreUpbd,precision, Ns, Nb, bitsPerSym, beta, span)
% function [y,EstFost] = frequencyCompDsss(sig,sps,pn,fs, checkFreUpbd,precision, Ns, Nb,bitsPerSym)
% 
% This function compensates for frequency offset priror to DSSS-based
% transversal filtering.  The frequency offset must be estimated using
% knowledge of the PN signal and the signal parameters.  
% Inputs 
%   sig - the original signal without frequency offset compensation.
%   sps - the number of samples per symbol
%   pn - the pn sequence
%   fs - the sampling rate
%   checkFreUpbd - upper bound for frequency offset search 
%   precision - granularity/precision of the frequency offset search
%   Ns - number of chips per bit
%   Nb - number of bits represented in the input
%   bitsPerSym - bits per symbol of modulation scheme
%
% Outputs
%   y - input signal with frequency offset removed
%   EstFost - estimated frequency offset
%

numIter = ceil(checkFreUpbd / precision)+1;
tempFost = 0;
estBvale = [];

shape = 'sqrt';
p = rcosdesign(beta,span,sps,shape);
rCosSpec =  fdesign.pulseshaping(sps,'Raised Cosine','Nsym,Beta',span,0.25);
rCosFlt = design ( rCosSpec );
rCosFlt.Numerator = rCosFlt.Numerator / max(rCosFlt.Numerator);
upsampled = upsample(pn, sps); % upsample
FltDelay = (span*sps)/2;           % shift
temp = filter(rCosFlt , [ upsampled , zeros(1,FltDelay) ] );
pulse_sig = temp(9:end);


CorrLength = min(length(pn),length(sig));


for i=1:numIter
    tempFost = i-1;
    tmp = pulse_sig(1:CorrLength).* (sig(1:CorrLength) .* exp(1j*2*pi*[1:length(sig(1:CorrLength))]*tempFost/fs));
    tmp2 = sum(reshape(tmp,Ns, CorrLength/Ns));
    estBvale(i) = sum(abs(tmp2).^2);
end
[maxVal,maxInd] = max(estBvale);
EstFost = maxInd - 1;

end

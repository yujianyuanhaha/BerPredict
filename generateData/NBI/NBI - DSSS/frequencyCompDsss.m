function [y,EstFost] = frequencyCompDsss(sig,sps,pn,fs, checkFreUpbd,precision, Ns, Nb,bitsPerSym)
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
temp_pn = repmat(pn', 1, 4);
temp_pn= reshape(temp_pn',1,length(pn)*4);
for i = 1:numIter
    if tempFost == 0
        temp_sig = sig .* temp_pn;
        temp_sig = reshape(temp_sig',Ns*sps,Nb/bitsPerSym);
        vc = abs((sum(temp_sig))).^2;
        estBvale(i) = sum(vc);
    else
        temp_sig = sig .* exp(1j*2*pi*[1:length(sig)]*tempFost/fs) ;
        temp_sig = temp_sig .* temp_pn;
        temp_sig = reshape(temp_sig',Ns*sps,Nb/bitsPerSym);
        vc = abs((sum(temp_sig))).^2;
        estBvale(i) = sum(vc);
    end
    tempFost = tempFost + precision;
end
idx = find(estBvale == max(estBvale));
EstFost = (ceil(sum(idx)/length(idx))-1) * precision;
y =  sig.* exp(1j*2*pi*[1:length(sig)]*EstFost/fs);
end


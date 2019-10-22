function [sig,EstOffset] = time_align_DSSS(sig_offset, beta, span, sps, Ns, pn)
% function [sig,EstOffset] = time_align_DSSS(sig_offset, beta, span, sps, Ns, pn)
%
% This function performs time synchronization for a DSSS signal.  This is
% to be used for interference mitigation of DSSS signals using a
% transversal filter which requires chip-level synchronization.
%
% Inputs
%   sig_offset - input signal with arbitrary time offset (both chip level 
%                and code level
%   beta, span  - pulse shaping filter parameters in the original signal
%   sps - number of samples per symbol (chip in this case)
%   Ns - number of chips per bit in DSSS signal
%   pn - psuedo-noise sequence
 
    MaxOffset = sps*Ns;
    OrigLength = length(sig_offset);
    sig_offset = [zeros(1,MaxOffset),sig_offset,zeros(1,MaxOffset)];
    timing_offset = 0;  
    shape = 'sqrt';
    p = rcosdesign(beta,span,sps,shape);
    rCosSpec =  fdesign.pulseshaping(sps,'Raised Cosine','Nsym,Beta',span,0.25);
    rCosFlt = design ( rCosSpec );
    rCosFlt.Numerator = rCosFlt.Numerator / max(rCosFlt.Numerator);
    upsampled = upsample(pn, sps); % upsample
    FltDelay = (span*sps)/2;           % shift
    temp = filter(rCosFlt , [ upsampled , zeros(1,FltDelay) ] );
    sig = temp(9-timing_offset:end-timing_offset);        
    
    MaxDelayCheck = round(MaxOffset*1.2);
    CorrLength = min(length(pn),length(sig_offset))-MaxDelayCheck;
    CorrLength = CorrLength - rem(CorrLength,Ns);
    
    
    for i=1:MaxDelayCheck
        offset = i-1;
        tmp = sig(1:CorrLength).*sig_offset(1+offset:CorrLength+offset);
        tmp2 = sum(reshape(tmp,Ns, CorrLength/Ns));
        Z(i) = sum(abs(tmp2).^2);
    end
    
    [maxVal,maxInd] = max(Z);
    EstOffset = maxInd - 1;
        
    sig = sig_offset(1+EstOffset:OrigLength+EstOffset);
    
    EstOffset = EstOffset - MaxOffset;
end


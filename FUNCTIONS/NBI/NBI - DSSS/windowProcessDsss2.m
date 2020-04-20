%@@ -0,0 +1,85 @@
function y = windowProcessDsss2(r,windowSize, qq ,sps, FilterLength,  beta, span, Ns, pn, N, fs,checkUpbd, precision, bitsPerSym, outEstOffset,outEstFost)
% function y = windowProcessDsss(r,windowSize, qq ,sps, FilterLength,  beta, span, Ns, pn, N, fs,checkUpbd, precision, bitsPerSym)
%
% This function cancels narrowband interference from a DSSS signal using a
% transversal filter on a window-by-window basis.
%
% Inputs
%   r - input signal containing the DSSS signal and interference
%   windowSize - size of the processing window
%   qq - forgetting factor from window to window
%   sps - number of samples per chip
%   FilterLength - length of the transversal filter
%   beta, span - filter coefficients for pulse shaping filter of original 
%                DSSS signal
%   Ns - number of chips per bit
%   pn - pseudo-noise sequence
%   N - number of bits represented by the input samples
%   fs - sampling frequency 
%   checkUpbd - max frequency offset to search over for frequency offset 
%               compensation
%   precision - granularity/precision of the frequency offset search
%   bitsPerSym - modulation order (bits per PSK symbol)
%
% Outputs
%   y - output samples which have interference mitigated
%


OrigLength = length(r);
dit = power(10,(floor(log10(windowSize))));
temp_win = floor(windowSize / dit);
windowSize = temp_win * dit;
temp_tail = rem(OrigLength,windowSize);
if temp_tail>0
                r = [r,zeros(1,windowSize-temp_tail)];
end
windowNum = floor(OrigLength/windowSize);
y = [];
previous_Rxx = 0;
EstOffset = 0;

%========find timing offset==========%
% It can use first window sequence to determine the time offset
%x_end1 = r(1:1+windowSize-1);
%pn_temp = pn(1:1+windowSize-1);
%[temp_sig,EstOffset] = time_align_DSSS(x_end1, beta, span, sps, Ns, pn_temp);
EstOffset = outEstOffset;
%=========adjust received sequence ======%
%  re-merging all window sequence and re-windowing (means the system should have a cache)
MaxOffset = sps*Ns;
r_offset = [zeros(1,MaxOffset),r,zeros(1,MaxOffset)];
MoveOffset =  MaxOffset + EstOffset;
r = r_offset(1+MoveOffset:OrigLength+MoveOffset);
%[r,EstFost] = frequencyCompDsss(r,sps,pn,fs,checkUpbd, precision, Ns, N, bitsPerSym);
EstFost = outEstFost;
r =  r.* exp(1j*2*pi*[1:length(r)]*(EstFost/Ns)/fs);
for idxW = 1:windowNum
    temp_idx = (idxW-1) * windowSize+1;
    x_end1 = r(temp_idx:temp_idx+windowSize-1);
    x_end2 = downsample(x_end1,sps);
     % calculate Rxx and p
    [Rxx, p] = CorrleationMatrixCalc(x_end2, FilterLength);
    Rxx = qq * Rxx + (1 - qq) * previous_Rxx;
    previous_Rxx = Rxx;
     % remove interference
    x_end3= DsssNbiCancel(x_end2, Rxx, p); 
    y = [y,x_end3];
end  

if length(y) < length(r)/sps
    %temp_idx = (idxW-1) * windowSize+1;
    x_end1 = r(length(y)*sps+1:end);
    x_end2 = downsample(x_end1,sps);
     % calculate Rxx and p
    [Rxx, p] = CorrleationMatrixCalc(x_end2, FilterLength);
    Rxx = qq * Rxx + (1 - qq) * previous_Rxx;
    %previous_Rxx = Rxx;
     % remove interference
    x_end3= DsssNbiCancel(x_end2, Rxx, p); 
    y = [y,x_end3];

end


end
function y = DsssNbiCancel(x, Rxx, p)
% function y = DsssNbiCancel(x, Rxx, p)
%
% This function cancels narrowband interference from a DSSS signal using a
% two-sided transversal filter.  
% 
% Inputs
%    Rxx - the estimated correlation matrix of the input signal 
%    p - correlation vector (correlation between the current sample and 
%        surrounding samples.
%    x - input samples containing a DSSS signal sampled once per chip
%        and narrowband interference
%
% Outputs
%    y - output signal with narrowband interference mitigated
% 
% Note that the filter length is inferred from the sizes of Rxx and p


w = inv(Rxx)*p;

FilterLength = length(w);
N = length(x);

for i=1:N

    
    if i < FilterLength/2+1
        tmp = [zeros(FilterLength/2-i+1,1); x(1:i-1).';x(i+1:i+FilterLength/2).'];
    elseif i < N-FilterLength/2
        tmp = [x(i-FilterLength/2:i-1).';x(i+1:i+FilterLength/2).'];
    else
        tmp = [x(i-FilterLength/2:i-1).';x((i+1):N).'; zeros(FilterLength/2-length(x((i+1):N).'),1)];
    end
    
    y(i) = x(i) - w'*tmp;
    
end


end


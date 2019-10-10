function ws = correctWinSize(windowsize, Ns,N,sps)
% function ws = correctWinSize(windowsize)
%
% This function changes the window size to a power of 2
%
% Inputs
%   windowsize - original specified window size
%
% Ouptuts
%   ws - output window size to be used
%
limit = Ns * N * sps;
if windowsize > limit
    windowsize = Ns * N;
    warning('The window size should be not bigger than the whole length of transmitted signal');
else
    windowsize = windowsize;
end
ws = 0;
bits = round(log2(windowsize));
ws = power(2,bits);
if ws ~=windowsize
    warning('The windowsize should be the exponential multiple of 2.');
    warning('The windowsize has been modified.');
end
end
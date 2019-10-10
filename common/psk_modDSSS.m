function [sig, pn] = psk_modDSSS(beta, span, sps, Nspd, bits, bitsPerSym)
% function [sig, pn] = psk_modDSSS(beta, span, sps, Nspd, bits, To, bitsPerSym) 
%
% This function performs psk modulation along with pulse shaping and DS
% spreading
%
% Inputs
%   beta, span,  - filter parameters 
%   sps - samples per symbol
%   Nspd - chips per bit for DSSS 
%   bits - bits for modulation
%   To - time offset
%   bitsPerSym - bits per symbol for PSK modulation
%
% Outputs
%   sig - complex samples of the PSK transmit symbols
%   pn - pseudo noise sequence


temp_bits = Xpsk(bitsPerSym, bits);
Ns = Nspd; % spreading gain 7
Nb_len = length(temp_bits);
pn = round(rand([1,Ns*Nb_len])) - 0.5;
pn = sign(pn);
tmp = repmat(temp_bits', 1,Ns);
tmp2= reshape(tmp',1,Ns*Nb_len);
temp_bits = tmp2.*pn;
shape = 'sqrt';
p = rcosdesign(beta,span,sps,shape);
rCosSpec =  fdesign.pulseshaping(sps,'Raised Cosine','Nsym,Beta',span,0.25);
rCosFlt = design ( rCosSpec );
rCosFlt.Numerator = rCosFlt.Numerator / max(rCosFlt.Numerator);
upsampled = upsample(temp_bits, sps); % upsample
FltDelay = (span*sps)/2;           % shift
temp = filter(rCosFlt , [ upsampled , zeros(1,FltDelay) ] );
sig = temp(9:end);        % to be fixed
end
function [minBER] = psk_demodDSSS(sig, pn, Ns, N, bitsPerSym, eBW, bits, trim, sps, addLength)
% function [minBER] = psk_demodDSSS(sig, pn, Ns, N, bitsPerSym, eBW, bits, trim, sps, addLength)
% 
% This function performs DSSS despreading and psk demodulation and returns
% the bit error rate (BER)
%
% Inputs
%   sig - the received signal with pulse modulated PSK symbols and DSSS
%         spreading
%   pn - the pseudo-noise (PN) sequence used to spread the signal
%   Ns - number of chips per bit
%   N - number of bits expected in this stream of samples
%   bitsPerSym - bits per symbol of PSK M = 2^bitsPerSym
%   eBW - pulse shape parameter (excess BW of raised cosine pulse)
%   bits - the transmitted bits (used for BER calculation)
%   trim - number of samples to eliminate due to PLL
%   sps - samples per symbol of pulse shape 
%   addLength - number of symbols to eliminate from end
%
% Outputs 
%   minBER - bit error rate achieved

temp_sig = sig.*pn;
    Nb_len = N / bitsPerSym;
    sig = 1/Nb_len*sum(reshape(temp_sig , Ns, Nb_len));
    
    if addLength > 0
        sig = sig(1:length(sig)-floor(addLength/sps/Ns));
    end
    
    M = 2^bitsPerSym;

    sig = sig(trim+1:end);  %trim off the beginning "trim" samples of the signal
%     carrier recovery goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig = PLL(sig, bitsPerSym);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% symbol timing
    avg_sym_mag = sum(reshape(abs(sig),1,[]),2);
    [val,sync_idx] = max(avg_sym_mag);
    
    syms = sig(sync_idx:1:end);

    
% phase offset (if not accounted for in carrier recovery)
    symPhase = angle(syms);
    symPhaseOrig = symPhase;
    symPhase(symPhase < 0) = symPhase(symPhase<0) + 2*pi;
    symPhase = wrap2pi(symPhase*M);
    offset = angle(sum(exp(1j*symPhase))) / M;
    
    syms = syms.*exp(-1j*offset);
    symPhase = wrap2pi(symPhaseOrig - offset);
    

    

% phase ambiguity (check BER for each phase offset, pick lowest)
    constellationPhase = (0:M-1)*2*pi / M;
    constellationPhase = repmat(constellationPhase(:),1,length(symPhase));
%     make hard decisions for each phase ambiguity and check BER for each
    for i=1:M
        symPhase1 = repmat(symPhase + (i-1)*2*pi/M,M,1);
        err = wrap2pi(symPhase1 - constellationPhase);
        [val,idx] = min(abs(err),[],1);  %idx is the closest constellation index of each symbol
        rxBits = zeros(1,length(syms)*bitsPerSym);
        for j=1:bitsPerSym  %convert symbol indexes to bits
            rxBits(j:bitsPerSym:end) = mod(floor((idx-1)/2^(j-1)),2);
        end
        
        BER(i) = checkBER(bits,rxBits);
    end
    minBER = min(BER);
end


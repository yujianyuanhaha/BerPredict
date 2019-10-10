function sigOut = PLL(sigIn, bitsPerSym)
% function sigOut = PLL(sigIn, bitsPerSym)
%
% This function performs phase-tracking of the desired signal and removes
% the (potentially time-varying) phase offset
%
% Inputs
%   sigIn - Input signal with PSK signal of interst 
%   bitsPerSym - number of bits per symbol of PSK symbols
%
% Outputs
%   sigOut - output signal with phase/frequency offsets removed
%

    if(bitsPerSym > 2)
        error('PLL designed for BPSK/QPSK only!');
    end
    % Costas loop with proportional plus integral loop filter implemented at baseband
    pllState = 0.0; % state information for PLL loop filter
    phs = 0.0; % pll phase state
    a0 = 4*0.001/10; % 0.0002 #0.001 # proportional loop filter coefficient
    a1 = 4*0.000005/10; % 0.000005 # integral loop filter coefficient

    initRMS = rms(sigIn);
    Kp = a0/initRMS;
    Ki = a1/initRMS;
    shift0 = exp(1i*pi/4.0);
    
    nx = numel(sigIn);
    sigOut = zeros(1,nx);
    %obj.save_phs = sigOut;
    for idx = 1:nx
        x = sigIn(idx);

        s1 = x * exp(1i*2*pi*phs); % complex signal mix
        if(imag(s1)>=0)
            s2 = real(s1);% * sign(imag(s1));
        else
            s2 = -real(s1);
        end
        if(real(s1)>=0)
            s3 = imag(s1);% * sign(real(s1));
        else
            s3 = -imag(s1);
        end

        err = (s3 - s2);
        pllState = pllState + err; % Integration of this state
        s4 = Kp * err + Ki * pllState; % Computation of P+I filter

        phs = phs + s4; % Phase accumulator
        sigOut(idx) = s1 * shift0; % Capture result
    end
end

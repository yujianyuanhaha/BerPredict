function symbols = myMod(bits,bitsPerSym)


if (mod(length(bits),bitsPerSym))
    bits = [bits zeros(1,bitsPerSym - mod(length(bits),bitsPerSym))];
end
bits = reshape(bits(:),bitsPerSym,[]);

constellation = exp(1j*(0:2^bitsPerSym-1)*2*pi / 2^bitsPerSym);
constIdx = zeros(1,size(bits,2));
for i=1:bitsPerSym
    constIdx = constIdx + bits(i,:)*2^(i-1);
end
symbols = constellation(constIdx+1);
end
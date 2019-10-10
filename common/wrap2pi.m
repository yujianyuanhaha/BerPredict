function out = wrap2pi(in)
% function out = wrap2pi(in)
%
% This function simply wraps phase values such that they are between -pi
% and pi
%
% Inputs
%   in - input phase values
%
% Outputs
%   out - output phase values mapped to -pi to pi
%

    out = rem(in,2*pi);
%     out = in;
%     while(jnkisempty(out(out >= pi)))
        out(out>=pi) = out(out >= pi) - 2*pi;
%     end
%     while(jnkisempty(out(out < -pi)))
        out(out < -pi) = out(out < -pi) + 2*pi;
%     end
%     for i=1:length(in)
%         while(out(i) > pi)
%             out(i) = out(i) - 2*pi;
%         end
%         while(out(i) <= -pi)
%             out(i) = out(i) + 2*pi;
%         end
%     end
end


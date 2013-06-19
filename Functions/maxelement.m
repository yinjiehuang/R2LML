function output = maxelement(a)
%Compute the maximum element of a vector

b = 1:max(a);
c = histc(a,b);
output = find(c == max(c));

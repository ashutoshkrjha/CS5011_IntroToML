function [clusterpurity] = cluspure(X)

m=size(X,1);
n=size(X,2);
maxval = max(X);
sumval = sum(X);
clusterpurity = maxval./sumval;

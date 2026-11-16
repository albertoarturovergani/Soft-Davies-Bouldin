function [centers,U,iteration] = fuzzy_c_means(X, k, m, seed, maxIter, tol)
if nargin < 3 || isempty(m), m = 2; end
if nargin < 4 || isempty(seed), seed = 42; end
if nargin < 5 || isempty(maxIter), maxIter = 300; end
if nargin < 6 || isempty(tol), tol = 1e-6; end
[n,d] = size(X);
if n < 2 || d < 1 || k < 2 || k > n || k ~= floor(k) || m <= 1 || maxIter < 1 || tol <= 0
    error('Invalid FCM parameters');
end
if any(~isfinite(X(:))) || ~isfinite(m)
    error('Inputs must be finite');
end
rng(seed);
U = rand(n,k);
U = U./sum(U,2);
for iteration = 1:maxIter
    P = U.^m;
    centers = (P'*X)./sum(P,1)';
    D = zeros(n,k);
    for j = 1:k
        D(:,j) = sqrt(sum((X - centers(j,:)).^2,2));
    end
    Unew = zeros(n,k);
    Z = D == 0;
    zeroRows = any(Z,2);
    if any(zeroRows)
        Unew(zeroRows,:) = Z(zeroRows,:)./sum(Z(zeroRows,:),2);
    end
    if any(~zeroRows)
        LW = -2/(m-1)*log(D(~zeroRows,:));
        LW = LW - max(LW,[],2);
        W = exp(LW);
        Unew(~zeroRows,:) = W./sum(W,2);
    end
    delta = max(abs(Unew(:) - U(:)));
    U = Unew;
    if delta < tol, break; end
end
P = U.^m;
centers = (P'*X)./sum(P,1)';
end

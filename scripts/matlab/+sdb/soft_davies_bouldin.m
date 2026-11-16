function score = soft_davies_bouldin(X, centers, U, q, membershipPower, separationTol)
if nargin < 4 || isempty(q), q = 1; end
if nargin < 5 || isempty(membershipPower), membershipPower = 1; end
if nargin < 6 || isempty(separationTol), separationTol = 1e-12; end
[n,d] = size(X);
[k,dc] = size(centers);
if n < 2 || d < 1 || k < 2 || dc ~= d || ~isequal(size(U), [n,k])
    error('Expected X (n,d), centers (k,d), U (n,k), n>=2 and k>=2');
end
if any(~isfinite(X(:))) || any(~isfinite(centers(:))) || any(~isfinite(U(:)))
    error('All inputs must be finite');
end
if q < 1 || ~isfinite(q) || membershipPower <= 0 || ~isfinite(membershipPower)
    error('q must be >=1 and membershipPower must be >0');
end
if separationTol < 0 || ~isfinite(separationTol)
    error('separationTol must be finite and nonnegative');
end
if any(U(:) < 0) || any(U(:) > 1) || any(abs(sum(U,2) - 1) > 1e-7)
    error('U entries must be in [0,1] and sum to one by row');
end
radius = zeros(n,k);
for i = 1:k
    delta = X - centers(i,:);
    radius(:,i) = sqrt(sum(delta.^2,2));
end
scatter = mean(radius.^q,1).^(1/q);
weightedScatter = scatter .* mean(U.^membershipPower,1);
between = zeros(k,k);
for i = 1:k
    delta = centers - centers(i,:);
    between(i,:) = sqrt(sum(delta.^2,2));
end
mask = ~eye(k);
if any(between(logical(mask)) <= separationTol)
    error('Cluster centers coincide or are too close');
end
between(1:k+1:end) = 1;
R = (weightedScatter(:) + weightedScatter(:)')./between;
R(1:k+1:end) = -Inf;
score = mean(max(R,[],2));
end

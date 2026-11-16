function score = davies_bouldin_crisp(X, labels)
labels = labels(:);
if size(X,1) ~= numel(labels) || size(X,1) < 2 || any(~isfinite(X(:)))
    error('Expected finite X (n,d) and labels (n,1)');
end
classes = unique(labels);
k = numel(classes);
if k < 2, error('At least two clusters are required'); end
centers = zeros(k,size(X,2));
within = zeros(k,1);
for i = 1:k
    members = X(labels == classes(i),:);
    centers(i,:) = mean(members,1);
    within(i) = mean(sqrt(sum((members - centers(i,:)).^2,2)));
end
between = zeros(k,k);
for i = 1:k
    between(i,:) = sqrt(sum((centers - centers(i,:)).^2,2))';
end
mask = ~eye(k);
if any(between(logical(mask)) <= 1e-12)
    error('Crisp cluster centroids coincide or are too close');
end
between(1:k+1:end) = 1;
R = (within + within')./between;
R(1:k+1:end) = -Inf;
score = mean(max(R,[],2));
end

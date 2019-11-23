clear all
close all
m(:,1) = [-1;0]; 
Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [1;0]; 
Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; 
Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]; 
thr = [0,cumsum(classPriors)];
N = 10000; 
u = rand(1,N); 
L = zeros(1,N); 
x = zeros(2,N);
figure(1),clf, colorList = 'rbg';
actual_from_class_1=0;
actual_from_class_2=0;
actual_from_class_3=0;
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    
    plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); 
    %axis equal 
    hold on
    grid on
    title('Actual Scatter plot of data generated');
    ylabel('Feature 2','FontSize',10);
    xlabel('Feature 1','FontSize',10);
    legend('Class 1','Class 2','Class 3');
end
for i=1:N
    if L(:,i)==1
        actual_from_class_1=actual_from_class_1 + 1;
    elseif L(:,i)==2
        actual_from_class_2=actual_from_class_2 + 1;
    else
        actual_from_class_3=actual_from_class_3 + 1;
    end
end
        
f=1;
k=1;
n=1;
for i=1:N
    e=x(:,i);
    a= mvnpdf(e,m(:,1),Sigma(:,:,1));
    b= mvnpdf(e,m(:,2),Sigma(:,:,2));
    c= mvnpdf(e,m(:,3),Sigma(:,:,3));
    g1= log(a)+log(classPriors(1)); 
    g2= log(b)+log(classPriors(2));
    g3= log(c)+log(classPriors(3));% g is the discriminant function
    if (g1>g2 & g1>g3)
        q1(:,f)=e;         %q1 contains elements classified as class 1
        f=f+1;
        h(i)=1;             %h contains the infered labels             
        Q(:,i)=e;
    elseif (g2>g3 & g2>g1)
        q2(:,k)=e;         %q2 contains elements classified as class 2
        k=k+1;
        h(i)=2;
        Q(:,i)=e;
    else
        q3(:,n)=e;         %q2 contains elements classified as class 2
        n=n+1;
        h(i)=3;
        Q(:,i)=e;    
    end
end
figure(2)
plot(q1(1,:),q1(2,:),'.');
hold on
plot(q2(1,:),q2(2,:),'.');
hold on
plot(q3(1,:),q3(2,:),'.');
hold on
z= eq(L,h);
r=0;
mb=1;
for i=1:N
    if z(:,i)==0
        r=r+1;
        R(:,mb)=Q(:,i);
        mb=mb+1;
    end
end
plot(R(1,:),R(2,:),'kx');

%axis equal
grid on
title('Infered Scatter plot of data');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2','Class 3','Misclassified');
hold off
per= (r/N)*100;
C= confusionmat(L,h);
figure(3)
confusionchart(L,h)
actual_from_class_1
actual_from_class_2
actual_from_class_3
classified_as_class_1 = length(q1)
classified_as_class_2 = length(q2)
classified_as_class_3 = length(q3)
misclassified_samples= r
probability_of_error_percentage= per
Confusion_Matrix= C

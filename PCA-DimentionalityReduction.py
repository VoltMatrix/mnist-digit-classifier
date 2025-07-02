import numpy as np







#Dimensionality reduction
#PCA(principal component analysis)





#The following code uses NUMPY svd to obtain all the components of training set, then extracts the 2 unit vectors that define the first 2 PCs:
#extracting the top 2 pcs
X_centered= X-X.mean(axis=0)
U,s,Vt= np.linalg.svd(X_centered)
c1= Vt.T[:,0]
c2=Vt.T[:,1]


#Reducing a data sets to 2 dimensions
from sklearn.decomposition import PCA
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
#Creating a PCA object and specifying we want 2 dimensions
pca=PCA(n_components=2)
#Fit PCA to the data(it centers the data automatically and finds the PCs)
X2D = pca.fit_transform(X)
print("Reduced data:\n",X2D)
print("Explained variance ratios:",pca.explained_variance_ratio_)


#The following code projects the training set onto the plane defined by the first 2 principal component
W2= Vt.T[:,:2]  #Get the first 2 principal components                         
X2D=  X_centered.dot(W2)#Project data on 2D space
#vpowert means the transpose of V,which contains the principal components as rows

                   
#Choosing the rght number of dimensions
#The following code performs PCA without reducing dimensionality, then computes
#the minimum number of dimensions required to preserve 95% of the training set’s
#variance:
pca=PCA()
pca.fit(X_train)
cumsum=np.cumsum(pca.explained_variance_ratio_)
d=np.argmax(cumsum>= 0.95) +1

#Scikt -learn automate this 
pca=PCA(n_components=0.95)                                  
X_reduced= pca.fit_transform(X_train)



#Compression means keeping the most important info and removing noice
#The following code compresses the MNIST datasets down to 154 dimensions,then use
# the inverse transform to decompress to 784 dimensions

pca=PCA(n_components=154)
X_reduced= pca.fit_transform(X_train)
X_recovered= pca.inverse_transform(X_reduced)

#Using randomized PCA

pc_rnd= PCA(n_components=154, svd_solver="randomized")
X_reduced= pc_rnd.fit_transform(X_train)


#Incremental PCA
from sklearn.decomposition import IncrementalPCA
#The following code splits the MNIST dataset into 100 mini-batches (using NumPy’s
#array_split() function) and feeds them to Scikit-Learn’s IncrementalPCA class5 to
#reduce the dimensionality of the MNIST dataset down to 154 dimensions 

n_batches= 100
inc_pca= IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
X_reduced= inc_pca.transform(X_train)


#kernal pca is the pca that uses kernal trick to find nonlinear patterns in data
from sklearn.decomposition import KernelPCA
rbf_pca=KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced= rbf_pca.fit_transform(X)


         
#Selecting a kernel and tuning hyperparameters
#Following code creates a 2 step pipeline 
# first reducing dimensionality to 2 dimensions using KPCA
# Then using logistic regression for classification
# Then using grid search cv to find the best kernel and gamma value for kpca


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf= Pipeline([
    ("kpca",KernelPCA(n_components=2)),
    ("log_reg",LogisticRegression()),
])
param_grid= [{
    "kpca_gamma":np.linspace(0.03,0.05,10),
    "kpca_kernel":["rbf","sigmoid"] 
}]                                                                                                    

grid_search=GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X,y)
print(grid_search.best_params_)


#performing the reconstruction
#Training a supervised regression model 
# With projected instances as training set
# And orignal instances as targets
#fit_inverse_transform does it all 

rbf_pca= KernelPCA(n_components=2, kernel="rbf", gamma=0.0433,
                   fit_inverse_transform=True)
X_reduced= rbf_pca.fit_transform(X)
X_preimage= rbf_pca.inverse_transform(X_reduced)
#Computing the reconstruction pre-image error
from sklearn.metrics import mean_squared_error
mean_squared_error(X,X_preimage)
#You can use grid-search with cross-validation to find the kernel and hyperparameters that minimize the error



#Locally linear embedding
from sklearn.manifold import LocallyLinearEmbedding

lle= LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced= lle.fit_transform(X)



































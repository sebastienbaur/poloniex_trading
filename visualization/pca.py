from sklearn.decomposition import PCA

from projet.lib.data_functions.utils import *

# GET COLORSET BASED ON PREDICTIONS


# GET PCA
pca = PCA(n_components=3)
pca.fit(X_train)
print "%s of explained variance : %s" % ('%', pca.explained_variance_ratio_)

first_component = pca.components_[0]
second_component = pca.components_[1]
third_component = pca.components_[2]

# Plot the projections onto the 2D principal plane and 3D principal space
projections_2d = np.array([[np.dot(X_train[k, :], first_component), np.dot(X_train[k, :], second_component)] for k in range(X_train.shape[0])])
projections_3d = np.array([[np.dot(X_train[k, :], first_component), np.dot(X_train[k, :], second_component), np.dot(X_train[k, :], third_component)] for k in range(X_train.shape[0])])

fig = plt.figure(1)
plt.scatter(projections_2d[:,0], projections_2d[:,1], c='r')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.title("Projection of points in the principal plane")
fig.show()

fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(projections_3d[:, 0], projections_3d[:, 1], projections_3d[:, 2], c='r')
ax.set_xlabel('First principal component')
ax.set_ylabel('Second principal component')
ax.set_zlabel('Third principal component')
fig2.show()

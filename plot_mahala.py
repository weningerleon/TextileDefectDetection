fig = plt.figure()

# Show data set
#plt.scatter(X[:, 0], X[:, 1], color='black', label='inliers')
axes = plt.gca()
#axes.set_xlim([80,100])
#axes.set_ylim([80,100])

print("CHANGED!!")
for idx, pt in enumerate(inner_points):
    if pt.label == "faulty":
        plt.scatter(pt.lower_dist, pt.area, color='red', label='inliers')
    else:
        plt.scatter(pt.lower_dist, pt.area, color='green', label='inliers')

# Show Center
plt.plot(robust_cov.location_[0],robust_cov.location_[1],'k*')

# Show contours of the distance functions
xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 1000),
                     np.linspace(plt.ylim()[0], plt.ylim()[1], 1000))

zz = np.c_[xx.ravel(), yy.ravel()]

mahal_robust_cov = robust_cov.mahalanobis(zz)
mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
robust_contour = plt.contour(xx, yy, mahal_robust_cov, levels = [10,25,100,300,600,1000])


plt.clabel(robust_contour, inline=1, fontsize=10)
plt.xlabel('Distance to lower neighboor')
plt.ylabel('Distance to upper neighboor')

plt.title('Warp float points and Mahalanobis distance from center ')


import numpy as np

def integral_transform(img, square = True, ravelled = True, dims = None):
	if square and ravelled:
		side_len = int(len(img) ** .5)
		img_reshaped = np.reshape(img, (side_len, side_len))
		ans = np.zeros((side_len, side_len))
		for i in range(2*side_len - 1):
			for j in range(i + 1):
				if j < side_len and i - j < side_len:
					if i == 0:
						ans[0][0] = img_reshaped[0][0]
					elif j == 0:
						ans[j][i-j] = ans[j][i-j-1] + img_reshaped[j][i-j]
					elif i == j:
						ans[j][i-j] = ans[j-1][i-j] + img_reshaped[j][i-j]
					else:
						ans[j][i-j] = ans[j][i-j-1] + ans[j-1][i-j] - ans[j-1][i-j-1] + img_reshaped[j][i-j]
	return ans
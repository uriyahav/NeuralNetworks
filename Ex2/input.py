import numpy as np
#
letters = []
# # # א
# # a = np.array([[1, 1, -1, -1, -1, -1, 1, 1, -1, -1],
# #               [-1, 1, 1, -1, -1, -1, 1, 1, -1, -1],
# #               [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
# #               [-1, -1, 1, 1, 1, -1, 1, 1, -1, -1],
# #               [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
# #               [-1, -1, 1, 1, -1, 1, 1, 1, -1, -1],
# #               [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
# #               [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1],
# #               [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
# #               [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1],
# #               ])
# # letters.append(a)
# #ב
# # b = np.array([[-1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
# #               [-1, 1, 1, 1, 1, 1, 1,1, -1, -1],
# #               [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1],
# #               [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1],
# #               [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1],
# #               [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1],
# #               [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1],
# #               [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1],
# #               [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# #               [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# #               ])
# # letters.append(b)
# # ג
# # c = np.array([[-1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
# #               [-1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1,-1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
# #               [-1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
# #               [-1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
# #               ])
# # letters.append(c)
# # # ד
# # d = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
# #               [1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               ])
# # letters.append(d)
# # ה
# e = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#               [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               ])
# letters.append(e)
# # # ו
# # f = np.array([[-1, -1, -1, -1, -1, 1, 1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
# #               ])
# # letters.append(f)
# # ז
# # g = np.array([[-1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
# #               [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
# #               ])
# # letters.append(g)
# # ח
# h = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               ])
# letters.append(h)
# # ט
# i = np.array([[1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, 1, 1, 1],
#               [1, 1, -1, -1, -1, 1, 1, 1, 1, 1],
#               [1, 1, -1, -1, -1, 1, 1, -1, 1, 1],
#               [1, 1, -1, -1, -1, 1, 1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
#               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#               ])
# letters.append(i)
h = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
              [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
              [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
              [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
              [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
              [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
              [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
              [1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
              ])
for i in range(10):
    letters.append(h)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1+0.2


# 构造线性模型
k = tf.Variable(0.)
b = tf.Variable(0.)
y = k*x_data+b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))

# 定义一个梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.02)

# 定义一个最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            [k1,b1]=sess.run([k,b])
            print(step, sess.run([k,b]))

print(k1)
predict = k1*x_data+b1
print(predict)

plt.plot(x_data,predict)
plt.scatter(x_data,y_data)

plt.show()

# # 创建常量op
# m1 = tf.constant([[3, 3]])  # 一行两列
# m2 = tf.constant([[2], [3]])  # 两行一列
# product = tf.matmul(m1, m2)  # 矩阵相乘
#
# a = tf.Variable([1, 2])
# b = tf.constant([2, 5])
#
# add = tf.add(a, b)
# sub = tf.subtract(a, b)
#
#
# # 定义一个会话，启动默认图
# # sess = tf.Session()
# # result = sess.run(product)
# # print(sess.run(product))
#
# step = tf.Variable(0, name="step")
#
# # 加1 op
# new_value = tf.add(step, 1)
# # 赋值op
# update = tf.assign(step, new_value)
#
#
#
# input1=tf.constant(1)
# input2=tf.constant(2)
# input3=tf.constant(3)
#
# add=tf.add(input1, input2)
# mul=tf.multiply(add,input3)
#
# input4=tf.placeholder(tf.float32)
# input5=tf.placeholder(tf.float32)
# output=tf.multiply(input4,input5)
#
# init = tf.global_variables_initializer()
#
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     print(sess.run(output,feed_dict={input4:3,input5:5}))
#
#     # result=sess.run([add, mul])
#     # print(result)
#
#
#     #print(sess.run(step))
#     # for _ in range(5):
#     #     sess.run(update)
#     #     print(sess.run(step))
#     #
#     #
#     # print(sess.run(add))
#     # print(sess.run(sub))







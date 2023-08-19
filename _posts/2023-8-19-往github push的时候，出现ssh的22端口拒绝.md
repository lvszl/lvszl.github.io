# 坑：ssh: connect to host github.com port 22: Connection refused
---
data: 2023-8-19
categories: git
# tags: 随笔
---
本文摘自：https://zhuanlan.zhihu.com/p/521340971，用来解决ssh: connect to host github.com port 22: Connection refused问题

# 问题
在使用Windows11时候，原本好好的，然后突然发现push不了github了，为啥，报错原因是ssh：connect to host github.com port 22: Connection refused，利用ssh的日志查看：
```git
$ ssh -vT git@github.com
```
发现ip之类的都没问题，只是22端口被自己电脑的防火墙给墙了。

# 解决方法
直接在git bash中输入：
```
$ vim ~/.ssh/config
```
然后进入vim界面，编辑config。
之后在config文件的末尾，添加如下：

```
Host github.com
  Hostname ssh.github.com
  Port 443
```

然后就能正常push了。

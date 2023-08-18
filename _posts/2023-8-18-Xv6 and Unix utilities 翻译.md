---
# title: Xv6 book——文件描述符与管道
data: 2023-8-18
categories: MIT-Xv6
# tags: 随笔
---

原文
# pingpong（简单）
`编写一个程序，使用UNIX系统调用通过一对管道(每个方向一个管道)在两个进程之间“pingpong”一个字节。父进程应该向子进程发送一个字节;子进程应该输出"<pid>: received ping"，其中<pid>是它的进程ID，将管道上的字节写入父进程，然后退出;父进程应该从子进程读取字节，打印"<pid>: received pong"，然后退出。您的解决方案应该在user/pingpong.c文件中。`

一些提示：
1. 使用pipe创建管道。 
2. 使用fork创建子进程。
3. 使用read从管道中读取，使用write向管道写入。
4. 使用getpid查找调用进程的进程ID。 
5. 将程序添加到Makefile中的UPROGS中。 
6. xv6上的用户程序只有一组有限的库函数可供使用。您可以在user/user.h中看到该列表;源代码(除了用于系统调用)在user/ulib.c、user/printf.c和user/umalloc.c中。
   
**测试：**
​

![Alt text](_posts\image1.png)
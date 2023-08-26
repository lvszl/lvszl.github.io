---
data: 2023-8-26
tag: linux
tag: MIT_Xv6
---

今天写OS课设时候，遇到了用`read`函数从标准命令行中读取数据的问题，记录一下：



```c
#include <stdio.h>
#include <string.h>

int main(){
   char input[32];
   int cnt = read(0, input, sizeof input);
   printf("sizeof input: %d\n",sizeof input);
   printf("cnt: %d\n",cnt);
   printf("%s\n", input);
}
```

输出的结果：

![](https://raw.githubusercontent.com/lvszl/figure/master/20230826174106.png)

说明了以下问题：

1. `read`函数是以换行符作为结束标志的，且只要读入的长度允许，则会读入换行符，如此测试程序，`read`读入了6个字节，包括了换行符。

2. `read` 不以空格符作为分离不同字符串，而是只要指定长度允许，就一口气读到换行符。

3. `sizeof` 返回的是数组的长度

   
这篇文章是我学习git工作流的笔记，可供参考。

首先扔上来一个网页 [mit git intro](https://missing.csail.mit.edu/2020/version-control/) 可供学习。

上面那个会比较简略，[pro git](https://git-scm.com/book/en/v2)里面有更加详尽的内容。

# git 的存在意义
git的核心人物就是完成version control的相关工作。而version control的目的就是让我们能够随时回到一个特定的版本。

那么，git究竟是什么？

## 直接记录快照，而非差异比较
git把数据看作是对小型文件的一系列快照。在Git中，每当提交更新或者保存项目状态的时候，系统就会对当时的*全部文件*创建一个快照并且保存这个快照的索引。如果文件没有修改，那么git就不会重新存储改文件，而是保存一个链接指向之前存储的文件。

## 近乎所有操作都是本地执行

## git保证完整性
git数据库中保存的信息都是数据的哈希值。

## git一般只添加数据

## git的三种状态
Git中的文件有三种状态。
- committed 数据已经安全地保存到本地数据库中。
- staged 对一个已修改文件的当前版本进行了标记，使之包含在下次提交的快照中。
- modified 修改了文件，但是还没有保存到数据库中。

基本的Git工作流程：
1. 在工作区中修改文件。
2. 把想要提交的更改添加到暂存区。
3. 提交更新，找到暂存区的文件，把快照永久存储到Git目录。

## 获取git的帮助（如果你忘记了一些基础知识）
使用```git --help <verb>```来获取帮助。使用```git <verb> -h```可以获得更简便的指示。

# Git基础
## 获取Git仓库
### 在本地已存在目录中初始化仓库
选择对应的仓库，然后```git init```
如果在创造Git仓库之前，这个文件夹内部已经有文件的话，那么我们应该手动追踪这些文件。
```bash
$ git add .
$ git add LICENSE
$ git commit -m 'initial project version'
```
如果想要进行交互式添加
```bash 
git add -i
```

### 克隆现有仓库
```git clone url```
使用```git clone url myname```可以把这个目标目录名命名为myname

## 记录每次更新到仓库
这里涉及一个基础概念，即文件的跟踪状态，可以通过```git status```查看所有文件的跟踪状态。
一般来说，只有不存在于上次快照的记录或者不在暂存区的文件才会是untracked的文件。

那么如何跟踪或者在暂存区中更新一个文件呢，很简单，只需要```git add <filename>```就可以了，这时候文件已经被放入了暂存区。

## 查看已经暂存和未暂存的修改
使用```git diff```查看暂存前后的变化。
使用```git diff --cached```查看已经暂存起来的变化。

## 提交更新
每次提交之前，先用```git status```检查一下。
然后使用```git commit -m "" ```，此时相当于把暂存区的文件拍一张快照，传到Git目录。

## 移除文件
使用```git rm <filename>```删除某一文件。
使用```git rm --cached README```取消Git 对文件的追踪。

## 改名文件
使用```git mv file_from file_to```对文件进行改名。

# 查看提交历史
当你想快速审查一份修改的时候，可以用这些指令。
```git log -p -<time>```显示每次提交所引入的差异,time表示限制只显示最近的两次提交。
```git log --stat```显示每次提交的简略统计信息。

# 撤销操作
## 撤销提交
如果你在提交之后忘记了暂存某些需要的修改，可以这样操作，最终你第二次提交将覆盖第一次提交。
```bash
$ git commit -m 'initial commit'
$ git add forgotten_file
$ git commit --amend
```
## 取消暂存的文件
如果你意外把一个还不想在下次进行提交的文件放到了暂存区，那么你可以使用
```git reset HEAD <filename>```来取消暂存。
或者也可以使用```git restore --staged file.txt```。
当然这个命令有一点危险，需要谨慎使用。

## 撤销对文件的修改
把文件还原成之前的某个版本的样子。
```git checkout --<filename>```
这同样也是一个非常危险的命令，git会用最近提交或者放入暂存区的版本覆盖掉你当前的文件做出的所有本地改动。

当然这个还是存在一定的风险的，现代Git更推荐使用
```git restore filename```它的语义更加明确，用来恢复工作区。

# 远程仓库的使用
git 给克隆的仓库服务器的默认名字是origin。你可以使用```git remote```命令查看远程仓库服务器的名字。

## 添加远程仓库
```git add <shortname> <url>```

## 从远程仓库拉取信息
如果你想拉取远程仓库中存在但是你自己没有的信息，可以运行```git fetch origin```
注意这个命令只会将数据下载到本地仓库，它并不会自动合并或者修改你当前的工作。

## 推送到远程仓库
```git push <remote> <branch>```

## 查看远程仓库的信息
```git remote show <remote>```

# 打标签
## 创建标签
- 附注标签
```bash
$ git tag -a v1.4 -m "my version 1.4"
```
使用```git show```就可以看到标签信息以及对应的提交信息。

- 后期打标签
给过去的commit历史中的某一条提交记录打标签。需要先找到指定提交的校验和（哈希码）。
```git tag -a v1.2 <hashcode>```

## 共享标签
默认情况下，```git push```不会传送标签到远程服务器，因此必须显式推送标签到服务器上，运行```git push origin <tagname>```。

# 分支模型
这部分可能需要一些图片辅助理解，我就不截图了，可以直接去[分支简介](https://git-scm.com/book/zh/v2/Git-%e5%88%86%e6%94%af-%e5%88%86%e6%94%af%e7%ae%80%e4%bb%8b)进行查看，里面讲的很详细。

## 分支创建
```git branch <branchname>```
但是这不会让你自动切换到新的分支里面去。

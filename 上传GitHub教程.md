# 🚀 RAG 项目上传 GitHub 教程（每一步都有截图说明）

> **前提**：你已经在浏览器登录了 GitHub 账号（截图显示已登录 ✅）
> **目标**：把 `D:\RAG` 里的代码推送到 `github.com/xaxqw/enterprise-ticket-routing-rag`

---

## 第 1 步：打开 Git Bash

### 方法 A（推荐）：在文件夹里直接打开

1. 打开**文件资源管理器**（按 `Win + E`）
2. 在地址栏输入 `D:\RAG` 然后回车
3. 在**空白处右键** → 找到 **"Git Bash Here"** 或 **"在此处打开 Git Bash"**
4. 点它，会弹出一个黑色命令行窗口 → 这就是 Git Bash

> ⚠️ 如果右键菜单里没有「Git Bash Here」：
> - 说明你可能没装 Git for Windows。去 https://git-scm.com/downloads 下载安装
> - 或者用方法 B

### 方法 B：从开始菜单打开

1. 按 `Win` 键，输入 `git bash`
2. 点击出现的 **Git Bash** 图标
3. 窗口打开后，输入以下命令进入项目目录：
   ```bash
   cd /d/RAG
   ```
4. 回车

---

## 第 2 步：确认你在正确的目录

在 Git Bash 里输入：

```bash
ls
```

然后回车。你应该看到一堆文件和文件夹名，比如：
- `README.md`、`app/`、`dashboard/`、`scripts/`、`data/`、`logs/`……

如果看到这些 → **对了，目录正确** ✅

如果报错或看不到这些文件 → 检查是不是输错了路径

---

## 第 3 步：检查本地提交状态

输入：

```bash
git log --oneline -5
```

回车。你应该看到 **3 行类似这样的内容**：

```
xxxxxxx docs: 归拢面试资料(简历项目描述模板+面试准备)到主项目目录
xxxxxxx chore: 代码打磨、微调链路修复与作品集完善
xxxxxxx feat: 企业级工单智能分流 RAG 系统 - 完全本地化(免费离线)+多智能体架构+离线评测体系
```

有这 3 行 → **提交都在，可以推了** ✅

如果只有 1 行或 0 行 → 告诉我，我帮你排查

---

## 第 4 步：推送代码到 GitHub ⭐（最关键的一步）

在同一个 Git Bash 窗口里，**原样复制粘贴**下面这条命令（整行一起复制）：

```bash
cd /d/RAG && git push -u origin master
```

然后**按回车**。

### 接下来会发生什么：

| 你看到的 | 含义 | 该做什么 |
|---------|------|---------|
| `Enumerating objects: xx, done.` | 正在统计要上传的文件 | **等** |
| `Compressing objects: xx%` | 正在压缩文件 | **等** |
| `Writing objects: xx%` | 正在上传到 GitHub | **等**（这一步最慢） |
| `Resolving deltas: 100%` | 快完了 | **等** |
| *弹出浏览器登录窗口* | GitHub 要你授权 | **点「Authorize」授权** |
| `To github.com:xaxqw/...` | 上传成功！ | **完成了！** ✅ |

### 可能出现的情况：

#### 情况 1：弹出浏览器窗口要求登录/授权
→ 这是正常的！GitHub 需要验证你是谁。
→ 如果已经登录了 GitHub，点 **「Authorize」**（授权）按钮即可
→ 授权完自动回到 Git Bash 继续上传

#### 情况 2：提示 `Authentication failed` 或 `403`
→ 用户名或 token 不对
→ 告诉我，我帮你重新配置

#### 情况 3：提示 `! [rejected] ... non-fast-forward`
→ 远程仓库有冲突（不太可能，因为是新仓库）
→ 告诉我，我给你解决命令

#### 情况 4：卡住超过 2 分钟没动静
→ 网络慢（GitHub 国内有时慢）
→ 再等 1-2 分钟；如果还是不动，`Ctrl + C` 取消后重试

---

## 第 5 步：验证上传成功

推送完成后，Git Bash 最后几行应该显示类似：

```
 * [new branch]      master -> master
Branch 'master' set up to track remote branch 'master' from 'origin'.
```

看到这两行 → **100% 成功** ✅

然后：
1. **回到你截图的那个浏览器页面**（GitHub 仓库页面）
2. **按 F5 刷新页面**
3. 你应该能看到所有文件列表了（`README.md`、`app/`、`dashboard/`……）

---

## 📋 完整操作清单（复制粘贴版）

打开 Git Bash 后，**依次执行**这三条命令：

```bash
# ① 进入项目目录
cd /d/RAG

# ② 确认提交存在（应该显示 3 行）
git log --oneline -5

# ③ 推送到 GitHub（⭐ 核心步骤）
git push -u origin master
```

每条命令执行完后看输出，没问题再执行下一条。

---

## ❓ 出问题了怎么办？

| 问题 | 解决方法 |
|------|---------|
| `git: command not found` | 没装 Git → 去 https://git-scm.com/downloads 安装 |
| `fatal: not a git repository` | 路径错了 → 确认输入的是 `cd /d/RAG` |
| `Authentication failed` | 凭证过期 → 告诉我，我重新配 |
| 弹窗被杀毒软件拦截 | 允许弹窗，或换用 PAT 方式 |
| 推送太慢/超时 | 网络问题 → 多试几次，或挂代理 |

---

## 🎉 推送成功后你就有了一个完整的公开作品集仓库！

**仓库地址**：`https://github.com/xaxqw/enterprise-ticket-routing-rag`

面试时可以直接把这个链接贴给面试官，或者写在简历上。

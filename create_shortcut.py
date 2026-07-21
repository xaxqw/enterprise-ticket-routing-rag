"""
重建桌面「RAG智能问答」快捷方式（指向 D:\RAG\运行RAG平台.bat）
用 pywin32 的 ShellLink COM 创建（手工拼二进制 .lnk 在 Win11 下点不了）。
"""
import os
import pythoncom
import win32com.shell.shell as shell
import win32com.shell.shellcon as shellcon

DESKTOP = r"C:\Users\xuanx\OneDrive\桌面"
LNK_PATH = os.path.join(DESKTOP, "RAG智能问答.lnk")
TARGET = r"D:\RAG\运行RAG平台.bat"
WORKDIR = r"D:\RAG"
ICON = r"D:\RAG\rag_proj_env\Scripts\python.exe"

# 目标不存在就报错退出
if not os.path.exists(TARGET):
    raise SystemExit(f"目标文件不存在：{TARGET}")

shortcut = pythoncom.CoCreateInstance(
    shell.CLSID_ShellLink,
    None,
    pythoncom.CLSCTX_INPROC_SERVER,
    shell.IID_IShellLink,
)
shortcut.SetPath(TARGET)
shortcut.SetWorkingDirectory(WORKDIR)
shortcut.SetDescription("一键启动 RAG 智能问答平台")
shortcut.SetIconLocation(ICON, 0)

persist = shortcut.QueryInterface(pythoncom.IID_IPersistFile)
persist.Save(LNK_PATH, 0)
print(f"已创建快捷方式：{LNK_PATH}")

# 回读校验
sc2 = pythoncom.CoCreateInstance(
    shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink
)
p2 = sc2.QueryInterface(pythoncom.IID_IPersistFile)
p2.Load(LNK_PATH, 0)
print("校验 -> Path:", sc2.GetPath(shell.SLGP_UNCPRIORITY)[0])
print("        WorkDir:", sc2.GetWorkingDirectory())

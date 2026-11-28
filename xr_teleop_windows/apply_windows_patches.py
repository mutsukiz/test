"""
Patch xr_teleoperate so that teleop_hand_and_arm.py runs on a Windows host.

Run once from the xr_teleoperate root (it is safe to run again; originals are kept as *.orig):
    cd C:\\...\\xr_teleoperate
    python apply_windows_patches.py            (or: python <path>\\apply_windows_patches.py)

What it changes
1. teleop/teleop_hand_and_arm.py
   - Windows: raise the system timer resolution to 1 ms (otherwise time.sleep() rounds up to
     ~15.6 ms and the 250 Hz arm command thread only runs at ~64 Hz).
   - Windows: store the DDS domain / network interface in environment variables, so that
     child processes started with "spawn" can open their own DDS connection.
2. teleop/robot_control/robot_hand_unitree.py (Dex3)
   - Windows: the hand control process cannot receive the parent's DDS publishers / threads
     (multiprocessing "spawn" pickles them -> crash). The child now creates its own DDS
     participant, publishers and HandRetargeting, then runs the unchanged control_process().
On Linux nothing changes in behaviour (all changes are guarded by os.name == "nt").

Not covered: Inspire / BrainCo hands (same pattern, not patched), --ipc (uses Linux-only
"ipc://@..." sockets: do not use --ipc on Windows; keyboard control via sshkeyboard works).
"""
import os
import re
import shutil
import sys

MARK = "[win-patch]"


def patch_file(path, fn):
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print(f"  already patched: {path}")
        return
    new = fn(src)
    if new == src:
        raise SystemExit(f"  pattern not found, not patched: {path}")
    if not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")
    open(path, "w", encoding="utf-8", newline="").write(new)
    print(f"  patched: {path}")


def patch_main(src):
    nl = "\r\n" if "\r\n" in src else "\n"
    anchor = "    args = parser.parse_args()" + nl
    add = (
        f"    if os.name == \"nt\":  # {MARK} Windows host support" + nl +
        "        import ctypes" + nl +
        "        ctypes.WinDLL(\"winmm\").timeBeginPeriod(1)  # 1 ms timer (default ~15.6 ms)" + nl +
        "        os.environ[\"XR_DDS_DOMAIN\"] = \"1\" if args.sim else \"0\"" + nl +
        "        os.environ[\"XR_DDS_IFACE\"] = args.network_interface or \"\"" + nl +
        "        if args.ipc:" + nl +
        "            parser.error(\"--ipc uses Linux-only ipc:// sockets; not available on Windows.\")" + nl
    )
    return src.replace(anchor, anchor + add, 1)


def patch_dex3(src):
    nl = "\r\n" if "\r\n" in src else "\n"
    pat = re.compile(r"(\n        )hand_control_process = Process\(target=self\.control_process, args=\((?P<args>.*?)\)\)",
                     re.S)
    m = pat.search(src)
    if not m:
        return src
    args = m.group("args")
    repl = (
        f"{m.group(1)}proc_args = ({args})" +
        f"{m.group(1)}if os.name == \"nt\":  # {MARK} spawn: child opens its own DDS connection" +
        f"{m.group(1)}    hand_control_process = Process(target=_dex3_spawn_main, args=(self.fps, self.Unit_Test) + proc_args)" +
        f"{m.group(1)}else:" +
        f"{m.group(1)}    hand_control_process = Process(target=self.control_process, args=proc_args)"
    )
    src = src[:m.start()] + repl + src[m.end():]
    helper = nl.join([
        "",
        "",
        f"def _dex3_spawn_main(fps, unit_test, *proc_args):  # {MARK}",
        "    \"\"\"Windows (spawn) entry of the Dex3 control process: own DDS participant + publishers.\"\"\"",
        "    domain = int(os.environ.get(\"XR_DDS_DOMAIN\", \"0\"))",
        "    iface = os.environ.get(\"XR_DDS_IFACE\") or None",
        "    ChannelFactoryInitialize(domain, iface)",
        "    ctrl = Dex3_1_Controller.__new__(Dex3_1_Controller)",
        "    ctrl.fps = fps",
        "    ctrl.Unit_Test = unit_test",
        "    ctrl.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test if unit_test else HandType.UNITREE_DEX3)",
        "    ctrl.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)",
        "    ctrl.LeftHandCmb_publisher.Init()",
        "    ctrl.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)",
        "    ctrl.RightHandCmb_publisher.Init()",
        "    ctrl.control_process(*proc_args)",
        "",
    ])
    anchor = "class Dex3_1_Left_JointIndex(IntEnum):"
    return src.replace(anchor, helper.lstrip(nl) + nl + nl + anchor, 1)


def main():
    root = os.getcwd()
    if not os.path.exists(os.path.join(root, "teleop", "teleop_hand_and_arm.py")):
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (here, os.path.dirname(here)):
            if os.path.exists(os.path.join(cand, "teleop", "teleop_hand_and_arm.py")):
                root = cand
                break
        else:
            raise SystemExit("Run this from the xr_teleoperate root directory.")
    print(f"xr_teleoperate: {root}")
    patch_file(os.path.join(root, "teleop", "teleop_hand_and_arm.py"), patch_main)
    patch_file(os.path.join(root, "teleop", "robot_control", "robot_hand_unitree.py"), patch_dex3)
    print("done.")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

__author__     = "Lluís Vilanova"
__copyright__  = "Copyright 2018, Lluís Vilanova"
__license__    = "GPL version 3 or later"

__maintainer__ = "Lluís Vilanova"
__email__      = "llvilanovag@gmail.com"


import collections
from contextlib import contextmanager
import inspect
import joblib
import logging
import os
import re
import six
import spur
try:
    import spur_watchdog_patch
    from spur_watchdog_patch import SshShell, LocalShell
    spur_patch = True
except:
    from spur import SshShell, LocalShell
    spur_patch = False
import StringIO
import time


logger = logging.getLogger(__name__)


def get_shell(server, user=None, password=None, port=22):
    """Get a new shell.

    If `server` is a spur shell, return that instead.

    Parameters
    ----------
    server : str or object
    user : str (optional)
    password : str (optional)
    port : int (optional)

    """
    if isinstance(server, SshShell) or isinstance(server, LocalShell):
        return server
    else:
        return SshShell(hostname=server,
                        username=user,
                        password=password,
                        port=port,
                        missing_host_key=spur.ssh.MissingHostKey.accept)


@contextmanager
def step(message):
    """A context manager to show simple progress messages around a piece of code.

    Example
    -------
    >>> with step("Doing something")
            print("some text")
    Doing something...
    some text
    Doing something... done

    """
    print(message, "...")
    yield
    print(message, "... done")


class threaded(object):
    """Context manager to run functions in parallel using threads.

    Example
    -------
    Run two processes in parallel and wait until both are finished:

    >>> with step("Running in parallel"), threaded() as t:
            @t.start
            def f1():
                shell = LocalShell()
                shell.run(["sleep", "2"])
                print("f1")

            @t.start
            def f2():
                shell = LocalShell()
                shell.run(["sleep", "1"])
                print("f2")
    Running in parallel...
    f2
    f1
    Running in parallel... done

    """
    def __init__(self, n_jobs=None):
	if n_jobs is None:
	    n_jobs = -1
	self._n_jobs = n_jobs
	self._jobs = []
	self.result = None

    def __enter__(self):
	return self

    def __exit__(self, *args):
	pool = joblib.Parallel(backend="threading", n_jobs=self._n_jobs)
	self.result = pool(joblib.delayed(job, check_pickle=False)(*args, **kwargs)
			   for job, args, kwargs in self._jobs)

    def start(self, target):
        """Decorator to start a function on a separate thread."""
	self._jobs.append((target, [], {}))

    def start_args(self, *args, **kwargs):
        """Callable decorator to start a function on a separate thread.

        Example
        -------
        >>> with threaded() as t:
                @t.start_args(1, b=2)
                def f(a, b):
                    print(a, b)
        1, 2

        """
	def wrapper(target):
	    self._jobs.append((target, args, kwargs))
	return wrapper


def install(shell, package):
    """Install given `package` using `shell`."""
    if isinstance(shell, SshShell):
        hostname = shell.hostname
    else:
        hostname = "localhost"
    shell.run([
        "bash", "-c",
        "dpkg -s %s >/dev/null 2>&1 || sudo apt-get install -y %s" % (package,
                                                                      package),
    ])


def install_deps(shell, packages=None):
    """Install all needed system packages.

    Must be called on a local shell before using other functions that require a
    shell, and before using other functions through the same shell.

    Parameters
    ----------
    shell
        Target system.
    packages : list of str, optional
        Additional packages to install.

    """
    install(shell, "cgroup-tools")
    install(shell, "hwloc")
    install(shell, "rsync")
    install(shell, "netcat-traditional")
    install(shell, "psmisc")
    install(shell, "util-linux")
    if packages is not None:
        for pkg in packages:
            install(shell, pkg)


def wait_run(shell, *args, **kwargs):
    """Run command with a timeout.

    Parameters
    ----------
    shell
       Shell used to run given command.
    timeout : int, optional
       Timeout before erroring out (in seconds). Default is no timeout.
    rerun_error : bool, optional
       Rerun command every time it fails. Default is False.
    args, kwargs
       Paramaters to the shell's spawn method.

    Returns
    -------
    spur.ExecutionResult

    """
    timeout = kwargs.pop("timeout", 0)
    rerun_error = kwargs.pop("rerun_error", False)
    func_kwargs = inspect.getcallargs(shell.spawn, *args, **kwargs)
    func_kwargs.pop("self")
    args = func_kwargs.pop("args", [])
    kwargs = func_kwargs.pop("kwargs", {})
    kwargs.update(func_kwargs)
    allow_error = func_kwargs.pop("allow_error", False)

    proc = None
    t_start = time.time()

    while True:
        t_now = time.time()
        if t_now - t_start > timeout and timeout > 0:
            raise Exception("Wait timed out" + repr((t_now - t_start, timeout)))
        if proc is None:
            proc = shell.spawn(*args, allow_error=True, **kwargs)
        if proc.is_running():
            time.sleep(2)
        else:
            res = proc.wait_for_result()
            if res.return_code == 0:
                return res
            elif not allow_error:
                if rerun_error:
                    proc = None
                    time.sleep(2)
                else:
                    raise res.to_error()
            else:
                return res


def wait_connection(shell, address, port, timeout=0):
    """Wait until we can connect to given address/port."""
    cmd = ["sh", "-c", "echo | nc %s %d" % (address, port)]
    wait_run(shell, cmd, timeout=timeout, rerun_error=True)


def wait_ssh(shell, timeout=0):
    """Wait until we can ssh through given shell."""
    if isinstance(shell, LocalShell):
        return
    local = LocalShell()
    cmd = [
        "sshpass", "-p", shell._password,
        "ssh",
        "-o", "ConnectTimeout=1",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(shell._port), shell.username+"@"+shell.hostname,
        "true",
    ]
    wait_run(local, cmd, timeout=timeout, rerun_error=True)


def print_stringio(obj):
    """Print contents of a StringIO object as they become available.

    Useful in combination with `wait_stringio` to print an output while
    processing it.

    Examples
    --------
    >>> stdout = StringIO.StringIO()
    >>> thread.start_new_thread(print_stringio, (stdout,))
    >>> proc.run(["sh", "-c", "sleep 1 ; echo start ; sleep 2; echo end ; sleep 1"], stdout=stdout)
    start
    end

    See also
    --------
    wait_stringio

    """
    if not isinstance(obj, StringIO.StringIO):
        raise TypeError("expected a StringIO object")
    seen = 0
    while True:
        time.sleep(0.5)
        contents = obj.getvalue()
        missing = contents[seen:]
        print(missing, end="")
        seen += len(missing)


def wait_stringio(obj, pattern):
    """Wait until a StringIO's contents match the given regex.

    Useful to trigger operations when a process generates certain output.

    Examples
    --------
    Count time between two lines of output in a process:

    >>> stdout = StringIO.StringIO()
    >>> def timer(obj):
            wait_stringio("^start$")
            t_start = time.time()
            wait_stringio("^end$")
            t_end = time.time()
            print("time:", int(t_end - t_start))
    >>> thread.start_new_thread(bench_detector, (stdout,))
    >>> proc.run(["sh", "-c", "sleep 1 ; echo start ; sleep 2; echo end ; sleep 1"], stdout=stdout)
    time: 2

    See also
    --------
    print_stringio

    """
    if not isinstance(obj, StringIO.StringIO):
        raise TypeError("expected a StringIO object")
    cre = re.compile(pattern, re.MULTILINE)
    while True:
        time.sleep(0.5)
        contents = obj.getvalue()
        match = cre.findall(contents)
        if len(match) > 0:
            return


def rsync(src_shell, src_path, dst_shell, dst_path, args=[]):
    """Synchronize two directories using rsync.

    Parameters
    ----------
    src_shell
        Source shell.
    src_path
        Source directory.
    dst_shell
        Destination shell.
    dst_path
        Destination directory.
    args : list of str, optional
        Additional arguments to rsync. Default is none.

    """
    def is_local_shell(shell):
        if spur_patch:
            return (isinstance(shell, spur_watchdog_patch.LocalShell)
                    or isinstance(shell, spur.LocalShell))
        else:
            return isinstance(shell, spur.LocalShell)
    if not is_local_shell(src_shell) and not is_local_shell(dst_shell):
        raise Exception("rsync cannot work with two remote shells")

    local = LocalShell()

    ssh_port = 22
    cmd_pass = []
    if is_local_shell(src_shell):
        cmd_src = [src_path]
    else:
        ssh_port = src_shell._port
        if src_shell._password is not None:
            cmd_pass = ["sshpass", "-p", src_shell._password]
        cmd_src = ["%s@%s:%s" % (src_shell.username, src_shell.hostname, src_path)]
    if is_local_shell(dst_shell):
        cmd_dst = [dst_path]
    else:
        ssh_port = dst_shell._port
        if dst_shell._password is not None:
            cmd_pass = ["sshpass", "-p", dst_shell._password]
        cmd_dst = ["%s@%s:%s" % (dst_shell.username, dst_shell.hostname, dst_path)]

    cmd = []
    cmd += cmd_pass
    cmd += ["rsync", "-az"]
    cmd += ["-e", "ssh -p %d -o StrictHostKeyChecking=no" % ssh_port]
    cmd += cmd_src
    cmd += cmd_dst
    cmd += args
    local.run(cmd)


def check_kernel_version(shell, target, fail=True):
    """Check that a specific linux kernel version is installed.

    Parameters
    ----------
    shell
        Target shell.
    target : str
        Target kernel version.
    fail : bool, optional
        Whether to raise an exception when a different version is
        installed. Default is True.

    Returns
    -------
    bool
        Whether the target kernel version is installed.

    """
    res = shell.run(["uname", "-r"])
    current = res.output.split("\n")[0]
    if current == target:
        return True
    else:
        if fail:
            raise Exception("Invalid kernel version: target=%s current=%s" % (target, current))
        return False


def install_kernel_version(shell, target, base_path):
    """Install and reboot into a given linux kernel version if it is not the current.

    Parameters
    ----------
    shell
        Target shell.
    target : str
        Target kernel version.
    base_path : str
        Base directory in target shell where kernel packages can be installed
        from.

    """
    if check_kernel_version(shell, target, fail=False):
        return

    for name in ["linux-image-%(v)s_%(v)s-*.deb",
                 "linux-headers-%(v)s_%(v)s-*.deb",
                 "linux-libc-dev_%(v)s-*.deb"]:
        name = os.path.join(base_path, name % {"v": target})
        res = shell.run(["sh", "-c", "ls %s" % name])
        files = res.output.split("\n")
        for path in files:
            if path == "":
                continue
            logger.warn("Installing %s..." % path)
            shell.run(["sudo", "dpkg", "-i", path])

    res = shell.run(["grep", "-E", "menuentry .* %s" % target, "/boot/grub/grub.cfg"])
    grub_ids = res.output.split("\n")
    pattern = r" '([a-z0-9.-]+-%s-[a-z0-9.-]+)' {" % re.escape(target)
    grub_id = re.search(pattern, grub_ids[0]).group(1)

    logger.warn("Updating GRUB %s..." % path)
    shell.run(["sudo", "sed", "-i", "-e", "s/^GRUB_DEFAULT=/GRUB_DEFAULT=\"saved\"/", "/etc/default/grub"])
    shell.run(["sudo", "update-grub"])
    shell.run(["sudo", "grub-set-default", grub_id])

    logger.warn("Rebooting into new kernel...")
    shell.run(["sudo", "reboot"], allow_error=True)
    wait_ssh(shell)

    check_kernel_version(shell, target)


def check_kernel_cmdline(shell, arg):
    """Check the linux kernel was booted with the given commandline.

    Parameters
    ----------
    shell
        Target shell.
    arg : str
        Command line argument to check.

    """
    shell.run(["grep", arg, "/proc/cmdline"])


def check_module_param(shell, module, param, value, fail=True):
    """Check that a linux kernel module was loaded with the given parameter value.

    Parameters
    ----------
    shell
        Target shell.
    module : str
        Module name.
    param : str
        Module name.
    value
        Module value (will be coverted to str).
    fail : bool, optional
        Raise an exception if the value is not equal. Default is True.

    Returns
    -------
    bool
        Whether the given kernel module was loaded with the given parameter
        value.

    """
    with shell.open("/sys/module/%s/parameters/%s" % (module, param), "r") as f:
        f_val = f.read().split("\n")[0]
        if f_val != value:
            if fail:
                raise Exception("invalid kernel parameter value: target=%s current=%s" % (value, f_val))
            return False
        else:
            return True


def set_freq(shell, linux_build=None, cpupower_path=None, freq="max"):
    """Set frequency scaling.

    Parameters
    ----------
    shell
        Target shell.
    linux_build : str, optional
        Path to linux' build directory to build cpupower (ignores
        `cpupower_path`). Default is use the system's cpupower tool.
    cpupower_path : str, optional
        Path to cpupower tool. Default is use the cpupower tool in the PATH.
    freq : str, optional
        Frequency to set in GHz. Default is use maximum frequency.

    """
    if linux_build is None:
        cpupower_base = ""
        if cpupower_path is None:
            install(shell, "linux-tools-common")
            cpupower_path = "cpupower"
    else:
        cpupower_base = linux_build
        cpupower_path = os.path.join(linux_build, "cpupower")
        try:
            shell.run(["ls", cpupower_path])
        except spur.results.RunProcessError:
            shell.run(["make", "-C", linux_build, "tools/cpupower"])

    check_kernel_cmdline(shell, "intel_pstate=disable")

    if freq == "max":
        max_freq = shell.run([
            "sh", "-c",
            "sudo LD_LIBRARY_PATH=%s %s frequency-info | grep 'hardware limits' | sed -e 's/.* - \\(.*\\) GHz/\\1/'" % (
                cpupower_base, cpupower_path)])
        freq = max_freq.output[:-1]

    shell.run(["sudo",
               "LD_LIBRARY_PATH=%s" % cpupower_base, cpupower_path,
               "-c", "all", "frequency-set", "--governor", "userspace"])
    shell.run(["sudo",
               "LD_LIBRARY_PATH=%s" % cpupower_base, cpupower_path,
               "-c", "all", "frequency-set", "--freq", freq + "GHz"])


def _get_mask(cpu_list):
    mask = 0
    for cpu in cpu_list:
        mask += 1 << cpu
    return mask

def set_manual_irqs(shell, *irqs, **kwargs):
    """Make irqbalance ignore the given IRQs, and instead set their SMP affinity.

    Parameters
    ----------
    shell
        Target system.
    irqs
        IRQ descriptors.
    ignore_errors : bool, optional
        Ignore errors when manually setting an IRQ's SMP affinity. Implies that
        irqbalance will manage that IRQ. Default is False.
    irqbalance_banned_cpus : list of int, optional
        CPUs that irqbalance should not use for balancing.
    irqbalance_args : list of str, optional
        Additional arguments to irqbalance.

    Each descriptor in `irqs` is a three-element tuple:
    * Type: either ``irq`` for the first column in /proc/interrupts, or
            ``descr`` for the interrupt description after the per-CPU counts.
    * Regex: a regular expression to apply to the fields above, or `True` to
             apply to all values (a shorthand to the regex ".*"), or an `int` (a
             shorthand to the regex "^int_value$").
    * SMP affinity: list of cpu numbers to set as the IRQ's affinity; if `True`
                    is used instead, let irqbalance manage this IRQ.

    All matching descriptors are applied in order for each IRQ. If no descriptor
    matches, or the last matching descriptor has `True` as its affinity value,
    the IRQ will be managed by irqbalance as before.

    Returns
    -------
    The irqbalance process.

    """
    ignore_errors = kwargs.pop("ignore_errors", False)
    irqbalance_args = kwargs.pop("irqbalance_args", [])
    irqbalance_banned_cpus = kwargs.pop("irqbalance_banned_cpus", [])
    irqbalance_banned_cpus_mask = _get_mask(irqbalance_banned_cpus)
    if len(kwargs) > 0:
        raise Exception("unknown argument: %s" % list(kwargs.keys())[0])

    irqs_parsed = []
    for arg_irq in irqs:
        if len(arg_irq) != 3:
            raise ValueError("wrong IRQ descriptor: %s" % repr(arg_irq))

        irq_type, irq_re, irq_cpus = arg_irq

        if isinstance(irq_re, int):
            irq_re = "^%d$" % irq_re
        if not isinstance(irq_re, bool) and not isinstance(irq_re, six.string_types):
            raise TypeError("wrong IRQ descriptor regex: %s" % str(irq_re))
        if not isinstance(irq_re, bool):
            irq_re = re.compile(irq_re)

        if (not isinstance(irq_cpus, bool) and (isinstance(irq_cpus, six.string_types) or
                                              not isinstance(irq_cpus, collections.Iterable))):
            raise TypeError("wrong IRQ descriptor CPU list: %s" % str(irq_cpus))

        if irq_type not in ["irq", "descr"]:
            raise ValueError("wrong IRQ descriptor type: %s" % str(irq_type))

        irqs_parsed.append((irq_type, irq_re, irq_cpus))

    irq_manual = []
    irqbalance_banned = set()

    cre = re.compile(r"(?P<irq>[^:]+):(?:\s+[0-9]+)+\s+(?P<descr>.*)")
    with shell.open("/proc/interrupts") as f:
        for line in f.read().split("\n"):
            match = cre.match(line)
            if match is None:
                continue

            irq = match.groupdict()["irq"].strip()
            descr = match.groupdict()["descr"].strip()

            cpus = True

            for irq_type, irq_cre, irq_cpus in irqs_parsed:
                if irq_type == "irq":
                    if irq_cre == True or irq_cre.match(irq):
                        cpus = irq_cpus
                elif irq_type == "descr":
                    if irq_cre == True or irq_cre.match(descr):
                        cpus = irq_cpus
                else:
                    assert False, irq_type

            if cpus != True:
                irq_manual.append((irq, cpus))
                irqbalance_banned.add(irq)

    for irq, cpus in irq_manual:
        mask = _get_mask(cpus)
        try:
            shell.run(["sudo", "sh", "-c",
                       "echo %x > /proc/irq/%s/smp_affinity" % (irq, mask)])
        except:
            if ignore_errors:
                irqbalance_banned.remove(irq)
            else:
                raise

    shell.run(["sudo", "service", "irqbalance", "stop"])
    proc = shell.spawn(["sudo", "IRQBALANCE_BANNED_CPUS=%x" % irqbalance_banned_cpus_mask,
                        "irqbalance"] + irqbalance_args +
                       ["--banirq=%s" % banned
                        for banned in irqbalance_banned])
    return proc


def get_process_pids(shell, pid, filter=None):
    """Get pids of all threads in a given process.

    Parameters
    ----------
    shell
        Target shell.
    pid : int
        Target process pid.
    filter : str, optional
        Return pids that contain given process name. Default is all pids.

    Returns
    -------
    list of int
        List of the selected process pids.

    """
    pid_tree = shell.run(["pstree", "-pal", str(pid)])
    lines = pid_tree.output.split("\n")
    res = []
    for line in lines:
        if line == "":
            continue
        if filter and filter not in line:
            continue
        thread_pid = line.split(",")[1].split(" ")[0]
        res.append(int(thread_pid))
    return res


def get_cpus(shell, node=None, package=None, core=None, pu=None, cgroup=None):
    """Get a set of all physical CPU indexes in the system.

    It uses the hwloc program to report available CPUs.

    Parameters
    ----------
    shell
        Target shell.
    node : int or list of int, optional
        NUMA nodes to check.
    package : int or list of int, optional
        Core packages to check.
    core : int or list of int, optional
        Cores to check.
    pu : int or list of int, optional
        PUs to check.
    cgroup : str, optional

    Returns
    -------
    set of int
        Physical CPU indexes (as used by Linux).

    """
    cmd = ["hwloc-ls", "--no-caches", "-c"]
    if cgroup is not None:
        cmd = ["sudo", "cgexec", "-g", cgroup] + cmd
    res = shell.run(cmd)
    lines = res.output.split("\n")

    # parse output

    def get_mask(line):
        parts = line.split("cpuset=")
        mask = parts[-1]
        return int(mask, base=16)

    def get_set(line):
        mask = get_mask(line)
        bin_mask = bin(mask)[2:]
        res = set()
        for idx, i in enumerate(reversed(bin_mask)):
            if i == "1":
                res.add(idx)
        return res

    root = []
    for line in lines:
        parts = line.strip().split(" ")
        if parts[0] == "NUMANode":
            target = root
            target.append([])
        elif parts[0] == "Package":
            target = root[-1]
            target.append([])
        elif parts[0] == "Core":
            target = root[-1]
            target = target[-1]
            target.append([])
        elif parts[0] == "PU":
            pus = get_set(parts[-1])
            target = root[-1]
            target = target[-1]
            target = target[-1]
            target.append(pus)

    # compute PU set

    def reduce_or(arg):
        res = set()
        for a in arg:
            res |= a
        return res

    def collect(level, target, parents):
        indexes = target[0]

        if indexes is None:
            indexes = range(len(level))
        elif isinstance(indexes, int):
            indexes = [indexes]

        if isinstance(indexes, collections.Iterable):
            res = set()
            for idx in indexes:
                try:
                    s = level[idx]
                except IndexError:
                    parent_names = ["node", "package", "core", "pu"]
                    path = zip(parent_names, parents+[idx])
                    raise Exception("invalid cpu path: %s" % " ".join(
                        "%s=%d" % (n, i) for n, i in path))
                else:
                    if len(target) == 1:
                        res |= s
                    else:
                        res |= collect(s, target[1:], parents+[idx])
            return res
        else:
            assert False, value

    res = collect(root, [node, package, core, pu], [])
    return res


def pin_pid(shell, pid, cpus, allow_error=False, stderr=None):
    """Pin pid to given physical CPU list.

    Parameters
    ----------
    shell
        Target shell.
    pid : int
        Target pid to pin.
    cpus : list of int
        Physical CPUs to pin the pid to.

    """
    shell.run(["sudo", "taskset", "-p",
               "-c", ",".join(str(c) for c in cpus), str(pid)],
              allow_error=allow_error, stderr=stderr)


def cgroup_create(shell, subsystem, name, **kwargs):
    """Create a cgroup for given subsystem.

    Parameters
    ----------
    shell
        Target shell.
    subsystem : str
        Cgroup subsystem to configure.
    name : str
        Cgroup name.
    kwargs : dict
        Subsystem parameters to set. Lists are comma-concatenated, all elements
        are transformed to str.

    """
    shell.run(["sudo", "cgcreate", "-g", subsystem+":"+name])
    for key, val in kwargs.items():
        if isinstance(val, six.string_types) or not isinstance(val, collections.Iterable):
            val = [val]
        val = ",".join(str(v) for v in val)
        shell.run(["sudo", "cgset", "-r", "%s.%s=%s" % (subsystem, key, val), name])


def cgroup_move(shell, subsystem, name, pids):
    """Move pids to a cgroup.

    Parameters
    ----------
    shell
        Target shell.
    subsystem : str
        Cgroup subsystem.
    name : str
        Cgroup name.
    pids : pid or list of pid
        Pids to move to the cgroup. All elements are transformed to str.

    """
    if isinstance(pids, six.string_types) or not isinstance(pids, collections.Iterable):
        pids = [pids]
    shell.run(["sudo", "cgclassify", "-g", "%s:/%s" % (subsystem, name)] +
              [str(p) for p in pids
               if str(p) != ""])


def cgroup_move_all(shell, subsystem, name):
    """Move all processes in the system into given cgroup.

    Ignores all errors while performing the move.

    Parameters
    ----------
    shell
        Target shell.
    subsystem : str
        Cgroup subsystem.
    name : str
        Cgroup name.

    """
    tasks_paths = shell.run(["find", "/sys/fs/cgroup/pids", "-name", "tasks"]).output
    tasks_paths = [p for p in tasks_paths.split("\n")
                   if p != ""]
    for tasks_path in tasks_paths:
        try:
            with shell.open(tasks_path, "r") as f:
                pids = [p for p in f.read().split("\n")
                        if p != ""]
                if len(pids) > 0:
                    cgroup_move(shell, subsystem, name, pids)
        except:
            pass


def cgroup_move_ssh(shell, subsystem, name):
    """Move systemd's ssh service to the given cgroup.

    Parameters
    ----------
    shell
        Target shell.
    subsystem : str
        Cgroup subsystem.
    name : str
        Cgroup name.

    """
    with shell.open("/sys/fs/cgroup/systemd/system.slice/ssh.service/tasks", "r") as f:
        pids = [p for p in f.read().split("\n")
                if p != ""]
        if len(pids) > 0:
            try:
                cgroup_move(shell, subsystem, name, pids)
            except:
                pass

    # NOTE: ensure connection is reopened
    shell._client.close()
    shell._client = None

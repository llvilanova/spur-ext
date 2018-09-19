# spur-ext

A small Python package with various functions that help to orchestrate tasks
using [spur](https://pypi.org/project/spur). It was originally written to
orchestrate experiments in a group of machines.

It includes tasks such as:
* Installing packages
* Bulk copying files or directories
* Running remote commands in parallel
* CPU discovery
* Listing PIDs of the system
* Pinning threads
* Managing cgroups


It is recommended to also get https://github.com/llvilanova/spur-watchdog-patch
in order to make spur handle process deaths more gracefully.

The functions here are only tested in Debian/Ubuntu systems, but patches are welcome.

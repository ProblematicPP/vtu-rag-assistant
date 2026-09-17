# Module 1: Introduction to Operating Systems

These are short original demo notes used to exercise the ingestion pipeline.
Replace or extend them with your own module notes.

## 1.1 What an Operating System Does

An operating system (OS) is the software layer that sits between user programs and the
computer hardware. It has two broad roles. As a resource allocator, it decides how the CPU,
memory, storage and I/O devices are shared among competing programs so that the system is
used efficiently and fairly. As a control program, it supervises the execution of user
programs to prevent errors and improper use of the hardware.

From the user's point of view the goal is convenience: a personal computer's OS is designed
mostly for ease of use, with some attention to performance and little to resource
utilisation. A large shared server, by contrast, is tuned to keep every user's work moving
and to use the hardware as fully as possible. Embedded systems in appliances and vehicles
often have little or no user interface and are designed to run with minimal intervention.

The one program that runs at all times is usually called the kernel. Alongside the kernel
there are system programs, which ship with the OS but are not part of the kernel, and
application programs, which are everything else.

## 1.2 Computer System Organization

A modern general-purpose computer consists of one or more CPUs and a number of device
controllers connected through a common bus that provides access to shared memory. Each
device controller is in charge of a specific type of device and has a local buffer and a set
of special-purpose registers. The operating system has a device driver for each controller;
the driver understands the controller and gives the rest of the OS a uniform interface.

### Interrupts

Hardware signals the CPU that an event needs attention by raising an interrupt. When the CPU
is interrupted it stops what it is doing, saves the state of the interrupted computation, and
transfers control to a fixed location that holds the interrupt service routine. An interrupt
vector — a table of addresses of service routines indexed by device number — lets the CPU
dispatch to the right handler quickly. After the routine finishes, the saved state is
restored and the interrupted computation resumes as though nothing happened.

Software can also trigger an interrupt, called a trap or exception, either because of an
error such as division by zero or invalid memory access, or because a program requests an
operating-system service through a system call.

### Storage Structure

Main memory (RAM) is the only large storage area that the CPU can access directly, but it is
volatile and too small to hold all programs and data permanently. Secondary storage, such as
hard disks and solid-state drives, extends main memory with large non-volatile capacity. The
storage hierarchy trades off speed, cost and volatility: registers and cache are fastest and
most expensive per byte, followed by main memory, SSDs, hard disks and finally optical or
tape storage. Caching — keeping a copy of frequently used data in faster storage — is an
important principle applied at many levels of this hierarchy.

## 1.3 Dual-Mode Operation

To protect the system from faulty or malicious programs, hardware provides at least two modes
of operation, distinguished by a mode bit: user mode and kernel mode (also called supervisor,
system or privileged mode). The system boots in kernel mode, loads the OS, and then starts
user applications in user mode. Whenever a trap or interrupt occurs, the hardware switches to
kernel mode; the OS switches back to user mode before passing control to a user program.

Instructions that could harm the system are designated privileged and may only be executed in
kernel mode. Examples include issuing I/O requests, changing memory-management registers and
setting the timer. If a user program attempts a privileged instruction, the hardware traps to
the operating system, which treats it as an error.

A timer prevents a user program from running forever without returning control. The OS sets
the timer to interrupt after a specified period; when it fires, control transfers to the OS,
which can terminate the program or give the CPU to another process.

## 1.4 System Calls

System calls provide the interface between a running program and the operating system. They
are generally made available through an application programming interface (API), such as the
POSIX API on UNIX-like systems or the Windows API. Programmers call API functions, which in
turn invoke the actual system calls; this keeps programs portable and hides details.

Three general methods pass parameters to the OS: in CPU registers; in a block or table in
memory whose address is passed in a register; or pushed onto the stack and popped by the OS.
System calls can be grouped into six major categories: process control, file management,
device management, information maintenance, communications and protection.

## 1.5 Operating System Structures

A monolithic structure places all kernel functionality in a single static binary running in a
single address space. It is fast, because there is little overhead in the system-call
interface, but difficult to implement and extend. The layered approach breaks the OS into
levels, where each layer uses only the services of lower layers; this simplifies debugging but
makes it hard to define the layers and can hurt performance.

The microkernel approach removes all non-essential components from the kernel and implements
them as user-level programs that communicate by message passing. This makes the system easier
to extend and more reliable, since a failing service does not crash the kernel, at the cost of
message-passing overhead. Loadable kernel modules combine benefits of both: a core kernel with
dynamically linked components for file systems, drivers and scheduling classes. Most modern
systems, including Linux, Windows and macOS, are hybrids of these structures.
